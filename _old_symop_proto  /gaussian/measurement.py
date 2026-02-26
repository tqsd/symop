from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from symop_proto.core.protocols import ModeOpProto
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.ops.common import (
    alpha_from_quadrature_mean,
    as_index_list,
)
from symop_proto.gaussian.ops.measurement import (
    QuadratureConditioningResult,
)
from symop_proto.gaussian.ops.measurement import (
    condition_on_quadratures as _condition_on_quadratures_ops,
)
from symop_proto.gaussian.ops.measurement import (
    quadrature_indices as _quadrature_indices_ops,
)


@dataclass(frozen=True)
class GaussianMeasurementResult:
    r"""High-level Gaussian measurement result.

    This wraps the ops-level quadrature conditioning result and additionally
    provides a posterior :class:`~symop_proto.gaussian.core.GaussianCore`
    on a valid *mode* subsystem.

    Attributes
    ----------
    outcome:
        Real measurement outcome vector ``y`` (shape ``(m,)``).
    log_prob:
        Log probability density ``log p(y)`` under the measurement model.
    core_post:
        Posterior Gaussian core on the kept *modes*.
    ops:
        The underlying ops-level conditioning result (quadrature domain).

    Notes
    -----
    ``log_prob`` is a log *density*, and is therefore often negative.

    """

    outcome: np.ndarray
    log_prob: float
    core_post: GaussianCore
    ops: QuadratureConditioningResult


def _mode_quadrature_index_pairs(n: int, mode_idx: Sequence[int]) -> NDArray[np.int_]:
    out: list[int] = []
    for i in mode_idx:
        out.append(int(i))
    for i in mode_idx:
        out.append(int(i) + n)
    return np.asarray(out, dtype=int)


def _mode_quadrature_indices(
    n: int, mode_idx: Sequence[int], *, which: str
) -> NDArray[np.int_]:
    which_s = str(which)
    idx_list = [int(i) for i in mode_idx]
    if which_s == "x":
        return np.asarray(idx_list, dtype=int)
    if which_s == "p":
        return np.asarray([i + n for i in idx_list], dtype=int)
    if which_s == "xp":
        return _mode_quadrature_index_pairs(n, idx_list)
    raise ValueError("which must be one of: 'x', 'p', 'xp'")


def _complement_modes(n: int, measured: Sequence[int]) -> tuple[int, ...]:
    mset = {int(i) for i in measured}
    return tuple(i for i in range(n) if i not in mset)


def _extract_subposterior(
    res: QuadratureConditioningResult,
    *,
    want_q_idx_global: NDArray[np.int_],
) -> tuple[np.ndarray, np.ndarray]:
    r"""Extract (d,V) for a desired subset of global quadrature indices from an ops
    posterior.

    The ops result stores posterior moments on ``keep_q_idx`` (global indices).
    We select those entries that match ``want_q_idx_global`` and preserve the
    ordering of ``want_q_idx_global``.
    """
    keep = res.keep_q_idx
    want = np.asarray(want_q_idx_global, dtype=int).reshape(-1)

    pos: list[int] = []
    for q in want.tolist():
        hits = np.nonzero(keep == int(q))[0]
        if hits.size != 1:
            raise ValueError(
                "Requested quadrature index is not present in the ops posterior. "
                "This usually indicates an inconsistent keep/measured partition."
            )
        pos.append(int(hits[0]))

    pos_arr = np.asarray(pos, dtype=int)
    d2 = res.d_post[pos_arr].copy()
    V2 = res.V_post[np.ix_(pos_arr, pos_arr)].copy()
    return d2, V2


def _uncentered_moments_from_quadratures(
    *,
    basis_gram: np.ndarray,
    d: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Convert quadrature moments to ladder moments (alpha, N, M) on the same basis.

    Quadrature ordering is

    .. math::

        R = (x_1,\ldots,x_n,p_1,\ldots,p_n)^T.

    Let

    .. math::

        V_{kl} = \tfrac{1}{2}\langle \{\Delta R_k, \Delta R_l\}\rangle

    be the symmetrized covariance, and let the commutator matrix be

    .. math::

        [R_k, R_l] = i \Omega_{kl}.

    Then the *unsymmetrized* second moment matrix of centered quadratures is

    .. math::

        \langle \Delta R\, \Delta R^T \rangle = V + \tfrac{i}{2}\Omega.

    With ladder operators related by

    .. math::

        a = \frac{x + i p}{\sqrt{2}}, \qquad
        a^\dagger = \frac{x - i p}{\sqrt{2}},

    define the stacked ladder vector

    .. math::

        r = (a_1,\ldots,a_n,a_1^\dagger,\ldots,a_n^\dagger)^T = S R

    with

    .. math::

        S = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
        I & iI \\
        I & -iI
        \end{pmatrix}.

    Then

    .. math::

        K_r = \langle \Delta r\, \Delta r^T \rangle = S
              \left(V + \tfrac{i}{2}\Omega\right) S^T

    has blocks

    .. math::

        K_r =
        \begin{pmatrix}
        M_0 & G + N_0^T \\
        N_0 & M_0^\dagger
        \end{pmatrix}.

    Finally, uncentered moments are

    .. math::

        N = N_0 + \alpha^* \alpha^T,\qquad
        M = M_0 + \alpha \alpha^T.

    Returns
    -------
    alpha, N, M
        Arrays of shapes ``(n,)``, ``(n,n)``, ``(n,n)``.

    """
    d = np.asarray(d, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    if V.shape != (d.shape[0], d.shape[0]):
        raise ValueError("d and V have incompatible shapes")

    n2 = int(d.shape[0])
    if n2 % 2 != 0:
        raise ValueError("quadrature mean must have even length")
    n = n2 // 2

    G = np.asarray(basis_gram, dtype=complex)
    if G.shape != (n, n):
        raise ValueError("basis_gram has incompatible shape")

    # Build Omega from G in the same convention as GaussianCore.symplectic_form.
    Omega = np.zeros((2 * n, 2 * n), dtype=complex)
    Omega[0:n, n : 2 * n] = G
    Omega[n : 2 * n, 0:n] = -G.T

    covRR = V.astype(complex) + 0.5j * Omega

    I = np.eye(n, dtype=complex)
    S = (1.0 / np.sqrt(2.0)) * np.block([[I, 1j * I], [I, -1j * I]])

    Kr = S @ covRR @ S.T

    M0 = Kr[0:n, 0:n]
    N0 = Kr[n : 2 * n, 0:n]

    alpha = alpha_from_quadrature_mean(d)

    N = N0 + np.outer(np.conjugate(alpha), alpha)
    M = M0 + np.outer(alpha, alpha)

    return alpha.astype(complex), N.astype(complex), M.astype(complex)


def _core_from_quadrature_moments(
    core_ref: GaussianCore,
    *,
    keep_modes: tuple[ModeOpProto, ...],
    d: np.ndarray,
    V: np.ndarray,
) -> GaussianCore:
    core_keep = core_ref.keep(keep_modes)
    alpha, N, M = _uncentered_moments_from_quadratures(
        basis_gram=core_keep.basis.gram,
        d=d,
        V=V,
    )
    return GaussianCore.from_moments(core_keep.basis, alpha=alpha, N=N, M=M)


def condition_quadratures(
    core: GaussianCore,
    *,
    meas_q_idx: NDArray[np.int_],
    outcome: np.ndarray | None = None,
    Vm: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> GaussianMeasurementResult:
    r"""Condition on an arbitrary quadrature measurement and return a mode-valid core
    if possible.

    This is a higher-level wrapper around
    :func:`symop_proto.gaussian.ops.measurement.condition_on_quadratures`. It
    attempts to convert the posterior quadrature moments into a
    :class:`~symop_proto.gaussian.core.GaussianCore`.

    Important restriction
    ---------------------
    A :class:`GaussianCore` represents moments of *modes*, i.e. each kept mode
    must contribute both its :math:`x` and :math:`p` quadratures. Therefore,
    this wrapper only succeeds if the ops posterior keeps a set of quadratures
    that corresponds exactly to a subset of full modes.

    If you want homodyne on a mode (measure one quadrature and discard the mode),
    use :func:`homodyne_x` or :func:`homodyne_p`.

    Parameters
    ----------
    core:
        Input Gaussian state.
    meas_q_idx:
        Global quadrature indices into
        :math:`R=(x_1,\ldots,x_n,p_1,\ldots,p_n)^T`.
    outcome:
        If provided, use this outcome. Otherwise sample.
    Vm:
        Measurement noise covariance (defaults to 0).
    rng:
        RNG used if sampling.

    Returns
    -------
    GaussianMeasurementResult
        Outcome + log density + posterior GaussianCore on kept modes.

    Raises
    ------
    ValueError
        If the posterior does not correspond to a subset of full modes.

    Examples
    --------
    Heterodyne a single mode (measure both quadratures) and keep nothing:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.channel import Displacement
        from symop_proto.gaussian.measurement import condition_quadratures
        from symop_proto.gaussian.ops.measurement import quadrature_indices

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])
        core = GaussianCore.vacuum(B)
        core = Displacement(modes=(m,), beta=np.array([0.2 + 0.3j], dtype=complex)).apply(core)

        q_idx = quadrature_indices(core, (m,), which="xp")
        res = condition_quadratures(core, meas_q_idx=q_idx, Vm=0.5 * np.eye(2))
        print("outcome:", res.outcome)
        print("kept modes:", res.core_post.basis.n)

    """
    res_ops = _condition_on_quadratures_ops(
        core,
        meas_q_idx=np.asarray(meas_q_idx, dtype=int),
        outcome=outcome,
        Vm=Vm,
        rng=rng,
    )

    # Determine whether keep_q_idx corresponds to full modes.
    n = core.basis.n
    keep_q = res_ops.keep_q_idx.astype(int)

    # A mode i is kept iff both i and i+n appear.
    keep_modes_idx: list[int] = []
    keep_set = {int(x) for x in keep_q.tolist()}
    for i in range(n):
        if (i in keep_set) and ((i + n) in keep_set):
            keep_modes_idx.append(i)

    # Check that there are no "dangling" quadratures that would break mode validity.
    want_keep_q = _mode_quadrature_index_pairs(n, keep_modes_idx)
    if set(int(x) for x in want_keep_q.tolist()) != keep_set:
        raise ValueError(
            "Posterior keeps a set of quadratures that does not correspond to full modes. "
            "Use homodyne_x/homodyne_p if you are measuring a single quadrature and discarding a mode."
        )

    keep_modes: tuple[ModeOpProto, ...] = tuple(
        core.basis.modes[i] for i in keep_modes_idx
    )
    core_post = _core_from_quadrature_moments(
        core,
        keep_modes=keep_modes,
        d=res_ops.d_post,
        V=res_ops.V_post,
    )

    return GaussianMeasurementResult(
        outcome=res_ops.outcome,
        log_prob=res_ops.log_prob,
        core_post=core_post,
        ops=res_ops,
    )


def heterodyne(
    core: GaussianCore,
    *,
    modes: Sequence[ModeOpProto],
    outcome: np.ndarray | None = None,
    meas_var: float = 0.5,
    rng: np.random.Generator | None = None,
) -> GaussianMeasurementResult:
    r"""Heterodyne measurement on modes (measure both quadratures and discard them).

    This is implemented as a noisy measurement of ``(x,p)`` with isotropic noise

    .. math::

        V_m = \sigma^2 I.

    A common convention for heterodyne adds vacuum noise with
    :math:`\sigma^2 = 1/2`, hence the default ``meas_var=0.5``.

    Parameters
    ----------
    core:
        Input Gaussian state.
    modes:
        Modes to measure (discarded after measurement).
    outcome:
        If provided, use this outcome (length ``2*len(modes)``). Else sample.
    meas_var:
        Measurement noise variance (>= 0).
    rng:
        RNG used if sampling.

    Returns
    -------
    GaussianMeasurementResult
        Posterior GaussianCore on the remaining modes.

    Examples
    --------
    Heterodyne one mode of a two-mode squeezed state (keeps the other mode):

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.bogoliubov import TwoModeSqueezer
        from symop_proto.gaussian.measurement import heterodyne

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.vacuum(B)
        core = TwoModeSqueezer(mode1=m1, mode2=m2, r=0.6, phi=0.0, check_ccr=True).apply(core)

        res = heterodyne(core, modes=(m2,))
        print("outcome:", res.outcome)
        print("kept modes:", res.core_post.basis.n)
        print("kept alpha:", res.core_post.alpha)

    """
    mv = float(meas_var)
    if mv < 0.0:
        raise ValueError("meas_var must be >= 0")

    q_idx = _quadrature_indices_ops(core, tuple(modes), which="xp")
    Vm = mv * np.eye(int(q_idx.size), dtype=float)

    return condition_quadratures(
        core,
        meas_q_idx=q_idx,
        outcome=outcome,
        Vm=Vm,
        rng=rng,
    )


def homodyne_x(
    core: GaussianCore,
    *,
    modes: Sequence[ModeOpProto],
    outcome: np.ndarray | None = None,
    meas_var: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> GaussianMeasurementResult:
    r"""Homodyne-x measurement on modes (measure x quadrature and discard the modes).

    Implementation strategy
    -----------------------
    We measure only the selected :math:`x` quadratures in the ops layer, then
    *discard the entire measured modes* by dropping the remaining unmeasured
    :math:`p` quadratures from the posterior.

    The ops-level conditioning uses

    .. math::

        y = x_B + \nu, \qquad \nu \sim \mathcal{N}(0, V_m),

    with :math:`V_m = \sigma^2 I`. A tiny positive ``meas_var`` is recommended
    for numerical stability.

    Parameters
    ----------
    core:
        Input Gaussian state.
    modes:
        Modes to measure and discard.
    outcome:
        If provided, use this outcome (length ``len(modes)``). Else sample.
    meas_var:
        Measurement noise variance (>= 0). Use a small value for near-ideal homodyne.
    rng:
        RNG used if sampling.

    Returns
    -------
    GaussianMeasurementResult
        Posterior GaussianCore on the remaining modes.

    Examples
    --------
    Homodyne-x one arm of a two-mode squeezed state (keeps the other arm):

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.bogoliubov import TwoModeSqueezer
        from symop_proto.gaussian.measurement import homodyne_x

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.vacuum(B)
        core = TwoModeSqueezer(mode1=m1, mode2=m2, r=0.6, phi=0.0, check_ccr=True).apply(core)

        res = homodyne_x(core, modes=(m2,), outcome=np.array([0.1]))
        print("log_prob:", res.log_prob)
        print("kept modes:", res.core_post.basis.n)
        print("kept V shape:", res.core_post.quadrature_covariance().shape)

    """
    mv = float(meas_var)
    if mv < 0.0:
        raise ValueError("meas_var must be >= 0")

    idx_m = as_index_list(core, modes)
    n = core.basis.n

    meas_q = _mode_quadrature_indices(n, idx_m, which="x")
    Vm = mv * np.eye(int(meas_q.size), dtype=float)

    res_ops = _condition_on_quadratures_ops(
        core,
        meas_q_idx=meas_q,
        outcome=outcome,
        Vm=Vm,
        rng=rng,
    )

    # Keep only quadratures of the remaining modes (discard measured modes entirely).
    keep_mode_idx = _complement_modes(n, idx_m)
    want_keep_q = _mode_quadrature_index_pairs(n, keep_mode_idx)

    d2, V2 = _extract_subposterior(res_ops, want_q_idx_global=want_keep_q)

    keep_modes: tuple[ModeOpProto, ...] = tuple(
        core.basis.modes[i] for i in keep_mode_idx
    )
    core_post = _core_from_quadrature_moments(
        core,
        keep_modes=keep_modes,
        d=d2,
        V=V2,
    )

    # Replace ops with a trimmed view (still include original ops for debugging).
    trimmed_ops = QuadratureConditioningResult(
        outcome=res_ops.outcome,
        log_prob=res_ops.log_prob,
        keep_q_idx=want_keep_q.copy(),
        meas_q_idx=res_ops.meas_q_idx.copy(),
        d_post=d2,
        V_post=V2,
    )

    return GaussianMeasurementResult(
        outcome=trimmed_ops.outcome,
        log_prob=trimmed_ops.log_prob,
        core_post=core_post,
        ops=trimmed_ops,
    )


def homodyne_p(
    core: GaussianCore,
    *,
    modes: Sequence[ModeOpProto],
    outcome: np.ndarray | None = None,
    meas_var: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> GaussianMeasurementResult:
    r"""Homodyne-p measurement on modes (measure p quadrature and discard the modes).

    This is analogous to :func:`homodyne_x` but measures :math:`p` instead of :math:`x`.
    """
    mv = float(meas_var)
    if mv < 0.0:
        raise ValueError("meas_var must be >= 0")

    idx_m = as_index_list(core, modes)
    n = core.basis.n

    meas_q = _mode_quadrature_indices(n, idx_m, which="p")
    Vm = mv * np.eye(int(meas_q.size), dtype=float)

    res_ops = _condition_on_quadratures_ops(
        core,
        meas_q_idx=meas_q,
        outcome=outcome,
        Vm=Vm,
        rng=rng,
    )

    keep_mode_idx = _complement_modes(n, idx_m)
    want_keep_q = _mode_quadrature_index_pairs(n, keep_mode_idx)

    d2, V2 = _extract_subposterior(res_ops, want_q_idx_global=want_keep_q)

    keep_modes: tuple[ModeOpProto, ...] = tuple(
        core.basis.modes[i] for i in keep_mode_idx
    )
    core_post = _core_from_quadrature_moments(
        core,
        keep_modes=keep_modes,
        d=d2,
        V=V2,
    )

    trimmed_ops = QuadratureConditioningResult(
        outcome=res_ops.outcome,
        log_prob=res_ops.log_prob,
        keep_q_idx=want_keep_q.copy(),
        meas_q_idx=res_ops.meas_q_idx.copy(),
        d_post=d2,
        V_post=V2,
    )

    return GaussianMeasurementResult(
        outcome=trimmed_ops.outcome,
        log_prob=trimmed_ops.log_prob,
        core_post=core_post,
        ops=trimmed_ops,
    )


def condition_on_modes(
    core: GaussianCore,
    *,
    modes: Sequence[ModeOpProto],
    which: str,
    outcome: np.ndarray | None = None,
    meas_var: float = 0.0,
    rng: np.random.Generator | None = None,
) -> GaussianMeasurementResult:
    r"""Convenience wrapper matching the older ops-style API.

    If ``which="xp"``, this is equivalent to :func:`heterodyne` (up to the chosen
    ``meas_var``). If ``which="x"`` or ``which="p"``, this keeps the *unmeasured*
    quadrature of the measured modes in the ops posterior, which is generally not
    representable as a :class:`GaussianCore` on modes. Therefore, this wrapper:

    - uses :func:`heterodyne` for ``which="xp"``
    - uses :func:`homodyne_x` / :func:`homodyne_p` for ``which="x"`` / ``which="p"``
      (measure one quadrature and discard the mode)

    Parameters
    ----------
    which:
        One of ``"x"``, ``"p"``, ``"xp"``.

    Returns
    -------
    GaussianMeasurementResult

    """
    w = str(which)
    if w == "xp":
        return heterodyne(
            core,
            modes=modes,
            outcome=outcome,
            meas_var=meas_var,
            rng=rng,
        )
    if w == "x":
        return homodyne_x(
            core,
            modes=modes,
            outcome=outcome,
            meas_var=meas_var if meas_var > 0.0 else 1e-9,
            rng=rng,
        )
    if w == "p":
        return homodyne_p(
            core,
            modes=modes,
            outcome=outcome,
            meas_var=meas_var if meas_var > 0.0 else 1e-9,
            rng=rng,
        )
    raise ValueError("which must be one of: 'x', 'p', 'xp'")
