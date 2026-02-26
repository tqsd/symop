from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from symop_proto.core.protocols import ModeOpProto
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.ops.common import (
    as_index_list,
    as_real_mat,
    as_real_vec,
    check_is_subset_indices,
    complement_indices,
)


@dataclass(frozen=True)
class QuadratureConditioningResult:
    r"""Result of conditioning a Gaussian state on a Gaussian quadrature measurement.

    This is an "ops-level" result that stays in the quadrature-moment domain.
    A higher-level wrapper can convert ``(d_post, V_post)`` back into a
    :class:`~symop_proto.gaussian.core.GaussianCore` (typically via a
    ``GaussianCore.from_quadrature(...)`` bridge).

    Attributes
    ----------
    outcome:
        Measurement outcome vector ``y`` (real), shape ``(m,)``.
    log_prob:
        Log-probability density ``log p(y)`` under the measurement model.
    keep_q_idx:
        Quadrature indices (into the original ``R``) that are kept (not measured).
    meas_q_idx:
        Quadrature indices (into the original ``R``) that were measured.
    d_post:
        Posterior quadrature mean on the kept subsystem, shape ``(2*n_keep,)``.
    V_post:
        Posterior quadrature covariance on the kept subsystem, shape
        ``(2*n_keep, 2*n_keep)``.

    """

    outcome: np.ndarray
    log_prob: float
    keep_q_idx: np.ndarray
    meas_q_idx: np.ndarray
    d_post: np.ndarray
    V_post: np.ndarray


def quadrature_indices(
    core: GaussianCore, modes: Sequence[ModeOpProto], *, which: str
) -> np.ndarray:
    r"""Build quadrature indices for a mode subset.

    Quadrature ordering is the one used by :meth:`GaussianCore.quadrature_mean`
    and :meth:`GaussianCore.quadrature_covariance`:

    .. math::

        R = (x_1, \ldots, x_n, p_1, \ldots, p_n)^T.

    For a mode subset with mode indices ``idx``:

    - ``which="x"`` returns ``idx``
    - ``which="p"`` returns ``idx + n``
    - ``which="xp"`` returns ``idx`` followed by ``idx + n``

    Parameters
    ----------
    core:
        Gaussian core defining the global basis and ordering.
    modes:
        Modes to index (must exist in ``core.basis``).
    which:
        One of ``"x"``, ``"p"``, ``"xp"``.

    Returns
    -------
    np.ndarray
        Integer array of indices into the quadrature vector ``R``.

    Examples
    --------
    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.ops.measurement import quadrature_indices

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])
        core = GaussianCore.vacuum(B)

        print(quadrature_indices(core, (m2,), which="x"))
        print(quadrature_indices(core, (m2,), which="p"))
        print(quadrature_indices(core, (m2,), which="xp"))

    """
    which_s = str(which)
    idx_list = as_index_list(core, modes)
    n = core.basis.n
    if which_s == "x":
        return np.asarray(idx_list, dtype=int)
    if which_s == "p":
        return np.asarray([i + n for i in idx_list], dtype=int)
    if which_s == "xp":
        return np.asarray(idx_list + [i + n for i in idx_list], dtype=int)
    raise ValueError("which must be one of: 'x', 'p', 'xp'")


def gaussian_logpdf(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    r"""Log-density of a multivariate normal distribution.

    Computes

    .. math::

        \log \mathcal{N}(y \mid \mu, \Sigma)
        = -\frac{1}{2}\left(
            m \log(2\pi) + \log\det\Sigma
            + (y-\mu)^T \Sigma^{-1} (y-\mu)
          \right)

    using a Cholesky factorization for numerical stability.

    Parameters
    ----------
    y:
        Real vector of shape ``(m,)``.
    mean:
        Real vector of shape ``(m,)``.
    cov:
        Real symmetric positive definite matrix of shape ``(m,m)``.

    Returns
    -------
    float
        The log probability density.

    """
    yv = np.asarray(y, dtype=float).reshape(-1)
    mv = np.asarray(mean, dtype=float).reshape(-1)
    if yv.shape != mv.shape:
        raise ValueError(
            f"y and mean must have same shape, got {yv.shape} vs {mv.shape}"
        )

    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != yv.shape[0]:
        raise ValueError("cov has incompatible shape")

    L = np.linalg.cholesky(S)
    diff = yv - mv

    # Solve L z = diff, then quadratic form is z^T z
    z = np.linalg.solve(L, diff)
    quad = float(z.T @ z)

    logdet = float(2.0 * np.sum(np.log(np.diag(L))))
    m = int(yv.shape[0])
    return float(-0.5 * (m * np.log(2.0 * np.pi) + logdet + quad))


def gaussian_sample(
    mean: np.ndarray, cov: np.ndarray, *, rng: np.random.Generator
) -> np.ndarray:
    r"""Sample from a multivariate normal distribution.

    Uses

    .. math::

        y = \mu + L \xi,\quad \xi \sim \mathcal{N}(0, I),

    where :math:`LL^T = \Sigma` is a Cholesky factorization.

    Parameters
    ----------
    mean:
        Real mean vector ``(m,)``.
    cov:
        Real covariance matrix ``(m,m)``.
    rng:
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Real sample of shape ``(m,)``.

    """
    mv = np.asarray(mean, dtype=float).reshape(-1)
    S = np.asarray(cov, dtype=float)
    if S.shape != (mv.shape[0], mv.shape[0]):
        raise ValueError("cov has incompatible shape")

    L = np.linalg.cholesky(S)
    xi = rng.normal(loc=0.0, scale=1.0, size=(mv.shape[0],))
    return mv + L @ xi


def condition_on_quadratures(
    core: GaussianCore,
    *,
    meas_q_idx: NDArray[np.int_],
    outcome: np.ndarray | None = None,
    Vm: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> QuadratureConditioningResult:
    r"""Condition a Gaussian state on a (noisy) quadrature measurement.

    Let the full quadrature vector be

    .. math::

        R = (x_1,\ldots,x_n,p_1,\ldots,p_n)^T,

    with mean :math:`d` and covariance :math:`V` (as provided by the core).

    Choose a measured index set :math:`B` (``meas_q_idx``), and let :math:`A`
    denote the complement (kept subsystem). The measurement model is

    .. math::

        y = R_B + \nu,\quad \nu \sim \mathcal{N}(0, V_m).

    Then the outcome distribution is

    .. math::

        p(y) = \mathcal{N}(y \mid d_B, S),\quad S = V_{BB} + V_m,

    and the conditional (posterior) state on :math:`A` has

    .. math::

        d_A' = d_A + V_{AB} S^{-1} (y - d_B),

    .. math::

        V_{AA}' = V_{AA} - V_{AB} S^{-1} V_{BA}.

    Parameters
    ----------
    core:
        Input Gaussian state.
    meas_q_idx:
        Quadrature indices (into ``R``) that are measured.
    outcome:
        If provided, use this outcome ``y``. If None, sample an outcome.
    Vm:
        Measurement noise covariance. If None, defaults to zeros (idealized),
        but note that ideal homodyne should typically use a small finite
        ``Vm`` for numerical stability.
    rng:
        RNG used when ``outcome`` is None. If None and sampling is required,
        a default generator is created.

    Returns
    -------
    QuadratureConditioningResult
        Posterior quadrature moments on the kept subsystem + log_prob.

    Notes
    -----
    This kernel operates purely on quadrature mean/covariance and does not
    convert back to ladder moments. A thin wrapper can convert ``(d_post,V_post)``
    back into a :class:`~symop_proto.gaussian.core.GaussianCore`.

    Examples
    --------
    Condition one arm of a two-mode squeezed state on an x-homodyne outcome.

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
        from symop_proto.gaussian.ops.measurement import quadrature_indices, condition_on_quadratures

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.vacuum(B)
        core = TwoModeSqueezer(mode1=m1, mode2=m2, r=0.6, phi=0.0, check_ccr=True).apply(core)

        q_idx = quadrature_indices(core, (m2,), which="x")

        # finite measurement resolution for stability
        Vm = (1e-9) * np.eye(len(q_idx), dtype=float)

        res = condition_on_quadratures(core, meas_q_idx=q_idx, Vm=Vm, outcome=np.array([0.1]))
        print("log_prob:", res.log_prob)
        print("kept dim:", res.d_post.shape[0], res.V_post.shape)

    """
    d = np.asarray(core.quadrature_mean(), dtype=float).reshape(-1)
    V = np.asarray(core.quadrature_covariance(), dtype=float)

    if V.shape != (d.shape[0], d.shape[0]):
        raise ValueError("core returned inconsistent quadrature shapes")

    m_idx = np.asarray(list(meas_q_idx), dtype=int).reshape(-1)
    if m_idx.size == 0:
        raise ValueError("meas_q_idx must not be empty")

    # Validate indices in the quadrature space
    dim = int(d.shape[0])
    if np.any(m_idx < 0) or np.any(m_idx >= dim):
        raise IndexError("meas_q_idx out of range for quadrature dimension")
    if len(set(int(i) for i in m_idx.tolist())) != int(m_idx.size):
        raise ValueError("meas_q_idx contains duplicates")

    keep_idx = complement_indices(dim, m_idx.tolist())

    dB = d[m_idx]
    VBB = V[np.ix_(m_idx, m_idx)]
    VAB = V[np.ix_(keep_idx, m_idx)]
    VAA = V[np.ix_(keep_idx, keep_idx)]

    if Vm is None:
        Vm_use = np.zeros((m_idx.size, m_idx.size), dtype=float)
    else:
        Vm_use = as_real_mat(Vm, shape=(m_idx.size, m_idx.size), name="Vm")

    S = VBB + Vm_use
    # Symmetrize numeric noise
    S = 0.5 * (S + S.T)

    if outcome is None:
        if rng is None:
            rng = np.random.default_rng()
        y = gaussian_sample(dB, S, rng=rng)
    else:
        y = as_real_vec(outcome, m=m_idx.size, name="outcome")

    # log p(y)
    lp = gaussian_logpdf(y, dB, S)

    # K = VAB S^{-1} without forming inverse:
    # K = (S^{-T} VAB^T)^T
    K = np.linalg.solve(S.T, VAB.T).T

    dA_post = d[keep_idx] + K @ (y - dB)
    VAA_post = VAA - K @ VAB.T
    VAA_post = 0.5 * (VAA_post + VAA_post.T)

    return QuadratureConditioningResult(
        outcome=np.asarray(y, dtype=float),
        log_prob=float(lp),
        keep_q_idx=keep_idx,
        meas_q_idx=m_idx,
        d_post=np.asarray(dA_post, dtype=float),
        V_post=np.asarray(VAA_post, dtype=float),
    )


def condition_on_modes(
    core: GaussianCore,
    *,
    modes: Sequence[ModeOpProto],
    which: str,
    outcome: np.ndarray | None = None,
    meas_var: float = 0.0,
    rng: np.random.Generator | None = None,
) -> QuadratureConditioningResult:
    r"""Convenience wrapper: condition on measuring quadratures of given modes.

    This is a thin wrapper around :func:`condition_on_quadratures` that converts
    ``modes`` into quadrature indices using the core's basis ordering.

    Parameters
    ----------
    core:
        Input Gaussian state.
    modes:
        Modes to measure.
    which:
        One of ``"x"``, ``"p"``, ``"xp"``.
    outcome:
        Outcome vector (shape depends on ``which``). If None, sample.
    meas_var:
        Scalar measurement noise variance added as ``meas_var * I``.
        For heterodyne, a common choice is ``meas_var = 0.5`` (vacuum noise).
    rng:
        RNG used when sampling.

    Returns
    -------
    QuadratureConditioningResult
        Posterior moments on the kept subsystem.

    Examples
    --------
    Heterodyne one mode of a displaced state:

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
        from symop_proto.gaussian.ops.measurement import condition_on_modes

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])
        core = GaussianCore.vacuum(B)

        core = Displacement(modes=(m,), beta=np.array([0.2 + 0.3j], dtype=complex)).apply(core)

        res = condition_on_modes(core, modes=(m,), which="xp", meas_var=0.5)
        print("outcome:", res.outcome)
        print("log_prob:", res.log_prob)
        print("kept dim:", res.d_post.shape[0])

    """
    idx = as_index_list(core, modes)
    check_is_subset_indices(core.basis.n, idx)

    q_idx = quadrature_indices(core, modes, which=which)

    mv = float(meas_var)
    if mv < 0.0:
        raise ValueError("meas_var must be >= 0")

    Vm = mv * np.eye(int(q_idx.size), dtype=float)
    return condition_on_quadratures(
        core,
        meas_q_idx=q_idx,
        outcome=outcome,
        Vm=Vm,
        rng=rng,
    )
