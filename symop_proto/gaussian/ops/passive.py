from __future__ import annotations

from typing import Sequence

import numpy as np

from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.ops.common import (
    check_is_square_matrix,
    check_is_unitary,
)


def apply_passive_unitary_subset(
    core: GaussianCore,
    *,
    idx: Sequence[int],
    U: np.ndarray,
    check_unitary: bool = False,
    check_gram_unitary: bool = False,
    atol: float = 1e-12,
) -> GaussianCore:
    r"""
    Apply a passive (photon-number preserving) linear optical map on a subset of modes.

    This is the low-level kernel used by passive Gaussian map objects
    (beam splitters, phase shifters, interferometers).

    Mathematical model
    ------------------

    Let :math:`a` be the column vector of annihilation operators associated
    with the basis stored in ``core.basis``. Let :math:`S` be a subset of mode
    indices (given by ``idx``) and define :math:`a_S` as the subvector of
    annihilation operators on that subset.

    In the Heisenberg picture, a passive linear optical device acts as

    .. math::

        a_S' = U\,a_S,

    where :math:`U \in \mathbb{C}^{k\times k}` is a unitary matrix,
    :math:`k = |S|`, and all other modes are unchanged.

    Moment update rules
    -------------------

    The Gaussian core stores first moments

    .. math::

        \alpha_i = \langle a_i \rangle,

    number correlations

    .. math::

        N_{ij} = \langle a_i^\dagger a_j \rangle,

    and pairing correlations

    .. math::

        M_{ij} = \langle a_i a_j \rangle.

    Under the passive transformation :math:`a_S' = U a_S`, these moments
    transform as

    .. math::

        \alpha_S' = U\,\alpha_S,

    .. math::

        N' = U_{\mathrm{full}}^\dagger\, N\, U_{\mathrm{full}},

    .. math::

        M' = U_{\mathrm{full}}\, M\, U_{\mathrm{full}}^{\mathsf T},

    where :math:`U_{\mathrm{full}}` is the identity on untouched modes and
    equals :math:`U` on the selected subset.

    Subset update strategy (no explicit :math:`U_{\mathrm{full}}`)
    --------------------------------------------------------------

    This kernel avoids constructing :math:`U_{\mathrm{full}}` by updating the
    relevant row/column blocks in-place:

    * First moments:
      ``alpha[idx] <- U @ alpha[idx]``

    * Number correlations (:math:`N' = U^\dagger N U`):
      ``N[idx, :] <- U^dag @ N[idx, :]``
      and
      ``N[:, idx] <- N[:, idx] @ U``

    * Pairing correlations (:math:`M' = U M U^T`):
      ``M[idx, :] <- U @ M[idx, :]``
      and
      ``M[:, idx] <- M[:, idx] @ U^T``

    These updates propagate correctly into cross-correlation blocks
    between acted-on and untouched modes, which is essential when the
    global Gaussian state is entangled.

    Numerical invariant stabilization
    ---------------------------------

    In exact arithmetic, valid moments satisfy

    .. math::

        N = N^\dagger, \qquad M = M^{\mathsf T}.

    Floating-point arithmetic can introduce tiny violations of Hermiticity
    and symmetry. This kernel restores them via

    .. math::

        N \leftarrow \tfrac{1}{2}(N + N^\dagger), \qquad
        M \leftarrow \tfrac{1}{2}(M + M^{\mathsf T}).

    Physical unitarity checks in non-orthogonal mode bases
    ------------------------------------------------------

    If your mode basis is not orthonormal, the commutators are described by
    the Gram block

    .. math::

        G_{SS} \;=\; [a_S, a_S^\dagger].

    The physically correct passive constraint is not necessarily
    :math:`U^\dagger U = I`, but rather CCR preservation on the subspace:

    .. math::

        U\,G_{SS}\,U^\dagger = G_{SS}.

    If you set ``check_gram_unitary=True``, the kernel enforces this condition
    (within tolerance) using ``core.basis.gram``. If you set
    ``check_unitary=True``, it enforces the standard unitarity condition
    :math:`U^\dagger U = I`, which is appropriate when :math:`G_{SS} = I`.

    Parameters
    ----------
    core:
        Input Gaussian core.
    idx:
        Indices of the acted-on subset (length ``k``).
    U:
        Local mixing matrix of shape ``(k,k)`` implementing :math:`a_S' = U a_S`.
    check_unitary:
        If True, validate :math:`U^\dagger U = I` (use when the subset basis is
        orthonormal).
    check_gram_unitary:
        If True, validate :math:`U G_{SS} U^\dagger = G_{SS}` using the Gram
        block from ``core.basis.gram`` (recommended when modes may be
        non-orthogonal).
    atol:
        Absolute tolerance for checks and invariant stabilization.

    Returns
    -------
    GaussianCore
        New Gaussian core with updated moments.

    Examples
    --------

    Vacuum is invariant under any passive map:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.ops.passive import apply_passive_unitary_subset

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.vacuum(B)

        theta = np.pi / 4.0
        U = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=complex)

        core2 = apply_passive_unitary_subset(core, idx=[0, 1], U=U, check_unitary=True)

        print("alpha:", core2.alpha)
        print("N:", core2.N)
        print("M:", core2.M)

    Coherent amplitudes mix as ``alpha_out = U @ alpha_in``:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.ops.passive import apply_passive_unitary_subset

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        alpha = np.array([1.0 + 0.2j, -0.3 + 0.0j], dtype=complex)
        core = GaussianCore.coherent(B, alpha)

        theta = np.pi / 4.0
        U = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=complex)

        core2 = apply_passive_unitary_subset(core, idx=[0, 1], U=U, check_unitary=True)

        print("alpha_in:", core.alpha)
        print("alpha_out:", core2.alpha)
        print("expected:", U @ alpha)

    Local action on an entangled state updates cross-correlations:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.ops.passive import apply_passive_unitary_subset

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        r = 0.7
        s2 = np.sinh(r) ** 2
        sc = np.sinh(r) * np.cosh(r)

        alpha0 = np.zeros((2,), dtype=complex)
        N = np.array([[s2, 0.0],
                      [0.0, s2]], dtype=complex)
        M = np.array([[0.0, sc],
                      [sc, 0.0]], dtype=complex)

        core = GaussianCore.from_moments(B, alpha=alpha0, N=N, M=M)

        phi = 0.3
        U = np.array([[np.exp(1j * phi), 0.0],
                      [0.0,            1.0]], dtype=complex)

        core2 = apply_passive_unitary_subset(core, idx=[0], U=U[:1, :1], check_unitary=True)

        expected = np.exp(1j * phi) * core.M[0, 1]
        print("M12 in:", core.M[0, 1])
        print("M12 out:", core2.M[0, 1])
        print("expected:", expected)

    """
    idx_list = list(idx)
    if len(idx_list) == 0:
        return core

    Uc = np.asarray(U, dtype=complex)
    check_is_square_matrix(Uc)

    k = len(idx_list)
    if Uc.shape != (k, k):
        raise ValueError(
            f"U shape mismatch: expected {(k, k)} for k={k}, got {Uc.shape}"
        )

    if check_unitary:
        # Appropriate when the selected modes form an orthonormal sub-basis.
        check_is_unitary(Uc, atol=atol)

    if check_gram_unitary:
        # Physically correct check in a non-orthogonal basis: preserve CCR.
        Gs = np.asarray(
            core.basis.gram[np.ix_(idx_list, idx_list)], dtype=complex
        )
        lhs = Uc @ Gs @ Uc.conj().T
        if not np.allclose(lhs, Gs, atol=atol, rtol=0.0):
            err = float(np.max(np.abs(lhs - Gs)))
            raise ValueError(
                "U does not preserve the Gram commutator block on the subset; "
                f"max abs err {err}"
            )

    alpha = np.asarray(core.alpha, dtype=complex).copy()
    N = np.asarray(core.N, dtype=complex).copy()
    M = np.asarray(core.M, dtype=complex).copy()

    # alpha_S' = U alpha_S
    alpha[idx_list] = Uc @ alpha[idx_list]

    # N' = U^dag N U (subset updates)
    N[idx_list, :] = Uc.conj().T @ N[idx_list, :]
    N[:, idx_list] = N[:, idx_list] @ Uc

    # M' = U M U^T (subset updates)
    M[idx_list, :] = Uc @ M[idx_list, :]
    M[:, idx_list] = M[:, idx_list] @ Uc.T

    # Stabilize algebraic invariants
    N = 0.5 * (N + N.conj().T)
    M = 0.5 * (M + M.T)

    return GaussianCore(basis=core.basis, alpha=alpha, N=N, M=M)
