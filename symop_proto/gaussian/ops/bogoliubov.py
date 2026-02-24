from __future__ import annotations

from typing import Sequence

import numpy as np

from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.ops.common import (
    check_ccr_preservation,
    check_is_same_shape,
    check_is_square_matrix,
    check_is_subset_indices,
    embed_subset_matrices,
)


def apply_bogoliubov_subset(
    core: GaussianCore,
    *,
    idx: Sequence[int],
    U: np.ndarray,
    V: np.ndarray,
    check_ccr: bool = False,
    atol: float = 1e-12,
) -> GaussianCore:
    r"""
    Apply a linear Bogoliubov transformation to a subset of modes.

    This implements the Heisenberg-picture map on annihilation operators

    .. math::

        a \mapsto U a + V a^\dagger,

    acting on a selected index subset ``idx`` inside the global basis.
    Modes outside ``idx`` are left unchanged. The update is performed on
    the Gaussian moment representation (first moment ``alpha`` and second
    moments ``N``, ``M``) while correctly updating *all* correlation blocks,
    including cross-correlations between acted and untouched modes.

    Moment conventions
    ------------------

    The Gaussian core stores (possibly non-orthogonal) moments:

    .. math::

        \alpha_i = \langle a_i \rangle,\quad
        N_{ij} = \langle a_i^\dagger a_j \rangle,\quad
        M_{ij} = \langle a_i a_j \rangle,

    and the basis commutator is encoded by the Gram matrix ``G``:

    .. math::

        [a_i, a_j^\dagger] = G_{ij},\qquad
        \langle a_i a_j^\dagger \rangle = G_{ij} + N_{ji}.

    Update formulas (centered moments)
    ----------------------------------

    Define centered operators ``delta a = a - alpha`` and centered moments
    ``N0 = <delta a^\dagger delta a>``, ``M0 = <delta a delta a>``.
    With the Bogoliubov map

    .. math::

        \delta a' = U \delta a + V \delta a^\dagger,

    one obtains

    .. math::

        \alpha' = U\alpha + V\alpha^*,

    .. math::

        N_0' = U N_0 U^\dagger
             + U M_0^* V^\dagger
             + V M_0 U^\dagger
             + V (G + N_0^T) V^\dagger,

    .. math::

        M_0' = U M_0 U^T
             + U (G + N_0^T) V^T
             + V N_0 U^T
             + V M_0^* V^T.

    Raw moments are reconstructed as
    ``N' = N0' + alpha'^* alpha'^T`` and ``M' = M0' + alpha' alpha'^T``.

    Parameters
    ----------
    core:
        Gaussian core to transform.
    idx:
        Indices of the acted subspace (in the ordering of ``core.basis``).
    U, V:
        Bogoliubov matrices of shape ``(k,k)`` where ``k=len(idx)``.
    check_ccr:
        If True, validate CCR preservation using the Gram matrix.
    atol:
        Tolerance for CCR checks and numerical stabilization.

    Returns
    -------
    GaussianCore
        Updated Gaussian core.

    Examples
    --------

    Single-mode squeezing on vacuum (canonical single-mode basis):

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.ops.bogoliubov import apply_bogoliubov_subset

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])

        core = GaussianCore.vacuum(B)

        r = 0.7
        phi = 0.2
        ch = np.cosh(r)
        sh = np.sinh(r)

        # One common squeezing convention in Heisenberg picture:
        # a' = cosh(r) a + e^{i phi} sinh(r) a^\dagger
        U = np.array([[ch]], dtype=complex)
        V = np.array([[np.exp(1j * phi) * sh]], dtype=complex)

        core2 = apply_bogoliubov_subset(core, idx=[0], U=U, V=V, check_ccr=True)

        # For vacuum input, the centered moments equal raw moments
        print("N_out:", core2.N[0, 0].real)
        print("expected ~ sinh(r)^2:", (sh * sh))
        print("M_out:", core2.M[0, 0])
        print("expected ~ exp(i phi) sinh(r) cosh(r):", np.exp(1j * phi) * sh * ch)

    Two-mode squeezing (entangling) between two orthogonal modes:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.ops.bogoliubov import apply_bogoliubov_subset

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.vacuum(B)

        r = 0.5
        ch = np.cosh(r)
        sh = np.sinh(r)

        # a1' = ch a1 + sh a2^dag
        # a2' = ch a2 + sh a1^dag
        U = np.array([[ch, 0.0], [0.0, ch]], dtype=complex)
        V = np.array([[0.0, sh], [sh, 0.0]], dtype=complex)

        core2 = apply_bogoliubov_subset(core, idx=[0, 1], U=U, V=V, check_ccr=True)

        print("N diag:", np.diag(core2.N).real)
        print("expected ~ sinh(r)^2:", sh * sh)
        print("M offdiag:", core2.M[0, 1])
        print("expected ~ sinh(r) cosh(r):", sh * ch)
    """
    n = core.basis.n

    U = np.asarray(U, dtype=complex)
    V = np.asarray(V, dtype=complex)

    check_is_subset_indices(n, idx)
    check_is_square_matrix(U, name="U")
    check_is_square_matrix(V, name="V")
    check_is_same_shape(U, V)

    k = len(list(idx))
    if U.shape != (k, k):
        raise ValueError(f"U must have shape ({k},{k}), got {U.shape}")
    if V.shape != (k, k):
        raise ValueError(f"V must have shape ({k},{k}), got {V.shape}")

    U_full, V_full = embed_subset_matrices(n, idx, U, V)

    if check_ccr:
        check_ccr_preservation(
            core.basis.gram, U_full, V_full, atol=float(atol)
        )

    alpha = core.alpha
    N = core.N
    M = core.M
    G = core.basis.gram

    # Centered moments
    N0 = N - np.outer(alpha.conj(), alpha)
    M0 = M - np.outer(alpha, alpha)

    # First moment
    alpha2 = U_full @ alpha + V_full @ alpha.conj()

    # Centered second moments (see docstring)
    N0_2 = (
        U_full @ N0 @ U_full.conj().T
        + U_full @ M0.conj() @ V_full.conj().T
        + V_full @ M0 @ U_full.conj().T
        + V_full @ (G + N0.T) @ V_full.conj().T
    )

    M0_2 = (
        U_full @ M0 @ U_full.T
        + U_full @ (G + N0.T) @ V_full.T
        + V_full @ N0 @ U_full.T
        + V_full @ M0.conj() @ V_full.T
    )

    # Reconstruct raw moments
    N2 = N0_2 + np.outer(alpha2.conj(), alpha2)
    M2 = M0_2 + np.outer(alpha2, alpha2)

    # Enforce algebraic symmetries (numerical stabilization)
    N2 = 0.5 * (N2 + N2.conj().T)
    M2 = 0.5 * (M2 + M2.T)

    return GaussianCore(basis=core.basis, alpha=alpha2, N=N2, M=M2)
