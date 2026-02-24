from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from symop_proto.core.protocols import ModeOpProto
from symop_proto.gaussian.basis import ModeBasis
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.ops.common import (
    check_is_square_matrix,
    check_is_unitary,
)


def apply_passive_basis_change(
    core: GaussianCore,
    *,
    new_modes: Tuple[ModeOpProto, ...],
    T: np.ndarray,
    check_unitary: bool = False,
    atol: float = 1e-12,
) -> GaussianCore:
    r"""
    Re-express a GaussianCore in a new (passively related) mode basis.

    This operation changes the operator basis but does not represent
    physical evolution. It is the Gaussian analogue of a "rewrite":
    you are representing the same state with respect to a different
    set of mode operators.

    Basis change
    ------------

    Let :math:`a` be the annihilation operator vector of the current basis
    and let :math:`a'` be the annihilation vector of the new basis. Define

    .. math::

        a' = T a.

    Then the moments transform as

    .. math::

        \alpha' = T \alpha,

    .. math::

        N' = T N T^\dagger,

    .. math::

        M' = T M T^T,

    and the Gram matrix transforms as

    .. math::

        G' = T G T^\dagger.

    Parameters
    ----------
    core:
        Input Gaussian core.
    new_modes:
        Ordered tuple of new basis modes. This defines the ordering of :math:`a'`.
    T:
        Basis change matrix of shape ``(n_new, n_old)`` mapping old annihilators
        to new annihilators. Most commonly ``n_new == n_old``.
    check_unitary:
        If True, require ``T`` to be square and unitary.
    atol:
        Numerical tolerance for the unitary check.

    Returns
    -------
    GaussianCore
        Same state represented in the new basis.

    Notes
    -----
    This is intentionally separate from physical devices such as BeamSplitter,
    PhaseShift, Bogoliubov, or channels. Those are maps on the state, not
    just a change of coordinates.

    Examples
    --------
    Change to a rotated two-mode basis (no physical device):

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.ops.basis_change import apply_passive_basis_change

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.coherent(B, np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex))

        theta = np.pi / 4.0
        T = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=complex)

        # define two new modes (could be "virtual" labels if you like)
        u1 = m1.with_path(PathLabel("U1"))
        u2 = m2.with_path(PathLabel("U2"))

        core_new = apply_passive_basis_change(core, new_modes=(u1, u2), T=T, check_unitary=True)

        print("old alpha:", core.alpha)
        print("new alpha:", core_new.alpha)
        print("expected:", T @ core.alpha)
    """
    T = np.asarray(T, dtype=complex)
    n_old = core.basis.n
    n_new = len(new_modes)

    if T.shape != (n_new, n_old):
        raise ValueError(
            f"T must have shape ({n_new}, {n_old}), got {T.shape}"
        )

    if check_unitary:
        check_is_square_matrix(T, name="T")
        check_is_unitary(T, atol=atol)

    alpha2 = T @ core.alpha
    N2 = T @ core.N @ T.conj().T
    M2 = T @ core.M @ T.T

    G2 = T @ core.basis.gram @ T.conj().T
    index_by_sig = {m.signature: i for i, m in enumerate(new_modes)}
    basis2 = ModeBasis(
        modes=tuple(new_modes), gram=G2, index_by_sig=index_by_sig
    )

    return GaussianCore(basis=basis2, alpha=alpha2, N=N2, M=M2)
