from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np

from symop_proto.core.operators import ModeOp, OperatorKind
from symop_proto.algebra.protocols import (
    KetPolyProto,
    DensityPolyProto,
    OpPolyProto,
)
from symop_proto.rewrites.protocols import RewriteDeviceProto
from symop_proto.rewrites.functions.substitution import (
    rewrite_ketpoly,
    rewrite_densitypoly,
    rewrite_oppoly,
)

from symop_proto.state.polynomial_state import KetPolyState, DensityPolyState


@dataclass(frozen=True)
class LinearModeUnitary(RewriteDeviceProto):
    r"""
    Passive linear device acting as a unitary on an ordered tuple of modes.

    Heisenberg map:

    .. math::

        a_k^\dagger \mapsto \sum_j U_{jk}\, a_j^\dagger, \qquad
        a_k \mapsto \sum_j U_{jk}^*\, a_j,

    where ``modes[j]`` is the output basis (columns of :math:`U` encode input images).

    Parameters:
    -----------
    modes
        Ordered tuple of :class:`~symop_proto.core.operators.ModeOp` forming the basis.
    U
        Complex matrix of shape ``(n, n)``. For passive devices, ``U`` is unitary.

    Notes:
    ------
    - Works for ket polynomials, density polynomials, operator polynomials, **and states**
      via :meth:`on_ketpoly`, :meth:`on_density`, :meth:`on_oppoly`, and :meth:`on_state`.
    - Non-orthogonal modes are supported; normal ordering and overlaps are handled by the algebra.

    Examples:
    ---------
    **Ket polynomial (50:50 BS on paths A,B)**

    .. jupyter-execute::

        import numpy as np
        from IPython.display import display
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.algebra.polynomial import KetPoly
        from symop_proto.rewrites.linear_mode_unitary import LinearModeUnitary

        env = GaussianEnvelope(omega0=0.0, sigma=1.0, tau=0.0, phi0=0.0)
        A = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))

        c = s = 1/np.sqrt(2)
        U = np.array([[c, s], [-s, c]], dtype=np.complex128)
        bs = LinearModeUnitary((A, B), U)

        psi_in = KetPoly.from_word(ops=(A.create,))
        display(bs.on_ketpoly(psi_in))

    **Density polynomial (:math:`\lvert 1_a \rangle \langle 1_A \rvert` through same BS)**

    .. jupyter-execute::

        from symop_proto.algebra.polynomial import DensityPoly
        rho_in = DensityPoly.pure(psi_in)
        display(bs.on_density(rho_in))

    **Operator polynomial (number on A)**

    .. jupyter-execute::

        from symop_proto.algebra.operator_polynomial import OpPoly, OpTerm
        N_A = OpPoly((OpTerm(coeff=1.0, ops=(A.create, A.ann)),))
        display(bs.on_oppoly(N_A))

    **State objects: KetPolyState and DensityPolyState**

    .. jupyter-execute::

        from symop_proto.state.polynomial_state import KetPolyState, DensityPolyState

        # Ket state |1_A>
        psi_state = KetPolyState.from_ketpoly(psi_in)
        psi_state_out = bs.on_state(psi_state)          # returns KetPolyState
        display(psi_state_out)

        # Density state rho = |1_A><1_A|
        rho_state = psi_state.to_density()
        rho_state_out = bs.on_state(rho_state)          # returns DensityPolyState
        display(rho_state_out)

    **Half-wave plate on one path (swap \(H \leftrightarrow V\) up to phases)**

    .. jupyter-execute::

        import numpy as np
        from symop_proto.labels.polarization_label import PolarizationLabel

        A_H = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        A_V = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.V()))

        theta, chi = np.pi/4, np.pi
        c, s = np.cos(theta), np.sin(theta)
        R  = np.array([[c, s], [-s, c]], dtype=np.complex128)
        D  = np.array([[1, 0], [0, np.exp(1j*chi)]], dtype=np.complex128)
        Uhv = (R @ D @ R.T.conj()).astype(np.complex128)

        hwp = LinearModeUnitary((A_H, A_V), Uhv)

        psiH = KetPolyState.from_ketpoly(KetPoly.from_word(ops=(A_H.create,)))
        psiV = hwp.on_state(psiH)
        display(hwp.on_state(psiH))   # ~ |1_V> (phase up to conventions)
        display(f"Initial polarization: {psiH.ket.terms[0].monomial.mode_ops[0].label.pol}")
        display(f"Final polarization: {psiV.ket.terms[0].monomial.mode_ops[0].label.pol}")

    **Polarizing beam splitter (route H -> T, V->R)**

    .. jupyter-execute::

        T_H  = ModeOp(env=env, label=ModeLabel(PathLabel("T"),  PolarizationLabel.H()))
        R_V  = ModeOp(env=env, label=ModeLabel(PathLabel("R"),  PolarizationLabel.V()))
        # Basis (In_H, In_V, T_H, R_V) with routing columns -> rows
        U_pbs = np.eye(4, dtype=np.complex128)
        U_pbs[2, 0] = 1.0   # In_H -> T_H
        U_pbs[3, 1] = 1.0   # In_V -> R_V
        U_pbs[0, 0] = 0.0
        U_pbs[1, 1] = 0.0
        pbs = LinearModeUnitary((A_H, A_V, T_H, R_V), U_pbs)

        # Diagonal input |D> = (H+V)/sqrt(2) at 'In'
        psiD = KetPolyState.from_ketpoly(
            (KetPoly.from_word(ops=(A_H.create,)) + KetPoly.from_word(ops=(A_V.create,))).scaled(1/np.sqrt(2))
        )
        display(pbs.on_state(psiD))   # superposition on T_H and R_V paths
    """

    modes: Tuple[ModeOp, ...]
    U: np.ndarray

    def __post_init__(self):
        n = len(self.modes)
        if self.U.shape != (n, n):
            raise ValueError("LinearModeUnitary: U must be nxn with n=len(modes)")
        # Optional unitary check:
        # if not np.allclose(self.U.conj().T @ self.U, np.eye(n), atol=1e-10):
        #     raise ValueError("U is not unitary")

    # --- core substitution used by all front-ends ----------------------------
    def _subst(self, op):
        try:
            k = self.modes.index(op.mode)
        except ValueError:
            return [(1.0 + 0.0j, op)]
        col = self.U[:, k]
        if op.kind is OperatorKind.CREATE:
            return [
                (complex(col[j]), self.modes[j].create) for j in range(len(self.modes))
            ]
        return [
            (complex(np.conj(col[j])), self.modes[j].ann)
            for j in range(len(self.modes))
        ]

    # --- polynomial APIs ------------------------------------------------------
    def on_ketpoly(self, poly: KetPolyProto) -> KetPolyProto:
        # creators-only states stay creators-only (passive linear device)
        return rewrite_ketpoly(poly, self._subst, apply_to_vacuum=True)

    def on_density(self, rho: DensityPolyProto) -> DensityPolyProto:
        return rewrite_densitypoly(rho, self._subst)

    def on_oppoly(self, op: OpPolyProto) -> OpPolyProto:
        return rewrite_oppoly(op, self._subst)

    # --- state API ------------------------------------------------------------
    def on_state(
        self, state: Union[KetPolyState, DensityPolyState]
    ) -> Union[KetPolyState, DensityPolyState]:
        """
        Rewrite a state and return the same concrete type:

        - KetPolyState -> KetPolyState (label/index preserved)
        - DensityPolyState -> DensityPolyState (trace-normalized; label/index preserved)
        """
        if isinstance(state, KetPolyState):
            new_ket = self.on_ketpoly(state.ket)
            # preserve label/index via the internal helper
            return state._with_ket(new_ket)

        if isinstance(state, DensityPolyState):
            new_rho = self.on_density(state.rho)
            out = DensityPolyState.from_densitypoly(new_rho, normalize_trace=True)
            # preserve human-facing label/index
            out = out.with_label(state.label).with_index(state.index)
            return out

        raise TypeError(f"Unsupported state type: {type(state)!r}")
