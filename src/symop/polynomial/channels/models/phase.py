"""Phase-shifter models for CCR polynomial representations.

This module provides helpers for applying a single-mode phase-shifter
unitary to ket, density, and operator polynomial objects.
"""

from __future__ import annotations

from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.protocols.density import DensityPoly as DensityPolyProtocol
from symop.ccr.protocols.op import OpPoly as OpPolyProtocol
from symop.core.protocols.ops import ModeOp as ModeOpProtocol
from symop.polynomial.channels.primitives.linear_mode_unitary import (
    LinearModeMap,
    apply_to_densitypoly,
    apply_to_ketpoly,
    apply_to_oppoly,
)
from symop.polynomial.channels.unitaries.phase import phase_u


def phase_ketpoly(
    poly: KetPoly,
    *,
    mode: ModeOpProtocol,
    phi: float,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> KetPoly:
    r"""Apply a single-mode phase shift to a ket polynomial."""
    U = phase_u(phi=phi)
    lmap = LinearModeMap(
        modes=(mode,),
        U=U,
        check_unitary=check_unitary,
        atol=atol,
    )
    return apply_to_ketpoly(poly, lmap=lmap)


def phase_densitypoly(
    rho: DensityPolyProtocol,
    *,
    mode: ModeOpProtocol,
    phi: float,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> DensityPolyProtocol:
    r"""Apply a single-mode phase shift to a density polynomial."""
    U = phase_u(phi=phi)
    lmap = LinearModeMap(
        modes=(mode,),
        U=U,
        check_unitary=check_unitary,
        atol=atol,
    )
    return apply_to_densitypoly(rho, lmap=lmap)


def phase_oppoly(
    op: OpPolyProtocol,
    *,
    mode: ModeOpProtocol,
    phi: float,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> OpPolyProtocol:
    r"""Apply a single-mode phase shift to an operator polynomial."""
    U = phase_u(phi=phi)
    lmap = LinearModeMap(
        modes=(mode,),
        U=U,
        check_unitary=check_unitary,
        atol=atol,
    )
    return apply_to_oppoly(op, lmap=lmap)
