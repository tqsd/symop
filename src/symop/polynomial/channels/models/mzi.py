"""Mach-Zehnder interferometer models for CCR polynomial representations.

This module provides helpers for applying a two-mode Mach-Zehnder
interferometer (MZI) unitary to ket, density, and operator polynomial
objects.
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
from symop.polynomial.channels.unitaries.mzi import mzi_u


def mzi_ketpoly(
    poly: KetPoly,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    theta1: float,
    theta2: float,
    phi_internal: float,
    phi_in0: float = 0.0,
    phi_in1: float = 0.0,
    phi_out0: float = 0.0,
    phi_out1: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> KetPoly:
    r"""Apply a two-mode Mach-Zehnder interferometer to a ket polynomial."""
    U = mzi_u(
        theta1=theta1,
        theta2=theta2,
        phi_internal=phi_internal,
        phi_in0=phi_in0,
        phi_in1=phi_in1,
        phi_out0=phi_out0,
        phi_out1=phi_out1,
        check_unitary=check_unitary,
        atol=atol,
    )
    lmap = LinearModeMap(
        modes=(mode0, mode1),
        U=U,
        check_unitary=False,
        atol=atol,
    )
    return apply_to_ketpoly(poly, lmap=lmap)


def mzi_densitypoly(
    rho: DensityPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    theta1: float,
    theta2: float,
    phi_internal: float,
    phi_in0: float = 0.0,
    phi_in1: float = 0.0,
    phi_out0: float = 0.0,
    phi_out1: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> DensityPolyProtocol:
    r"""Apply a two-mode Mach-Zehnder interferometer to a density polynomial."""
    U = mzi_u(
        theta1=theta1,
        theta2=theta2,
        phi_internal=phi_internal,
        phi_in0=phi_in0,
        phi_in1=phi_in1,
        phi_out0=phi_out0,
        phi_out1=phi_out1,
        check_unitary=check_unitary,
        atol=atol,
    )
    lmap = LinearModeMap(
        modes=(mode0, mode1),
        U=U,
        check_unitary=False,
        atol=atol,
    )
    return apply_to_densitypoly(rho, lmap=lmap)


def mzi_oppoly(
    op: OpPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    theta1: float,
    theta2: float,
    phi_internal: float,
    phi_in0: float = 0.0,
    phi_in1: float = 0.0,
    phi_out0: float = 0.0,
    phi_out1: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> OpPolyProtocol:
    r"""Apply a two-mode Mach-Zehnder interferometer to an operator polynomial."""
    U = mzi_u(
        theta1=theta1,
        theta2=theta2,
        phi_internal=phi_internal,
        phi_in0=phi_in0,
        phi_in1=phi_in1,
        phi_out0=phi_out0,
        phi_out1=phi_out1,
        check_unitary=check_unitary,
        atol=atol,
    )
    lmap = LinearModeMap(
        modes=(mode0, mode1),
        U=U,
        check_unitary=False,
        atol=atol,
    )
    return apply_to_oppoly(op, lmap=lmap)
