r"""Beamsplitter models for CCR polynomial representations.

This module provides helpers for applying a two-mode beamsplitter
unitary to ket, density, and operator polynomial objects.

The model-layer API uses a physically constrained parameterization in
terms of a mixing angle ``theta`` and optional transmission/reflection
phases. The amplitudes are constructed as

.. math::

    t = \cos(\theta), \qquad r = \sin(\theta),

so that the resulting 2x2 transformation is unitary by construction
under the package beamsplitter convention.
"""

from __future__ import annotations

import math

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
from symop.polynomial.channels.unitaries.beamsplitter import beamsplitter_u


def beamsplitter_ketpoly(
    poly: KetPoly,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    theta: float,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> KetPoly:
    r"""Apply a two-mode beamsplitter unitary to a ket polynomial.

    The beamsplitter is parameterized by a mixing angle ``theta`` and
    optional transmission and reflection phases, with

    .. math::

        t = \cos(\theta), \qquad r = \sin(\theta).

    Parameters
    ----------
    poly:
        Input ket polynomial.
    mode0, mode1:
        Ordered pair of modes on which the beamsplitter acts.
    theta:
        Mixing angle controlling the splitting ratio.
    phi_t:
        Transmission phase.
    phi_r:
        Reflection phase.
    check_unitary:
        If True, validate the resulting 2x2 unitary.
    atol:
        Tolerance for optional unitary validation.

    Returns
    -------
    KetPoly
        Rewritten ket polynomial.

    """
    t = float(math.cos(theta))
    r = float(math.sin(theta))

    U = beamsplitter_u(t=t, r=r, phi_t=phi_t, phi_r=phi_r)
    lmap = LinearModeMap(
        modes=(mode0, mode1),
        U=U,
        check_unitary=check_unitary,
        atol=atol,
    )
    return apply_to_ketpoly(poly, lmap=lmap)


def beamsplitter_densitypoly(
    rho: DensityPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    theta: float,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> DensityPolyProtocol:
    r"""Apply a two-mode beamsplitter unitary to a density polynomial.

    The beamsplitter is parameterized by a mixing angle ``theta`` and
    optional transmission and reflection phases, with

    .. math::

        t = \cos(\theta), \qquad r = \sin(\theta).

    Parameters
    ----------
    rho:
        Input density polynomial.
    mode0, mode1:
        Ordered pair of modes on which the beamsplitter acts.
    theta:
        Mixing angle controlling the splitting ratio.
    phi_t:
        Transmission phase.
    phi_r:
        Reflection phase.
    check_unitary:
        If True, validate the resulting 2x2 unitary.
    atol:
        Tolerance for optional unitary validation.

    Returns
    -------
    DensityPoly
        Rewritten density polynomial.

    """
    t = float(math.cos(theta))
    r = float(math.sin(theta))

    U = beamsplitter_u(t=t, r=r, phi_t=phi_t, phi_r=phi_r)
    lmap = LinearModeMap(
        modes=(mode0, mode1),
        U=U,
        check_unitary=check_unitary,
        atol=atol,
    )
    return apply_to_densitypoly(rho, lmap=lmap)


def beamsplitter_oppoly(
    op: OpPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    theta: float,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> OpPolyProtocol:
    r"""Apply a two-mode beamsplitter unitary to an operator polynomial.

    The beamsplitter is parameterized by a mixing angle ``theta`` and
    optional transmission and reflection phases, with

    .. math::

        t = \cos(\theta), \qquad r = \sin(\theta).

    Parameters
    ----------
    op:
        Input operator polynomial.
    mode0, mode1:
        Ordered pair of modes on which the beamsplitter acts.
    theta:
        Mixing angle controlling the splitting ratio.
    phi_t:
        Transmission phase.
    phi_r:
        Reflection phase.
    check_unitary:
        If True, validate the resulting 2x2 unitary.
    atol:
        Tolerance for optional unitary validation.

    Returns
    -------
    OpPoly
        Rewritten operator polynomial.

    """
    t = float(math.cos(theta))
    r = float(math.sin(theta))

    U = beamsplitter_u(t=t, r=r, phi_t=phi_t, phi_r=phi_r)
    lmap = LinearModeMap(
        modes=(mode0, mode1),
        U=U,
        check_unitary=check_unitary,
        atol=atol,
    )
    return apply_to_oppoly(op, lmap=lmap)


def beamsplitter_50_50_ketpoly(
    poly: KetPoly,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> KetPoly:
    r"""Apply a balanced 50:50 beamsplitter to a ket polynomial."""
    return beamsplitter_ketpoly(
        poly,
        mode0=mode0,
        mode1=mode1,
        theta=math.pi / 4.0,
        phi_t=phi_t,
        phi_r=phi_r,
        check_unitary=check_unitary,
        atol=atol,
    )


def beamsplitter_50_50_densitypoly(
    rho: DensityPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> DensityPolyProtocol:
    r"""Apply a balanced 50:50 beamsplitter to a density polynomial."""
    return beamsplitter_densitypoly(
        rho,
        mode0=mode0,
        mode1=mode1,
        theta=math.pi / 4.0,
        phi_t=phi_t,
        phi_r=phi_r,
        check_unitary=check_unitary,
        atol=atol,
    )


def beamsplitter_50_50_oppoly(
    op: OpPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> OpPolyProtocol:
    r"""Apply a balanced 50:50 beamsplitter to an operator polynomial."""
    return beamsplitter_oppoly(
        op,
        mode0=mode0,
        mode1=mode1,
        theta=math.pi / 4.0,
        phi_t=phi_t,
        phi_r=phi_r,
        check_unitary=check_unitary,
        atol=atol,
    )
