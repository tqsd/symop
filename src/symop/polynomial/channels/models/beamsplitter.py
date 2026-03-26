"""Beamsplitter models for CCR polynomial representations.

This module provides helpers for applying a two-mode beamsplitter
unitary to ket, density, and operator polynomial objects.

This implementation performs a *mode-expanding substitution*:
each input mode is mapped to a superposition of *output-path modes*.

This is required for correct Hong-Ou-Mandel behavior when mode identity
includes envelope and path.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from symop.ccr.algebra.ket.poly import KetPoly
from symop.ccr.protocols.density import DensityPoly as DensityPolyProtocol
from symop.ccr.protocols.op import OpPoly as OpPolyProtocol
from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.protocols.ops import LadderOp as LadderOpProtocol
from symop.core.protocols.ops import ModeOp as ModeOpProtocol
from symop.core.types.operator_kind import OperatorKind
from symop.polynomial.rewrites.substitution import (
    rewrite_densitypoly,
    rewrite_ketpoly,
    rewrite_oppoly,
)

SubstFn = Callable[[LadderOpProtocol], list[tuple[complex, LadderOpProtocol]]]


def _beamsplitter_substitution(
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    out0: PathProtocol,
    out1: PathProtocol,
    theta: float,
    phi_t: float,
    phi_r: float,
) -> SubstFn:
    r"""Construct the ladder-operator substitution for a two-mode beamsplitter.

    Parameters
    ----------
    mode0:
        First input mode.
    mode1:
        Second input mode.
    out0:
        Output path for the first output arm.
    out1:
        Output path for the second output arm.
    theta:
        Beamsplitter mixing angle.
    phi_t:
        Transmission phase.
    phi_r:
        Reflection phase.

    Returns
    -------
    SubstFn
        Substitution function mapping each input ladder operator to a
        linear combination of output-path ladder operators.

    Notes
    -----
    The induced transformation on creation operators is

    .. math::

        \hat a_0^\dagger &\mapsto
        t\,\hat a_{0,\mathrm{out0}}^\dagger +
        r\,\hat a_{0,\mathrm{out1}}^\dagger, \\

        \hat a_1^\dagger &\mapsto
        r\,\hat a_{1,\mathrm{out0}}^\dagger -
        t\,\hat a_{1,\mathrm{out1}}^\dagger,

    where

    .. math::

        t = \cos(\theta)e^{i\phi_t},
        \qquad
        r = \sin(\theta)e^{i\phi_r}.

    Annihilation operators are transformed by the corresponding conjugate
    coefficients.

    """
    t = complex(math.cos(theta)) * complex(math.cos(phi_t), math.sin(phi_t))
    r = complex(math.sin(theta)) * complex(math.cos(phi_r), math.sin(phi_r))

    # Output copies of modes (preserve envelope + polarization)
    m0_out0 = mode0.with_path(out0)
    m0_out1 = mode0.with_path(out1)

    m1_out0 = mode1.with_path(out0)
    m1_out1 = mode1.with_path(out1)

    def subst(op: LadderOpProtocol) -> list[tuple[complex, LadderOpProtocol]]:
        sig = op.mode.signature

        if sig == mode0.signature:
            if op.kind == OperatorKind.CRE:
                return [
                    (t, m0_out0.create),
                    (r, m0_out1.create),
                ]
            return [
                (t.conjugate(), m0_out0.ann),
                (r.conjugate(), m0_out1.ann),
            ]

        if sig == mode1.signature:
            if op.kind == OperatorKind.CRE:
                return [
                    (r, m1_out0.create),
                    (-t, m1_out1.create),
                ]
            return [
                (r.conjugate(), m1_out0.ann),
                ((-t).conjugate(), m1_out1.ann),
            ]

        return [(1.0 + 0.0j, op)]

    return subst


def beamsplitter_ketpoly(
    poly: KetPoly,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    out0: PathProtocol,
    out1: PathProtocol,
    theta: float,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
) -> KetPoly:
    r"""Apply a two-mode beamsplitter to a ket polynomial.

    Parameters
    ----------
    poly:
        Input ket polynomial.
    mode0:
        First input mode.
    mode1:
        Second input mode.
    out0:
        Output path for the first output arm.
    out1:
        Output path for the second output arm.
    theta:
        Beamsplitter mixing angle.
    phi_t:
        Transmission phase.
    phi_r:
        Reflection phase.

    Returns
    -------
    KetPoly
        Rewritten ket polynomial after the beamsplitter transformation.

    Notes
    -----
    The transformation is implemented by substituting each selected ladder
    operator with its beamsplitter image and expanding linearly.

    Terms that contain annihilation operators after rewriting are discarded,
    so the result remains a valid creators-only ket polynomial (up to the
    identity term).

    """
    subst = _beamsplitter_substitution(
        mode0=mode0,
        mode1=mode1,
        out0=out0,
        out1=out1,
        theta=theta,
        phi_t=phi_t,
        phi_r=phi_r,
    )

    rewritten = rewrite_ketpoly(poly, subst)

    return KetPoly(
        tuple(
            t
            for t in rewritten.terms
            if t.monomial.is_creator_only or t.monomial.is_identity
        )
    ).combine_like_terms()


def beamsplitter_densitypoly(
    rho: DensityPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    out0: PathProtocol,
    out1: PathProtocol,
    theta: float,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
) -> DensityPolyProtocol:
    r"""Apply a two-mode beamsplitter to a density polynomial.

    Parameters
    ----------
    rho:
        Input density polynomial.
    mode0:
        First input mode.
    mode1:
        Second input mode.
    out0:
        Output path for the first output arm.
    out1:
        Output path for the second output arm.
    theta:
        Beamsplitter mixing angle.
    phi_t:
        Transmission phase.
    phi_r:
        Reflection phase.

    Returns
    -------
    DensityPolyProtocol
        Rewritten density polynomial after the beamsplitter transformation.

    Notes
    -----
    The transformation is applied independently to the ladder operators
    appearing in the density polynomial and expanded linearly.

    """
    subst = _beamsplitter_substitution(
        mode0=mode0,
        mode1=mode1,
        out0=out0,
        out1=out1,
        theta=theta,
        phi_t=phi_t,
        phi_r=phi_r,
    )

    return rewrite_densitypoly(rho, subst)


def beamsplitter_oppoly(
    op: OpPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    out0: PathProtocol,
    out1: PathProtocol,
    theta: float,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
) -> OpPolyProtocol:
    r"""Apply a two-mode beamsplitter to an operator polynomial.

    Parameters
    ----------
    op:
        Input operator polynomial.
    mode0:
        First input mode.
    mode1:
        Second input mode.
    out0:
        Output path for the first output arm.
    out1:
        Output path for the second output arm.
    theta:
        Beamsplitter mixing angle.
    phi_t:
        Transmission phase.
    phi_r:
        Reflection phase.

    Returns
    -------
    OpPolyProtocol
        Rewritten operator polynomial after the beamsplitter transformation.

    Notes
    -----
    This applies the same mode-expanding substitution used for ket and
    density polynomials, but without imposing ket-specific creator-only
    constraints.

    """
    subst = _beamsplitter_substitution(
        mode0=mode0,
        mode1=mode1,
        out0=out0,
        out1=out1,
        theta=theta,
        phi_t=phi_t,
        phi_r=phi_r,
    )

    return rewrite_oppoly(op, subst)


# Convenience wrappers


def beamsplitter_50_50_ketpoly(
    poly: KetPoly,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    out0: PathProtocol,
    out1: PathProtocol,
) -> KetPoly:
    r"""Apply a 50:50 beamsplitter to a ket polynomial.

    Parameters
    ----------
    poly:
        Input ket polynomial.
    mode0:
        First input mode.
    mode1:
        Second input mode.
    out0:
        Output path for the first output arm.
    out1:
        Output path for the second output arm.

    Returns
    -------
    KetPoly
        Rewritten ket polynomial after a balanced beamsplitter
        transformation.

    Notes
    -----
    This is a convenience wrapper around :func:`beamsplitter_ketpoly`
    with

    .. math::

        \theta = \frac{\pi}{4}.

    """
    return beamsplitter_ketpoly(
        poly,
        mode0=mode0,
        mode1=mode1,
        out0=out0,
        out1=out1,
        theta=math.pi / 4.0,
    )


def beamsplitter_50_50_densitypoly(
    rho: DensityPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    out0: PathProtocol,
    out1: PathProtocol,
) -> DensityPolyProtocol:
    r"""Apply a 50:50 beamsplitter to a density polynomial.

    Parameters
    ----------
    rho:
        Input density polynomial.
    mode0:
        First input mode.
    mode1:
        Second input mode.
    out0:
        Output path for the first output arm.
    out1:
        Output path for the second output arm.

    Returns
    -------
    DensityPolyProtocol
        Rewritten density polynomial after a balanced beamsplitter
        transformation.

    Notes
    -----
    This is a convenience wrapper around :func:`beamsplitter_densitypoly`
    with

    .. math::

        \theta = \frac{\pi}{4}.

    """
    return beamsplitter_densitypoly(
        rho,
        mode0=mode0,
        mode1=mode1,
        out0=out0,
        out1=out1,
        theta=math.pi / 4.0,
    )


def beamsplitter_50_50_oppoly(
    op: OpPolyProtocol,
    *,
    mode0: ModeOpProtocol,
    mode1: ModeOpProtocol,
    out0: PathProtocol,
    out1: PathProtocol,
) -> OpPolyProtocol:
    r"""Apply a 50:50 beamsplitter to an operator polynomial.

    Parameters
    ----------
    op:
        Input operator polynomial.
    mode0:
        First input mode.
    mode1:
        Second input mode.
    out0:
        Output path for the first output arm.
    out1:
        Output path for the second output arm.

    Returns
    -------
    OpPolyProtocol
        Rewritten operator polynomial after a balanced beamsplitter
        transformation.

    Notes
    -----
    This is a convenience wrapper around :func:`beamsplitter_oppoly`
    with

    .. math::

        \theta = \frac{\pi}{4}.

    """
    return beamsplitter_oppoly(
        op,
        mode0=mode0,
        mode1=mode1,
        out0=out0,
        out1=out1,
        theta=math.pi / 4.0,
    )
