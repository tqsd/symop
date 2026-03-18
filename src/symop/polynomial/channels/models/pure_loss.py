"""Pure-loss channels for symbolic density polynomials.

Implements the bosonic pure-loss channel as a specialization of a
unitary dilation: a signal mode is mixed with a vacuum environment mode
on a beamsplitter, and the environment is traced out afterward.

Convenience helpers are provided for applying loss to a single mode,
multiple modes sequentially, or via mappings from modes to
transmissivities.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.protocols.density import DensityPoly as DensityPolyProtocol
from symop.core.protocols.ops import ModeOp as ModeOpProtocol
from symop.polynomial.channels.primitives.unitary_dilation import (
    UnitaryDilation,
    apply_unitary_dilation_densitypoly,
)
from symop.polynomial.channels.unitaries.beamsplitter import loss_dilation_u


def _require_eta(eta: float) -> float:
    r"""Validate transmissivity eta.

    Parameters
    ----------
    eta:
        Power transmissivity :math:`\eta` for the pure-loss channel.

    Returns
    -------
    float
        ``eta`` as a float.

    Raises
    ------
    ValueError
        If ``eta`` is not finite or not in ``[0, 1]``.

    Notes
    -----
    Clipping or other parameter preprocessing is expected to happen
    upstream. This helper performs validation only.

    """
    e = float(eta)
    if not math.isfinite(e):
        raise ValueError(f"eta must be finite, got {eta!r}")
    if e < 0.0 or e > 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta!r}")
    return e


def pure_loss_densitypoly(
    rho: DensityPolyProtocol,
    *,
    signal_mode: ModeOpProtocol,
    env_mode: ModeOpProtocol,
    eta: float,
    normalize_trace: bool = True,
    eps: float = 1e-14,
) -> DensityPoly:
    r"""Apply a single-mode pure-loss channel to a symbolic density polynomial.

    Channel definition
    ------------------
    The pure-loss channel with transmissivity :math:`\eta` is realized
    as a unitary dilation using a beamsplitter acting on the signal mode
    and an environment mode prepared in vacuum, followed by tracing out
    the environment:

    .. math::

        \mathcal{E}_\eta(\rho)
        =
        \mathrm{Tr}_E\!\left[
            U_{\mathrm{BS}}(\eta)\;
            (\rho \otimes |0\rangle\langle 0|)\;
            U_{\mathrm{BS}}(\eta)^\dagger
        \right].

    We use a 2x2 beamsplitter unitary with

    .. math::

        t = \sqrt{\eta},
        \qquad
        r = \sqrt{1-\eta}.

    Parameters
    ----------
    rho:
        Input symbolic density polynomial.
    signal_mode:
        The mode to which the channel is applied.
    env_mode:
        Environment mode used for the dilation and traced out at the end.

        This should be a fresh mode not otherwise part of the retained
        system, otherwise the final trace would remove genuine system
        degrees of freedom.
    eta:
        Transmissivity :math:`\eta \in [0, 1]`.
    normalize_trace:
        If True, trace-normalize the reduced output density.
    eps:
        Threshold used by trace normalization when enabled.

    Returns
    -------
    DensityPoly
        Reduced density polynomial after applying the dilation and
        tracing out the environment.

    Notes
    -----
    The environment is treated as vacuum implicitly in the same sense as
    in the generic unitary-dilation primitive: the input density need
    not explicitly contain operators on ``env_mode``.

    """
    e = _require_eta(eta)

    dilation = UnitaryDilation(
        modes=(signal_mode, env_mode),
        U=loss_dilation_u(eta=e),
        trace_out_modes=(env_mode,),
        check_unitary=False,
    )

    return cast(
        DensityPoly,
        apply_unitary_dilation_densitypoly(
            rho,
            dilation=dilation,
            normalize_trace=normalize_trace,
            eps=eps,
        ),
    )


@dataclass(frozen=True)
class PureLossSpec:
    r"""Per-mode specification for a pure-loss channel application.

    Attributes
    ----------
    signal_mode:
        Mode to which the pure-loss channel is applied.
    env_mode:
        Environment mode used for the dilation.
    eta:
        Transmissivity :math:`\eta \in [0, 1]`.

    """

    signal_mode: ModeOpProtocol
    env_mode: ModeOpProtocol
    eta: float


def pure_loss_densitypoly_many(
    rho: DensityPolyProtocol,
    *,
    specs: Iterable[PureLossSpec],
    normalize_trace: bool = True,
    eps: float = 1e-14,
) -> DensityPoly:
    r"""Apply multiple pure-loss channels sequentially.

    This is a convenience wrapper that repeatedly applies
    :func:`pure_loss_densitypoly`, once for each specification.

    Parameters
    ----------
    rho:
        Input density polynomial.
    specs:
        Iterable of pure-loss specifications.
    normalize_trace:
        If True, trace-normalize after each application.
    eps:
        Threshold used by trace normalization when enabled.

    Returns
    -------
    DensityPoly
        Output density polynomial after all channel applications.

    Notes
    -----
    Sequential application is the safest default because each dilation
    introduces its own environment mode, which is traced out
    immediately.

    """
    out = rho
    for s in specs:
        out = pure_loss_densitypoly(
            out,
            signal_mode=s.signal_mode,
            env_mode=s.env_mode,
            eta=s.eta,
            normalize_trace=normalize_trace,
            eps=eps,
        )
    return cast(DensityPoly, out)


def pure_loss_densitypoly_by_mode(
    rho: DensityPolyProtocol,
    *,
    eta_by_mode: Mapping[ModeOpProtocol, float],
    env_by_signal_mode: Mapping[ModeOpProtocol, ModeOpProtocol],
    normalize_trace: bool = True,
    eps: float = 1e-14,
) -> DensityPoly:
    r"""Apply pure loss to several modes specified by mappings.

    Parameters
    ----------
    rho:
        Input density polynomial.
    eta_by_mode:
        Mapping from signal mode to transmissivity.
    env_by_signal_mode:
        Mapping from signal mode to the corresponding environment mode
        used in that mode's dilation.
    normalize_trace:
        If True, trace-normalize after each application.
    eps:
        Threshold used by trace normalization when enabled.

    Returns
    -------
    DensityPoly
        Output density polynomial.

    Raises
    ------
    KeyError
        If an environment mode is missing for a signal mode.

    """
    specs = [
        PureLossSpec(
            signal_mode=signal_mode,
            env_mode=env_by_signal_mode[signal_mode],
            eta=float(eta),
        )
        for signal_mode, eta in eta_by_mode.items()
    ]
    return pure_loss_densitypoly_many(
        rho,
        specs=specs,
        normalize_trace=normalize_trace,
        eps=eps,
    )
