"""Amplifier channels for symbolic density polynomials.

Implements a quantum-limited phase-insensitive bosonic amplifier as a
Bogoliubov transformation on a signal mode and a vacuum environment
mode, followed by tracing out the environment.

Convenience helpers are provided for applying amplification to a single
mode, multiple modes sequentially, or via mappings from modes to gains.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from symop.ccr.protocols.density import DensityPoly as DensityPolyProtocol
from symop.core.protocols.ops import ModeOp as ModeOpProtocol
from symop.polynomial.channels.primitives.bogoliubov import (
    BogoliubovMap,
)
from symop.polynomial.channels.primitives.bogoliubov import (
    apply_to_densitypoly as apply_bogoliubov_to_densitypoly,
)


def _require_gain(gain: float) -> float:
    r"""Validate amplifier gain.

    Parameters
    ----------
    gain:
        Power gain :math:`G` of the amplifier.

    Returns
    -------
    float
        ``gain`` as a float.

    Raises
    ------
    ValueError
        If ``gain`` is not finite or is smaller than 1.

    Notes
    -----
    The quantum-limited phase-insensitive amplifier is defined for
    :math:`G \ge 1`.

    """
    g = float(gain)
    if not math.isfinite(g):
        raise ValueError(f"gain must be finite, got {gain!r}")
    if g < 1.0:
        raise ValueError(f"gain must be >= 1, got {gain!r}")
    return g


def amplifier_bogoliubov_xy(*, gain: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the Bogoliubov matrices for a two-mode amplifier dilation.

    The ordered mode basis is assumed to be ``(signal_mode, env_mode)``.
    For gain :math:`G \ge 1`, the quantum-limited amplifier is realized by

    .. math::

        a_{\mathrm{sig}}
        \mapsto
        \sqrt{G}\, a_{\mathrm{sig}}
        +
        \sqrt{G-1}\, a_{\mathrm{env}}^\dagger,

        a_{\mathrm{env}}
        \mapsto
        \sqrt{G-1}\, a_{\mathrm{sig}}^\dagger
        +
        \sqrt{G}\, a_{\mathrm{env}}.

    In the primitive convention

    .. math::

        a_k^\dagger \mapsto \sum_j X_{j k} a_j^\dagger + Y_{j k} a_j,

    this corresponds to diagonal ``X`` and off-diagonal ``Y``.

    Parameters
    ----------
    gain:
        Power gain :math:`G`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The pair ``(X, Y)``, each of shape ``(2, 2)``.

    """
    g = _require_gain(gain)
    s = math.sqrt(g)
    t = math.sqrt(g - 1.0)

    X = np.asarray(
        [
            [s, 0.0],
            [0.0, s],
        ],
        dtype=np.complex128,
    )
    Y = np.asarray(
        [
            [0.0, t],
            [t, 0.0],
        ],
        dtype=np.complex128,
    )
    return X, Y


def amplifier_densitypoly(
    rho: DensityPolyProtocol,
    *,
    signal_mode: ModeOpProtocol,
    env_mode: ModeOpProtocol,
    gain: float,
    normalize_trace: bool = True,
    eps: float = 1e-14,
    check_ccr: bool = False,
    atol: float = 1e-10,
) -> DensityPolyProtocol:
    r"""Apply a single-mode quantum-limited amplifier channel.

    Channel definition
    ------------------
    The phase-insensitive amplifier with gain :math:`G \ge 1` can be
    realized by a two-mode-squeezing Bogoliubov transformation acting on
    the signal mode and an environment mode prepared in vacuum, followed
    by tracing out the environment.

    Parameters
    ----------
    rho:
        Input symbolic density polynomial.
    signal_mode:
        Signal mode to be amplified.
    env_mode:
        Environment mode used for the dilation and traced out at the end.

        This should be a fresh mode not otherwise part of the retained
        system.
    gain:
        Power gain :math:`G \ge 1`.
    normalize_trace:
        If True, trace-normalize the reduced output density.
    eps:
        Threshold used by trace normalization when enabled.
    check_ccr:
        If True, validate the Bogoliubov CCR constraints.
    atol:
        Tolerance for optional CCR validation.

    Returns
    -------
    DensityPoly
        Reduced density polynomial after applying the amplifier dilation
        and tracing out the environment.

    Notes
    -----
    This implementation is the active-Gaussian analogue of the pure-loss
    dilation model. The environment is treated as vacuum implicitly in
    the same sense as the other channel primitives.

    """
    X, Y = amplifier_bogoliubov_xy(gain=gain)
    bmap = BogoliubovMap(
        modes=(signal_mode, env_mode),
        X=X,
        Y=Y,
        check_ccr=check_ccr,
        atol=atol,
    )

    rho_after = apply_bogoliubov_to_densitypoly(rho, bmap=bmap)
    rho_red = rho_after.partial_trace((env_mode,))

    if normalize_trace:
        rho_red = rho_red.normalize_trace(eps=eps)
    return rho_red


@dataclass(frozen=True)
class AmplifierSpec:
    r"""Per-mode specification for amplifier application.

    Attributes
    ----------
    signal_mode:
        Mode to be amplified.
    env_mode:
        Environment mode used for the dilation.
    gain:
        Power gain :math:`G \ge 1`.

    """

    signal_mode: ModeOpProtocol
    env_mode: ModeOpProtocol
    gain: float


def amplifier_densitypoly_many(
    rho: DensityPolyProtocol,
    *,
    specs: Iterable[AmplifierSpec],
    normalize_trace: bool = True,
    eps: float = 1e-14,
    check_ccr: bool = False,
    atol: float = 1e-10,
) -> DensityPolyProtocol:
    r"""Apply multiple amplifier channels sequentially.

    Parameters
    ----------
    rho:
        Input density polynomial.
    specs:
        Iterable of amplifier specifications.
    normalize_trace:
        If True, trace-normalize after each application.
    eps:
        Threshold used by trace normalization when enabled.
    check_ccr:
        If True, validate the Bogoliubov CCR constraints for each step.
    atol:
        Tolerance for optional CCR validation.

    Returns
    -------
    DensityPoly
        Output density polynomial after all channel applications.

    """
    out = rho
    for s in specs:
        out = amplifier_densitypoly(
            out,
            signal_mode=s.signal_mode,
            env_mode=s.env_mode,
            gain=s.gain,
            normalize_trace=normalize_trace,
            eps=eps,
            check_ccr=check_ccr,
            atol=atol,
        )
    return out


def amplifier_densitypoly_by_mode(
    rho: DensityPolyProtocol,
    *,
    gain_by_mode: Mapping[ModeOpProtocol, float],
    env_by_signal_mode: Mapping[ModeOpProtocol, ModeOpProtocol],
    normalize_trace: bool = True,
    eps: float = 1e-14,
    check_ccr: bool = False,
    atol: float = 1e-10,
) -> DensityPolyProtocol:
    r"""Apply amplification to several modes specified by mappings.

    Parameters
    ----------
    rho:
        Input density polynomial.
    gain_by_mode:
        Mapping from signal mode to amplifier gain.
    env_by_signal_mode:
        Mapping from signal mode to corresponding environment mode.
    normalize_trace:
        If True, trace-normalize after each application.
    eps:
        Threshold used by trace normalization when enabled.
    check_ccr:
        If True, validate the Bogoliubov CCR constraints for each step.
    atol:
        Tolerance for optional CCR validation.

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
        AmplifierSpec(
            signal_mode=signal_mode,
            env_mode=env_by_signal_mode[signal_mode],
            gain=float(gain),
        )
        for signal_mode, gain in gain_by_mode.items()
    ]
    return amplifier_densitypoly_many(
        rho,
        specs=specs,
        normalize_trace=normalize_trace,
        eps=eps,
        check_ccr=check_ccr,
        atol=atol,
    )


__all__ = [
    "AmplifierSpec",
    "amplifier_bogoliubov_xy",
    "amplifier_densitypoly",
    "amplifier_densitypoly_many",
    "amplifier_densitypoly_by_mode",
]
