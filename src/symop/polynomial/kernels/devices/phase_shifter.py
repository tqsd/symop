r"""Polynomial kernels for path phase shifter devices.

This module implements backend kernels for applying a constant phase
rotation to all modes on a selected path in polynomial ket and density
states.

The semantic planning stage is expected to provide the target path and
phase angle in ``action.params``.

Notes
-----
The kernel performs the representation-specific phase rewrite for each
mode on the selected path. No additional label-edit phase is required
for the physical phase-shifter action.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.protocols.density import DensityPoly as DensityPolyProtocol
from symop.core.protocols.modes.labels import Path
from symop.devices.protocols.action import DeviceAction as DeviceActionProtocol
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.polynomial.channels.models.phase import (
    phase_densitypoly,
    phase_ketpoly,
)
from symop.polynomial.state.density import DensityPolyState
from symop.polynomial.state.ket import KetPolyState


@dataclass(frozen=True)
class _PhaseShifterKernelParams:
    """Parsed kernel parameters for one phase-shifter application.

    Attributes
    ----------
    path:
        Target path on which the phase shift is applied.
    phi:
        Phase angle in radians.

    """

    path: Path
    phi: float


def _parse_phase_shifter_params(
    action: DeviceActionProtocol,
) -> _PhaseShifterKernelParams:
    """Extract and validate kernel parameters from a device action.

    Parameters
    ----------
    action:
        Semantic device action produced during planning.

    Returns
    -------
    _PhaseShifterKernelParams
        Parsed phase-shifter parameters.

    Raises
    ------
    TypeError
        If ``action.params`` does not have the expected mapping shape or if
        parameter values have invalid types.
    KeyError
        If a required key is missing.

    """
    params = action.params
    if not isinstance(params, Mapping):
        raise TypeError("PhaseShifter kernel expects action.params to be a mapping")

    if "path" not in params:
        raise KeyError("PhaseShifter action is missing required key 'path'")
    if "phi" not in params:
        raise KeyError("PhaseShifter action is missing required key 'phi'")

    raw_path = params["path"]
    raw_phi = params["phi"]

    if not isinstance(raw_phi, int | float):
        raise TypeError("PhaseShifter kernel expects params['phi'] to be a real number")

    return _PhaseShifterKernelParams(
        path=cast(Path, raw_path),
        phi=float(raw_phi),
    )


def phase_shifter_poly_density(
    *,
    state: DensityPolyState,
    action: DeviceActionProtocol,
    ctx: ApplyContextProtocol,
) -> DensityPolyState:
    """Apply a path phase shifter to a polynomial density state.

    Parameters
    ----------
    state:
        Input polynomial density state.
    action:
        Semantic device action containing phase-shifter parameters.
    ctx:
        Unused for direct phase-shifter application. Included for API
        consistency with other kernels.

    Returns
    -------
    DensityPolyState
        Output density state after applying the phase shift to all modes
        on the selected path.

    Notes
    -----
    The phase shift is applied mode-by-mode to all modes whose labels lie
    on the selected path. Modes on other paths are left unchanged.

    """
    del ctx

    parsed = _parse_phase_shifter_params(action)
    rho: DensityPolyProtocol = state.rho

    for mode in state.modes_on_path(parsed.path):
        rho = phase_densitypoly(
            rho,
            mode=mode,
            phi=parsed.phi,
        )

    return DensityPolyState.from_densitypoly(cast(DensityPoly, rho))


def phase_shifter_poly_ket(
    *,
    state: KetPolyState,
    action: DeviceActionProtocol,
    ctx: ApplyContextProtocol,
) -> KetPolyState:
    """Apply a path phase shifter to a polynomial ket state.

    Parameters
    ----------
    state:
        Input polynomial ket state.
    action:
        Semantic device action containing phase-shifter parameters.
    ctx:
        Unused for direct phase-shifter application. Included for API
        consistency with other kernels.

    Returns
    -------
    KetPolyState
        Output ket state after applying the phase shift to all modes
        on the selected path.

    Notes
    -----
    The phase shift is applied mode-by-mode to all modes whose labels lie
    on the selected path. Modes on other paths are left unchanged.

    """
    del ctx

    parsed = _parse_phase_shifter_params(action)
    ket = state.ket

    for mode in state.modes_on_path(parsed.path):
        ket = phase_ketpoly(
            ket,
            mode=mode,
            phi=parsed.phi,
        )

    return KetPolyState.from_ketpoly(ket)
