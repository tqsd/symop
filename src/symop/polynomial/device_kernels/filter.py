r"""Polynomial density kernel for filter-like attenuation device.

This module implement a polynomial-density kernel that applies
mode-dependent atenuation through a pure-loss dilation.

For each affected mode with transmissivity :math:`\eta`, the channel
is realized by coupling the signal mode to a fresh vacuum environment
mode through a beamsplitter dilation, followed by tracing out the
environment.

Notes
-----
The semantic planning stage is expected to provide the mapping

``action.params["eta_by_mode"]``

from input mode signatures to transmissivities.

Any output relabeling should be expressed through ``action.edits`` and
is applied by the device after kernel execution. The kexnel itself
should only perform the representation-specific transformation.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from symop.devices.protocols.action import DeviceAction as DeviceActionProtocol
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.polynomial.channels.models.pure_loss import pure_loss_densitypoly
from symop.polynomial.state.density import DensityPolyState
from symop.polynomial.state.ket import KetPolyState


@dataclass(frozen=True)
class _FilterKernelParams:
    """Parsed krenel parameters for attenuation-style filter devices.

    Attributes
    ----------
    eta_by_mode: Mapping[Any, float]

    """

    eta_by_mode: Mapping[Any, float]


def _parse_filter_params(
    action: DeviceActionProtocol,
) -> _FilterKernelParams:
    """Extract and validate kernel parameters from a device action.

    Parameters
    ----------
    action : DeviceActionProtocol
        Semantic device action produced during planning.

    Returns
    -------
    _FilterKernelParams
        Parsed attenuation parameters.

    Raises
    ------
    TypeError
        If ``action.params`` or ``action.params["eta_by_mode"]`` does not
        have the expected mapping shape.

    """
    params = action.params
    if not isinstance(params, Mapping):
        raise TypeError("Filter kernel expects action.params to be a mapping")

    eta_by_mode = params.get("eta_by_mode")
    if not isinstance(eta_by_mode, Mapping):
        raise TypeError("Filter kernel expects params['eta_by_mode'] to be a mapping")

    return _FilterKernelParams(eta_by_mode=eta_by_mode)


def _make_env_mode_for_loss(
    *,
    ctx: ApplyContextProtocol,
    signal_mode: Any,
) -> Any:
    """Create a fresh environment mode for a pure-loss dilation.

    Parameters
    ----------
    ctx : ApplyContextProtocol
        Apply context used to allocate a fresh environment path.
    signal_mode : Any
        Signal mode to be attenuated.

    Returns
    -------
    Any
        Environment mode matching the signal mode envelope on a fresh path.

    """
    env_path = ctx.allocate_path(hint="env")
    env_mode = signal_mode.with_path(env_path)
    env_mode = env_mode.with_envelope(signal_mode.label.envelope)
    return env_mode


def filter_poly_density(
    *,
    state: DensityPolyState,
    action: DeviceActionProtocol,
    ctx: ApplyContextProtocol,
) -> DensityPolyState:
    """Apply mode-dependent attenuation to a polynomial density state.

    Parameters
    ----------
    state : DensityPolyState
        Input polynomial density state.
    action : DeviceActionProtocol
        Semantic device action containing ``eta_by_mode`` in ``params``.
    ctx : ApplyContextProtocol
        Apply context used for fresh environment-path allocation.

    Returns
    -------
    DensityPolyState
        Output density state after applying all requested pure-loss channels.

    Notes
    -----
    This kernel performs only the backend-specific channel application.
    Any label edits are expected to be applied later by the device runtime.

    """
    parsed = _parse_filter_params(action)

    modes_by_sig = state.mode_by_signature
    rho = state.rho

    for sig in sorted(parsed.eta_by_mode.keys(), key=repr):
        eta = float(parsed.eta_by_mode[sig])

        signal_mode = modes_by_sig.get(sig)
        if signal_mode is None:
            continue

        env_mode = _make_env_mode_for_loss(
            ctx=ctx,
            signal_mode=signal_mode,
        )

        rho = pure_loss_densitypoly(
            rho,
            signal_mode=signal_mode,
            env_mode=env_mode,
            eta=eta,
            normalize_trace=True,
        )

    out = DensityPolyState.from_densitypoly(rho)
    out = out.normalize_trace()
    return out


def filter_poly_ket(
    *,
    state: KetPolyState,
    action: DeviceActionProtocol,
    ctx: ApplyContextProtocol,
) -> DensityPolyState:
    """Apply mode-dependent attenuation to a polynomial ket state.

    Parameters
    ----------
    state : KetPolyState
        Input polynomial density state.
    action : DeviceActionProtocol
        Semantic device action containing ``eta_by_mode`` in ``params``.
    ctx : ApplyContextProtocol
        Apply context used for fresh environment-path allocation.

    Returns
    -------
    DensityPolyState
        Output density state after applying all requested pure-loss channels.

    Notes
    -----
    This is a thin wrapper, which expands a ket to dense representation

    """
    state2 = state.to_density()

    return filter_poly_density(
        state=cast(DensityPolyState, state2), action=action, ctx=ctx
    )
