r"""Polynomial ket kernel for a simple photon source device.

This module implements a density-polynomial kernel that emits a specified
number of excitations into designated output modes.

The semantic planning stage is expected to provide:

``action.params["source_modes"]``
    Iterable of output modes to populate.

``action.params["excxitations_by_mode"]``
    Mappind from mode signature to a nonnegative integer exccitation count.


Notes
-----
This implementation constructs a pure number-state source and returns
it as a polynomial ket state.

"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from symop.core.protocols.ops import LadderOp as LadderOpProtocol
from symop.core.protocols.ops import ModeOp as ModeOpProtocol
from symop.core.types.operator_kind import OperatorKind
from symop.core.types.signature import Signature
from symop.devices.protocols.action import DeviceAction as DeviceActionProtocol
from symop.devices.protocols.apply_context import (
    ApplyContext as ApplyContextProtocol,
)
from symop.polynomial.state.density import DensityPolyState
from symop.polynomial.state.ket import KetPolyState


@dataclass(frozen=True)
class _NumberStateSourceParams:
    r"""Parsed parameters for the number-state source kernel.

    Attributes
    ----------
    source_modes:
        Tuple of output modes into which excitations are emitted.
    excitations_by_mode:
        Mapping from mode signature to a nonnegative integer specifying
        the number of excitations in that mode.

    """

    source_modes: tuple[ModeOpProtocol, ...]
    excitations_by_mode: Mapping[Signature, int]


def _parse_number_state_source_params(
    action: DeviceActionProtocol,
) -> _NumberStateSourceParams:
    r"""Parse and validate parameters for the number-state source.

    Parameters
    ----------
    action:
        Device action containing semantic parameters.

    Returns
    -------
    _NumberStateSourceParams
        Validated and normalized parameter container.

    Raises
    ------
    TypeError
        If required parameters are missing or have incorrect types.
    ValueError
        If excitation counts are negative.

    Notes
    -----
    Expected entries in ``action.params``:

    - ``"source_modes"``: iterable of modes
    - ``"excitations_by_mode"``: mapping from signatures to counts

    """
    params = action.params
    if not isinstance(params, Mapping):
        raise TypeError(
            "NumberStateSource kernel expects action.params to be a mapping"
        )

    source_modes_obj = params.get("source_modes")
    if not isinstance(source_modes_obj, Iterable):
        raise TypeError(
            "NumberStateSource kernel expects params['source_modes'] to be an iterable"
        )

    source_modes = tuple(cast(ModeOpProtocol, mode) for mode in source_modes_obj)

    excitations_obj = params.get("excitations_by_mode")
    if not isinstance(excitations_obj, Mapping):
        raise TypeError(
            "NumberStateSource kernel expects params['excitations_by_mode'] "
            "to be a mapping"
        )

    excitations_by_mode: dict[Signature, int] = {}
    for key, value in excitations_obj.items():
        if not isinstance(value, int):
            raise TypeError("Excitation counts must be integers.")
        if value < 0:
            raise ValueError("Excitation counts must be nonnegative.")
        excitations_by_mode[cast(Signature, key)] = value

    return _NumberStateSourceParams(
        source_modes=source_modes,
        excitations_by_mode=excitations_by_mode,
    )


def _creator_word_for_number_state(
    *,
    source_modes: tuple[ModeOpProtocol, ...],
    excitations_by_mode: Mapping[Signature, int],
) -> tuple[LadderOpProtocol, ...]:
    r"""Construct a creation-operator word for a number state.

    Parameters
    ----------
    source_modes:
        Modes to populate with excitations.
    excitations_by_mode:
        Mapping from mode signature to excitation count.

    Returns
    -------
    tuple[LadderOpProtocol, ...]
        Ordered tuple of creation operators representing the number state.

    Raises
    ------
    ValueError
        If a mode does not provide a valid creation operator.

    Notes
    -----
    Each mode contributes ``n`` creation operators according to its
    excitation count, resulting in a normally ordered product.

    """
    creators: list[LadderOpProtocol] = []

    for mode in source_modes:
        n = excitations_by_mode.get(mode.signature, 0)
        if n == 0:
            continue

        create = mode.cre
        if create.kind != OperatorKind.CRE:
            raise ValueError(f"Expected creation operator for mode {mode.signature!r}.")

        for _ in range(n):
            creators.append(create)

    return tuple(creators)


def number_state_source_poly_ket(
    *,
    state: KetPolyState,
    action: DeviceActionProtocol,
    ctx: ApplyContextProtocol,
) -> KetPolyState:
    r"""Generate a polynomial ket representing a multimode number state.

    Parameters
    ----------
    state:
        Input ket state. Currently ignored by this source kernel.
    action:
        Device action containing source-mode and excitation data.
    ctx:
        Apply context (unused in this implementation).

    Returns
    -------
    KetPolyState
        Polynomial ket state constructed from creation operators acting
        on the vacuum.

    Notes
    -----
    - The output state is independent of the input ``state``.
    - The number state is constructed via repeated application of
      creation operators on the vacuum.

    """
    del state
    del ctx

    parsed = _parse_number_state_source_params(action)

    creators = _creator_word_for_number_state(
        source_modes=parsed.source_modes,
        excitations_by_mode=parsed.excitations_by_mode,
    )

    return KetPolyState.from_creators(creators)


def number_state_source_poly_density(
    *,
    state: DensityPolyState,
    action: DeviceActionProtocol,
    ctx: ApplyContextProtocol,
) -> DensityPolyState:
    """Emit a multimode pure number state and return it as a density state.

    Parameters
    ----------
    state : DensityPolyState
        Input density state. Currently ignored by this source kernel.
    action : DeviceActionProtocol
        Semantic device action containing source-mode and excitation data.
    ctx : ApplyContextProtocol
        Apply context forwarded to the ket kernel.

    Returns
    -------
    DensityPolyState
        Pure density state of the emitted number-state source.

    """
    del state

    ket_state = number_state_source_poly_ket(
        state=KetPolyState.vacuum(),
        action=action,
        ctx=ctx,
    )
    return cast(DensityPolyState, ket_state.to_density())
