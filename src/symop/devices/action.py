r"""Semantic device action representation.

Defines the :class:`DeviceAction` data structure, which captures the
result of device planning.

A device action encodes all information required for execution by the
runtime, including bound ports, device kind, parameters, and label edits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from symop.core.protocols.devices.label_edit import LabelEdit
from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.devices.types.device_kind import DeviceKind


@dataclass(frozen=True)
class DeviceAction:
    r"""Semantic description of a device application.

    Parameters
    ----------
    ports:
        Mapping from logical port names to bound path labels.
    kind:
        Device kind identifier.
    params:
        Device-specific parameters required for execution.
    edits:
        Sequence of label edits to be applied to the output state.
    selection:
        Optional device-specific selection or configuration object.
    requires_kernel:
        Whether runtime must dispatch to a representation-specific apply
        kernel before applying edits. Set to False for edit-only devices
        whose effect is fully captured by label updates.

    Notes
    -----
    - This object is produced during device planning and consumed by
      the runtime.
    - It is representation-independent and serves as an intermediate
      form between semantic devices and execution kernels.

    """

    ports: Mapping[str, PathProtocol]
    kind: DeviceKind
    params: Mapping[str, object]
    edits: Sequence[LabelEdit]
    selection: object | None = None
    requires_kernel: bool = True
