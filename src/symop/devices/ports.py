r"""Device port specification types.

Defines the static interface for device ports, including direction and
binding requirements.

These specifications are used by the runtime to validate port mappings
provided during device application and to ensure consistency between
device definitions and their usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PORT_DIR = Literal["in", "out", "inout"]


@dataclass(frozen=True)
class PortSpec:
    """Stable declared port interface for a device.

    Attributes
    ----------
    name:
        Logical port name used when binding paths during device application.
    direction:
        Port direction in the semantic device model.
    required:
        Whether the caller must explicitly bind this port.

    """

    name: str
    direction: PORT_DIR
    required: bool = True
