from __future__ import annotations

from enum import StrEnum


class DeviceKind(StrEnum):
    NUMBER_STATE_SOURCE = "number_state_source"
    SPECTRAL_FILTER = "spectral_filter"
    POLARIZING_FILTER = "polarizing_filter"
    BEAMSPLITTER = "beamsplitter"
    MZI = "mzi"
