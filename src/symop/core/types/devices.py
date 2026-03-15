from __future__ import annotations

from enum import StrEnum


class DeviceKind(StrEnum):
    SPECTRAL_FILTER = "spectral_filter"
    POLARIZING_FILTER = "polarizing_filter"
    BEAMSPLITTER = "beamsplitter"
    MZI = "mzi"
