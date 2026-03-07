from __future__ import annotations

from enum import StrEnum


class OperatorKind(StrEnum):
    ANN = "a"
    ANNIHILATE = "a"
    CRE = "adag"
    CREATE = "adag"
