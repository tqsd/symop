from __future__ import annotations

from typing import Final, Literal

StateKind = Literal["ket", "density"]

KET: Final[StateKind] = "ket"
DENSITY: Final[StateKind] = "density"
