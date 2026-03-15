from __future__ import annotations

from typing import Final, Literal

StateKind = Literal["ket", "density"]

KET: Final[Literal["ket"]] = "ket"
DENSITY: Final[Literal["density"]] = "density"
