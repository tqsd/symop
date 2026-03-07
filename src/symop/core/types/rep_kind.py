from __future__ import annotations

from typing import Final, Literal

RepKind = Literal["poly", "gaussian"]

POLY: Final[RepKind] = "poly"
GAUSSIAN: Final[RepKind] = "gaussian"
