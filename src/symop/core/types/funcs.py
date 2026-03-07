from __future__ import annotations

from collections.abc import Callable

from symop.modes.types import FloatArray, RCArray

TimeFunc = Callable[[FloatArray], RCArray]
FreqFunc = Callable[[FloatArray], RCArray]
