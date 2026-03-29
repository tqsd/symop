from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
ComplexArray: TypeAlias = npt.NDArray[np.complexfloating[Any, Any]]
RCArray: TypeAlias = npt.NDArray[np.complex128] | npt.NDArray[np.float64]
