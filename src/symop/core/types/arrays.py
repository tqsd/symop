from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complexfloating[Any, Any]]
RCArray = npt.NDArray[np.complex128] | npt.NDArray[np.float64]
