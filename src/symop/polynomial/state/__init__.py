"""State containers for symbolic polynomial representations.

This package defines state objects that wrap symbolic CCR polynomial
structures (e.g., density polynomials) together with convenience
operations such as normalization, mode relabeling, and channel
application.

The classes in this package provide the interface used by higher-level
device kernels operating on symbolic quantum states.
"""

from .density import DensityPolyState
from .ket import KetPolyState

__all__ = ["DensityPolyState", "KetPolyState"]
