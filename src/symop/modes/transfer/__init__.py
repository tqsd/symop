"""Transfer functions and transfer-function algebra.

This package contains frequency-domain transfer functions and combinators
(e.g. products and compositions) used by envelope/filter machinery.
"""

from .cascade import Cascade
from .constant_phase import ConstantPhase
from .gaussian_bandpass import GaussianBandpass
from .gaussian_highpass import GaussianHighpass
from .gaussian_lowpass import GaussianLowpass
from .product import Product
from .quadratic_dispersion import QuadraticDispersion
from .rect_bandpass import RectBandpass
from .supergaussian_bandpass import SuperGaussianBandpass
from .time_delay import TimeDelay

__all__ = [
    "GaussianLowpass",
    "GaussianHighpass",
    "GaussianBandpass",
    "RectBandpass",
    "SuperGaussianBandpass",
    "ConstantPhase",
    "TimeDelay",
    "QuadraticDispersion",
    "Product",
    "Cascade",
]
