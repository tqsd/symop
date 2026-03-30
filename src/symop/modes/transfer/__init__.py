r"""Transfer functions and transfer-function algebra.

This package provides frequency-domain transfer functions and
combinators used to transform mode envelopes.

Overview
--------
A transfer function :math:`H(\omega)` acts multiplicatively in the
frequency domain:

.. math::

    Z_{\mathrm{out}}(\omega)
    =
    H(\omega)\,Z_{\mathrm{in}}(\omega).

The package supports two execution regimes:

- **Gaussian-closed (analytic) transfers**
  Transfers that admit a closed-form action on Gaussian envelopes.
  These preserve an analytic representation and avoid numerical FFT
  evaluation.

- **General transfers**
  Arbitrary transfer functions evaluated numerically via spectral
  sampling (FFT-based filtering).

Transfer algebra
----------------
Transfer functions can be combined to build more complex responses:

- :class:`Cascade`
    Sequential composition (product) of multiple transfer functions

    .. math::

        H(\omega) = H_n(\omega)\cdots H_2(\omega) H_1(\omega)

Available transfers
-------------------
Gaussian-closed transfers (analytic):

- :class:`GaussianLowpass`
- :class:`GaussianBandpass`
- :class:`GaussianHighpass`
- :class:`ConstantPhase`
- :class:`TimeDelay`

General transfers (numerical fallback):

- :class:`RectBandpass`
- :class:`SuperGaussianBandpass`
- :class:`QuadraticDispersion`

Notes
-----
- All transfers operate on **mode descriptors** (envelopes), not directly
  on quantum states.
- Power loss (transmissivity :math:`\eta`) is handled separately from the
  envelope representation.
- When no analytic path is available, transfers are evaluated via
  :class:`~symop.modes.envelopes.filtered.FilteredEnvelope`.

"""

from .cascade import Cascade
from .gaussian import (
    ConstantPhase,
    GaussianBandpass,
    GaussianHighpass,
    GaussianLowpass,
    TimeDelay,
)
from .quadratic_dispersion import QuadraticDispersion
from .rect_bandpass import RectBandpass
from .supergaussian_bandpass import SuperGaussianBandpass

__all__ = [
    "GaussianLowpass",
    "GaussianHighpass",
    "GaussianBandpass",
    "RectBandpass",
    "SuperGaussianBandpass",
    "ConstantPhase",
    "TimeDelay",
    "QuadraticDispersion",
    "Cascade",
]
