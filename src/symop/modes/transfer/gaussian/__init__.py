r"""Gaussian-closed transfer functions.

This subpackage provides transfer functions that admit a **closed-form
analytic action** on Gaussian-closed envelopes.

Overview
--------
All transfers exposed here implement the
:class:`~symop.modes.protocols.transfer.SupportsGaussianClosedTransfer`
capability via
:class:`~symop.modes.transfer.gaussian.base.GaussianClosedTransferBase`.

Such transfers can be applied without numerical FFT-based filtering,
preserving an analytic representation of the envelope.

The action follows two main patterns:

- **Expansion-based filtering**
  The transfer is represented as a finite sum of Gaussian atoms

  .. math::

      H(\omega) = c_0 + \sum_k c_k G_k(\omega),

  and applied analytically via
  :class:`~symop.modes.transfer.gaussian.formalism.GaussianTransferExpansion`.
  This typically produces a Gaussian *mixture*.

- **Parameter transforms**
  The transfer modifies envelope parameters directly without changing
  its Gaussian structure. This preserves a single Gaussian representation
  and avoids mixture growth.

Available transfers
-------------------
The following Gaussian-compatible transfer functions are provided:

- :class:`ConstantPhase`
    Frequency-independent phase shift

    .. math::

        H(\omega) = e^{i\phi_0}

- :class:`TimeDelay`
    Time shift in the frequency domain

    .. math::

        H(\omega) = e^{-i\omega\tau}

    This corresponds to a translation of the envelope in time.

- :class:`GaussianLowpass`
    Gaussian low-pass filter

- :class:`GaussianBandpass`
    Gaussian band-pass filter

- :class:`GaussianHighpass`
    Complement of a Gaussian low-pass

Notes
-----
- These transfers operate on **mode descriptors** (normalized envelopes).
- Any loss (transmissivity :math:`\eta`) is returned separately and should
  be applied at the quantum state level.
- Expansion-based transfers may increase representation complexity
  (Gaussian mixtures).
- Parameter transforms preserve a single Gaussian structure.
- Transfers that are not representable in this formalism fall back to the
  numerical filtering path (see :mod:`symop.modes.transfer.apply`).

"""

from .bandpass import GaussianBandpass
from .constant_phase import ConstantPhase
from .highpass import GaussianHighpass
from .lowpass import GaussianLowpass
from .time_delay import TimeDelay

__all__ = [
    "ConstantPhase",
    "GaussianBandpass",
    "GaussianHighpass",
    "GaussianLowpass",
    "TimeDelay",
]
