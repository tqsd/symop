r"""Filtering devices for mode descriptors.

This package provides devices that modify mode descriptors by applying
frequency- or polarization-dependent filtering operations.

Filtering devices operate on the *semantic layer* of the simulation by
transforming mode labels and recording physical attenuation parameters.
The actual physical channels are applied later by the backend runtime
kernels.

Two filtering devices are currently provided:

SpectralFilter
    Applies a spectral transfer function to the envelope of every mode
    on the selected input path. The resulting envelope and transmission
    efficiency are computed during planning, while the backend kernel
    realizes the corresponding attenuation channel.

PolarizingFilter
    Applies a polarization-selective filtering operation. The planner
    updates polarization-related mode descriptors and records the
    corresponding transmissivities for execution by the backend kernel.

Notes
-----
Filtering devices follow the standard device execution pipeline:

1. **Planning stage**
   The device inspects mode descriptors and computes descriptor updates
   together with device parameters (e.g., transmissivities).

2. **Kernel execution**
   Backend kernels apply the corresponding physical channels to the
   state representation (e.g., polynomial, Gaussian, or other supported
   representations).

The devices in this module operate purely at the descriptor level and
do not directly manipulate quantum states.

"""

from .polarizing_filter import PolarizingFilter
from .spectral_filter import SpectralFilter

__all__ = ["SpectralFilter", "PolarizingFilter"]
