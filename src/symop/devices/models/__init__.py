r"""Device models.

Provides concrete implementations of semantic devices used in the
simulation framework, such as sources, linear-optical elements,
filters, and measurement components.

The modules in this package define device behavior at the planning level,
producing :class:`DeviceAction` objects that are later executed by
representation-specific kernels.

Subpackages
-----------
sources
    Devices that generate quantum states (e.g., photon sources).
filters
    Devices that transform states via selection or attenuation
    (e.g., spectral or polarization filters).
beamsplitters
    Two-path mixing devices implementing linear mode coupling.
phase_shifters
    Single-path devices applying constant phase rotations.

Notes
-----
- Device models are representation-agnostic and operate at the semantic level.
- Execution is delegated to the runtime via registered kernels.
- Linear-optical elements (e.g., beamsplitters, phase shifters) define
  unitary transformations at the operator level, realized by backend kernels.
- New device types should be added as separate modules and registered
  through the device and kernel registries.

"""

from .beamsplitters import BeamSplitter
from .filters import PolarizingFilter, SpectralFilter
from .phase_shifters import PhaseShifter
from .sources import NumberStateSource

__all__ = [
    "BeamSplitter",
    "SpectralFilter",
    "PolarizingFilter",
    "PhaseShifter",
    "NumberStateSource",
]
