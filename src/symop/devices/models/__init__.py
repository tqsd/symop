r"""Device models.

Provides concrete implementations of semantic devices used in the
simulation framework, such as sources, filters, and other optical
components.

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

Notes
-----
- Device models are representation-agnostic and operate at the semantic level.
- Execution is delegated to the runtime via registered kernels.
- New device types should be added as separate modules and registered
  through the device and kernel registries.

"""
