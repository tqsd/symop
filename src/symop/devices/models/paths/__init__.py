r"""Semantic device model package.

This package groups concrete semantic device models used by the planning
layer of the device system.

Device models define the representation-independent behavior of optical
or measurement components. They typically specify port structure and
translate a device invocation into a semantic action, optional label
edits, and backend-facing parameters.

Subpackages are organized by model family, for example beam splitters,
detectors, filters, path operations, and sources.

Additional model families that would naturally belong here include phase
shifters, switches, lossy channels, polarization elements,
interferometers, nonlinear devices, modulators, resonators, and other
semantic building blocks for optical experiments.
"""
