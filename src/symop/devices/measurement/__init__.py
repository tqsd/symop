r"""Measurement subsystem.

This package defines the semantic measurement layer of the device
framework.

It provides representation-independent abstractions for describing,
planning, and structuring quantum measurements. The components here
separate *what* is measured from *how* it is evaluated by backend-
specific implementations.

Main components
---------------
action
    Semantic measurement actions such as observe, detect, and postselect.
base
    Abstract base classes for measurement devices.
outcomes
    Concrete classical outcome representations.
resolution
    Structures describing how measurement results are grouped and reported.
resolved
    Containers for resolved measurement probabilities and derived quantities.
result
    Result objects returned after measurement evaluation.
specs
    Semantic measurement specifications (projective, POVM, instrument).
target
    Definitions of which subsystem or modes are measured.

Notes
-----
The measurement layer is fully backend-agnostic. It constructs semantic
descriptions and actions that are later interpreted by measurement
kernels for a specific state representation.

This separation enables:

- reuse of measurement definitions across representations,
- consistent handling of observation, detection, and postselection,
- and extensibility toward more general measurement models such as
  POVMs and instruments.

"""
