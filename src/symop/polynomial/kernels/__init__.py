r"""Polynomial measurement kernel package.

This package contains backend implementations of measurement operations
for the polynomial state representation.

Measurement kernels implement the runtime evaluation of semantic
measurement actions, translating high-level measurement intents into
concrete operations on polynomial states.

Supported measurement intents include:

- ``observe``:
    Evaluate full probability distributions without modifying the state.
- ``detect``:
    Sample a measurement outcome and return the corresponding
    post-measurement state.
- ``postselect``:
    Condition on a specified outcome and return the corresponding branch.

Structure
---------
The package is organized by measurement type:

- ``number``:
    Photon-number measurement kernels, including single-target and
    joint per-port measurements.

- ``registry``:
    Registration helpers that bind measurement kernels to the global
    :class:`MeasurementKernelRegistry`.

Notes
-----
Measurement kernels are selected dynamically based on:

- device kind (e.g. ``NUMBER_DETECTOR``)
- measurement intent (observe / detect / postselect)
- representation kind (e.g. ``POLY``)
- input state kind (ket or density)

This package only contains polynomial-specific implementations. Other
representations (e.g. Gaussian) are expected to provide analogous
packages.

"""
