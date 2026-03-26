r"""Registration of polynomial measurement kernels.

This module registers polynomial-representation measurement kernels with
a :class:`MeasurementKernelRegistry`.

The registered kernels implement number measurement for the
:class:`DeviceKind.NUMBER_DETECTOR` device, covering the following
measurement intents:

- ``observe``:
    Compute full probability distributions without modifying the state.
- ``detect``:
    Sample an outcome and return a post-measurement state.
- ``postselect``:
    Condition on a specified outcome and return the corresponding branch.

Both ket and density representations are supported where applicable.
For operations that require state collapse (``detect`` and
``postselect``), only density-state kernels are registered.

Notes
-----
This function wires semantic measurement actions to concrete backend
implementations for the polynomial representation (``rep=POLY``).

The registry dispatch key consists of:

- device kind (e.g. ``NUMBER_DETECTOR``)
- measurement intent (``observe``, ``detect``, ``postselect``)
- representation kind (``POLY``)
- input state kind (``KET`` or ``DENSITY``)

This allows the runtime to resolve the correct kernel dynamically
based on the input state and requested operation.

"""

from __future__ import annotations

from typing import Any

from symop.core.types.rep_kind import POLY
from symop.core.types.state_kind import DENSITY, KET, StateKind
from symop.devices.protocols.registry import MeasurementKernelRegistry
from symop.devices.types.device_kind import DeviceKind
from symop.devices.types.measurement import MeasurementIntent
from symop.polynomial.kernels.measurements.number.detect import (
    detect_number_detector_poly_density,
)
from symop.polynomial.kernels.measurements.number.observe import (
    observe_number_detector_poly_density,
    observe_number_detector_poly_ket,
)
from symop.polynomial.kernels.measurements.number.postselect import (
    postselect_number_detector_poly_density,
)


def register_polynomial_measurement_kernels(
    *,
    measurement_registry: MeasurementKernelRegistry,
) -> None:
    r"""Register polynomial number-measurement kernels.

    Parameters
    ----------
    measurement_registry:
        Registry into which the polynomial measurement kernels are
        registered.

    Notes
    -----
    The following registrations are performed for
    :class:`DeviceKind.NUMBER_DETECTOR`:

    - ``observe``:
        - ket input → :func:`observe_number_detector_poly_ket`
        - density input → :func:`observe_number_detector_poly_density`

    - ``detect``:
        - density input → :func:`detect_number_detector_poly_density`

    - ``postselect``:
        - density input → :func:`postselect_number_detector_poly_density`

    Ket-based kernels are provided only for observation, since detection
    and postselection require state collapse, which is canonically handled
    at the density-state level.

    This function should be called during initialization of the polynomial
    backend to ensure that measurement actions are correctly dispatched.

    """

    def _reg(
        device_kind: DeviceKind,
        in_kind: StateKind,
        intent: MeasurementIntent,
        fn: Any,
    ) -> None:
        """Register a single measurement kernel entry."""
        measurement_registry.register(
            device_kind=device_kind,
            intent=intent,
            rep=POLY,
            in_kind=in_kind,
            fn=fn,
        )

    # NUMBER DETECTOR ---------------------------------------------

    # observe: fast ket path + canonical density path
    _reg(
        device_kind=DeviceKind.NUMBER_DETECTOR,
        in_kind=KET,
        intent="observe",
        fn=observe_number_detector_poly_ket,
    )
    _reg(
        device_kind=DeviceKind.NUMBER_DETECTOR,
        in_kind=DENSITY,
        intent="observe",
        fn=observe_number_detector_poly_density,
    )

    # detect: canonical density path only
    _reg(
        device_kind=DeviceKind.NUMBER_DETECTOR,
        in_kind=DENSITY,
        intent="detect",
        fn=detect_number_detector_poly_density,
    )

    # postselect: canonical density path only
    _reg(
        device_kind=DeviceKind.NUMBER_DETECTOR,
        in_kind=DENSITY,
        intent="postselect",
        fn=postselect_number_detector_poly_density,
    )
