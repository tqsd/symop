r"""Registration of polynomial device kernels.

This module registers kernel functions for polynomial (CCR-based)
state representations with a kernel registry.

The registered kernels implement device-specific transformations
(e.g., sources and filters) acting on polynomial ket and density
states.

Notes
-----
- All kernels are registered under the ``POLY`` representation.
- Registration is type-driven based on device kind and input/output
  state kinds.
- This module should be imported for its side effects (kernel registration).

"""

from __future__ import annotations

from typing import Any, cast

from symop.core.types.rep_kind import POLY
from symop.core.types.state_kind import DENSITY, KET, StateKind
from symop.devices.protocols.kernel import KernelFn
from symop.devices.protocols.registry import (
    KernelRegistry as KernelRegistryProtocol,
)
from symop.devices.types.device_kind import DeviceKind
from symop.polynomial.kernels.devices.beamsplitter import (
    beamsplitter_poly_density,
    beamsplitter_poly_ket,
)
from symop.polynomial.kernels.devices.filter import (
    filter_poly_density,
    filter_poly_ket,
)
from symop.polynomial.kernels.devices.number_state_source import (
    number_state_source_poly_density,
    number_state_source_poly_ket,
)
from symop.polynomial.kernels.devices.phase_shifter import (
    phase_shifter_poly_density,
    phase_shifter_poly_ket,
)


def register_polynomial_kernels(*, device_registry: KernelRegistryProtocol) -> None:
    r"""Register polynomial kernels for supported device types.

    Parameters
    ----------
    device_registry:
        Kernel registry used to associate device kinds and state
        transformations with kernel implementations.

    Returns
    -------
    None

    Notes
    -----
    The following kernel mappings are registered:

    - NUMBER_STATE_SOURCE:
        - KET → KET via :func:`number_state_source_poly_ket`
        - DENSITY → DENSITY via :func:`number_state_source_poly_density`

    - SPECTRAL_FILTER:
        - DENSITY → DENSITY via :func:`filter_poly_density`
        - KET → DENSITY via :func:`filter_poly_ket`

    - POLARIZING_FILTER:
        - DENSITY → DENSITY via :func:`filter_poly_density`

    Each registration is performed for the ``POLY`` representation.

    """

    def _reg(
        device_kind: DeviceKind,
        in_kind: StateKind,
        out_kind: StateKind,
        fn: Any,
    ) -> None:
        """Register a single device kernel entry."""
        device_registry.register(
            device_kind=device_kind,
            rep=POLY,
            in_kind=in_kind,
            out_kind=out_kind,
            fn=cast(KernelFn, fn),
        )

    _reg(
        device_kind=DeviceKind.NUMBER_STATE_SOURCE,
        in_kind=KET,
        out_kind=KET,
        fn=number_state_source_poly_ket,
    )
    _reg(
        device_kind=DeviceKind.NUMBER_STATE_SOURCE,
        in_kind=DENSITY,
        out_kind=DENSITY,
        fn=number_state_source_poly_density,
    )

    _reg(
        device_kind=DeviceKind.SPECTRAL_FILTER,
        in_kind=DENSITY,
        out_kind=DENSITY,
        fn=filter_poly_density,
    )
    _reg(
        device_kind=DeviceKind.SPECTRAL_FILTER,
        in_kind=KET,
        out_kind=DENSITY,
        fn=filter_poly_ket,
    )
    _reg(
        device_kind=DeviceKind.POLARIZING_FILTER,
        in_kind=DENSITY,
        out_kind=DENSITY,
        fn=filter_poly_density,
    )
    _reg(
        device_kind=DeviceKind.BEAMSPLITTER,
        in_kind=KET,
        out_kind=KET,
        fn=beamsplitter_poly_ket,
    )
    _reg(
        device_kind=DeviceKind.BEAMSPLITTER,
        in_kind=DENSITY,
        out_kind=DENSITY,
        fn=beamsplitter_poly_density,
    )
    _reg(
        device_kind=DeviceKind.PHASE_SHIFTER,
        in_kind=KET,
        out_kind=KET,
        fn=phase_shifter_poly_ket,
    )
    _reg(
        device_kind=DeviceKind.PHASE_SHIFTER,
        in_kind=DENSITY,
        out_kind=DENSITY,
        fn=phase_shifter_poly_density,
    )
