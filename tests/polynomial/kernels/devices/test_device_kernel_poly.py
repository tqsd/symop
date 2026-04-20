from __future__ import annotations

import unittest

from symop.core.types.rep_kind import POLY
from symop.core.types.state_kind import DENSITY, KET
from symop.devices.types.device_kind import DeviceKind
from symop.polynomial.kernels.devices.registry import register_polynomial_kernels


class FakeKernelRegistry:
    def __init__(self) -> None:
        self.calls = []

    def register(self, *, device_kind, rep, in_kind, out_kind, fn) -> None:
        self.calls.append(
            {
                "device_kind": device_kind,
                "rep": rep,
                "in_kind": in_kind,
                "out_kind": out_kind,
                "fn": fn,
            }
        )


class TestPolynomialKernelRegistration(unittest.TestCase):
    def test_registers_expected_entries(self) -> None:
        registry = FakeKernelRegistry()

        register_polynomial_kernels(device_registry=registry)

        keys = {
            (c["device_kind"], c["rep"], c["in_kind"], c["out_kind"])
            for c in registry.calls
        }

        self.assertIn((DeviceKind.NUMBER_STATE_SOURCE, POLY, KET, KET), keys)
        self.assertIn((DeviceKind.NUMBER_STATE_SOURCE, POLY, DENSITY, DENSITY), keys)
        self.assertIn((DeviceKind.SPECTRAL_FILTER, POLY, DENSITY, DENSITY), keys)
        self.assertIn((DeviceKind.SPECTRAL_FILTER, POLY, KET, DENSITY), keys)
        self.assertIn((DeviceKind.POLARIZING_FILTER, POLY, DENSITY, DENSITY), keys)
        self.assertIn((DeviceKind.BEAMSPLITTER, POLY, KET, KET), keys)
        self.assertIn((DeviceKind.BEAMSPLITTER, POLY, DENSITY, DENSITY), keys)
        self.assertIn((DeviceKind.PHASE_SHIFTER, POLY, KET, KET), keys)
        self.assertIn((DeviceKind.PHASE_SHIFTER, POLY, DENSITY, DENSITY), keys)
