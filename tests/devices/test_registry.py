from __future__ import annotations

import unittest

from symop.devices.registry import KernelRegistry, MeasurementKernelRegistry
from symop.devices.types.device_kind import DeviceKind


def dummy_kernel(*, state: object, action: object, ctx: object) -> object:
    return state


class TestKernelRegistry(unittest.TestCase):
    def test_register_and_resolve_returns_same_function(self) -> None:
        registry = KernelRegistry()

        registry.register(
            device_kind=DeviceKind.BEAMSPLITTER,
            rep="poly",
            in_kind="ket",
            out_kind="ket",
            fn=dummy_kernel,
        )

        resolved = registry.resolve(
            device_kind=DeviceKind.BEAMSPLITTER,
            rep="poly",
            in_kind="ket",
            out_kind="ket",
        )

        self.assertIs(resolved, dummy_kernel)

    def test_duplicate_registration_raises_key_error(self) -> None:
        registry = KernelRegistry()

        registry.register(
            device_kind=DeviceKind.BEAMSPLITTER,
            rep="poly",
            in_kind="ket",
            out_kind="ket",
            fn=dummy_kernel,
        )

        with self.assertRaises(KeyError):
            registry.register(
                device_kind=DeviceKind.BEAMSPLITTER,
                rep="poly",
                in_kind="ket",
                out_kind="ket",
                fn=dummy_kernel,
            )

    def test_missing_resolution_raises_key_error(self) -> None:
        registry = KernelRegistry()

        with self.assertRaises(KeyError):
            registry.resolve(
                device_kind=DeviceKind.BEAMSPLITTER,
                rep="poly",
                in_kind="ket",
                out_kind="ket",
            )

    def test_missing_resolution_lists_available_keys_for_same_device(self) -> None:
        registry = KernelRegistry()

        registry.register(
            device_kind=DeviceKind.BEAMSPLITTER,
            rep="poly",
            in_kind="ket",
            out_kind="density",
            fn=dummy_kernel,
        )

        with self.assertRaises(KeyError) as cm:
            registry.resolve(
                device_kind=DeviceKind.BEAMSPLITTER,
                rep="poly",
                in_kind="ket",
                out_kind="ket",
            )

        message = str(cm.exception)
        self.assertIn("Available for this device", message)
        self.assertIn("beamsplitter", message)
        self.assertIn("density", message)


class TestMeasurementKernelRegistry(unittest.TestCase):
    def test_register_and_resolve_returns_same_function(self) -> None:
        registry = MeasurementKernelRegistry()

        registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="observe",
            rep="poly",
            in_kind="density",
            fn=dummy_kernel,
        )

        resolved = registry.resolve(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="observe",
            rep="poly",
            in_kind="density",
        )

        self.assertIs(resolved, dummy_kernel)

    def test_duplicate_registration_raises_key_error(self) -> None:
        registry = MeasurementKernelRegistry()

        registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="observe",
            rep="poly",
            in_kind="density",
            fn=dummy_kernel,
        )

        with self.assertRaises(KeyError):
            registry.register(
                device_kind=DeviceKind.NUMBER_DETECTOR,
                intent="observe",
                rep="poly",
                in_kind="density",
                fn=dummy_kernel,
            )

    def test_missing_resolution_raises_key_error(self) -> None:
        registry = MeasurementKernelRegistry()

        with self.assertRaises(KeyError):
            registry.resolve(
                device_kind=DeviceKind.NUMBER_DETECTOR,
                intent="observe",
                rep="poly",
                in_kind="density",
            )

    def test_missing_resolution_lists_available_keys_for_same_device(self) -> None:
        registry = MeasurementKernelRegistry()

        registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="detect",
            rep="poly",
            in_kind="density",
            fn=dummy_kernel,
        )

        with self.assertRaises(KeyError) as cm:
            registry.resolve(
                device_kind=DeviceKind.NUMBER_DETECTOR,
                intent="observe",
                rep="poly",
                in_kind="density",
            )

        message = str(cm.exception)
        self.assertIn("Available for this device", message)
        self.assertIn("number_detector", message)
        self.assertIn("detect", message)
