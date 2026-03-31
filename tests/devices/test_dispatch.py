from __future__ import annotations

import unittest

from symop.devices.dispatch import (
    dispatch_apply,
    dispatch_detect,
    dispatch_measurement,
    dispatch_observe,
    dispatch_postselect,
)
from symop.devices.registry import KernelRegistry, MeasurementKernelRegistry
from symop.devices.types.device_kind import DeviceKind
from tests.devices.support.fakes import (
    FakeDensityState,
    FakeDevice,
    FakeMeasurementAction,
    FakeMeasurementDevice,
    FakeState,
    FakeStateWithToDensity,
    RecordingKernel,
)


class TestDispatchApply(unittest.TestCase):
    def test_dispatch_apply_uses_input_state_kind_when_out_kind_is_none(self) -> None:
        registry = KernelRegistry()
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = object()
        ctx = object()
        result_state = FakeState(rep_kind="poly", state_kind="ket")
        kernel = RecordingKernel(result=result_state)

        registry.register(
            device_kind=DeviceKind.BEAMSPLITTER,
            rep="poly",
            in_kind="ket",
            out_kind="ket",
            fn=kernel,
        )

        device = FakeDevice(
            kind=DeviceKind.BEAMSPLITTER,
            port_specs=(),
            action=object(),
        )

        result = dispatch_apply(
            registry=registry,
            device=device,
            state=state,
            action=action,
            ctx=ctx,
        )

        self.assertIs(result, result_state)
        self.assertEqual(len(kernel.calls), 1)
        self.assertIs(kernel.calls[0]["state"], state)
        self.assertIs(kernel.calls[0]["action"], action)
        self.assertIs(kernel.calls[0]["ctx"], ctx)

    def test_dispatch_apply_uses_explicit_out_kind(self) -> None:
        registry = KernelRegistry()
        state = FakeState(rep_kind="poly", state_kind="ket")
        result_state = FakeDensityState(rep_kind="poly", state_kind="density")
        kernel = RecordingKernel(result=result_state)

        registry.register(
            device_kind=DeviceKind.BEAMSPLITTER,
            rep="poly",
            in_kind="ket",
            out_kind="density",
            fn=kernel,
        )

        device = FakeDevice(
            kind=DeviceKind.BEAMSPLITTER,
            port_specs=(),
            action=object(),
        )

        result = dispatch_apply(
            registry=registry,
            device=device,
            state=state,
            action=object(),
            ctx=object(),
            out_kind="density",
        )

        self.assertIs(result, result_state)


class TestDispatchMeasurement(unittest.TestCase):
    def test_dispatch_measurement_resolves_directly_when_kernel_exists(self) -> None:
        registry = MeasurementKernelRegistry()
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = FakeMeasurementAction(intent="observe")
        ctx = object()
        result_object = object()
        kernel = RecordingKernel(result=result_object)

        registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="observe",
            rep="poly",
            in_kind="ket",
            fn=kernel,
        )

        device = FakeMeasurementDevice(
            kind=DeviceKind.NUMBER_DETECTOR,
            port_specs=(),
            observe_action=action,
            detect_action=action,
            postselect_action=action,
        )

        result = dispatch_measurement(
            registry=registry,
            device=device,
            state=state,
            action=action,
            ctx=ctx,
        )

        self.assertIs(result, result_object)
        self.assertEqual(len(kernel.calls), 1)
        self.assertIs(kernel.calls[0]["state"], state)
        self.assertIs(kernel.calls[0]["action"], action)
        self.assertIs(kernel.calls[0]["ctx"], ctx)

    def test_dispatch_measurement_falls_back_to_density(self) -> None:
        registry = MeasurementKernelRegistry()
        density_state = FakeDensityState(rep_kind="poly", state_kind="density")
        state = FakeStateWithToDensity(
            rep_kind="poly",
            state_kind="ket",
            density_state=density_state,
        )
        action = FakeMeasurementAction(intent="observe")
        ctx = object()
        result_object = object()
        kernel = RecordingKernel(result=result_object)

        registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="observe",
            rep="poly",
            in_kind="density",
            fn=kernel,
        )

        device = FakeMeasurementDevice(
            kind=DeviceKind.NUMBER_DETECTOR,
            port_specs=(),
            observe_action=action,
            detect_action=action,
            postselect_action=action,
        )

        result = dispatch_measurement(
            registry=registry,
            device=device,
            state=state,
            action=action,
            ctx=ctx,
        )

        self.assertIs(result, result_object)
        self.assertEqual(state.to_density_calls, 1)
        self.assertEqual(len(kernel.calls), 1)
        self.assertIs(kernel.calls[0]["state"], density_state)

    def test_dispatch_measurement_reraises_when_no_kernel_and_no_density_fallback(self) -> None:
        registry = MeasurementKernelRegistry()
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = FakeMeasurementAction(intent="observe")

        device = FakeMeasurementDevice(
            kind=DeviceKind.NUMBER_DETECTOR,
            port_specs=(),
            observe_action=action,
            detect_action=action,
            postselect_action=action,
        )

        with self.assertRaises(KeyError):
            dispatch_measurement(
                registry=registry,
                device=device,
                state=state,
                action=action,
                ctx=object(),
            )

    def test_dispatch_observe_delegates_to_measurement_dispatch(self) -> None:
        registry = MeasurementKernelRegistry()
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = FakeMeasurementAction(intent="observe")
        result_object = object()
        kernel = RecordingKernel(result=result_object)

        registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="observe",
            rep="poly",
            in_kind="ket",
            fn=kernel,
        )

        device = FakeMeasurementDevice(
            kind=DeviceKind.NUMBER_DETECTOR,
            port_specs=(),
            observe_action=action,
            detect_action=action,
            postselect_action=action,
        )

        result = dispatch_observe(
            registry=registry,
            device=device,
            state=state,
            action=action,
            ctx=object(),
        )

        self.assertIs(result, result_object)

    def test_dispatch_detect_delegates_to_measurement_dispatch(self) -> None:
        registry = MeasurementKernelRegistry()
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = FakeMeasurementAction(intent="detect", destructive=True)
        result_object = object()
        kernel = RecordingKernel(result=result_object)

        registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="detect",
            rep="poly",
            in_kind="ket",
            fn=kernel,
        )

        device = FakeMeasurementDevice(
            kind=DeviceKind.NUMBER_DETECTOR,
            port_specs=(),
            observe_action=action,
            detect_action=action,
            postselect_action=action,
        )

        result = dispatch_detect(
            registry=registry,
            device=device,
            state=state,
            action=action,
            ctx=object(),
        )

        self.assertIs(result, result_object)

    def test_dispatch_postselect_delegates_to_measurement_dispatch(self) -> None:
        registry = MeasurementKernelRegistry()
        state = FakeState(rep_kind="poly", state_kind="density")
        action = FakeMeasurementAction(intent="postselect", outcome="keep")
        result_object = object()
        kernel = RecordingKernel(result=result_object)

        registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="postselect",
            rep="poly",
            in_kind="density",
            fn=kernel,
        )

        device = FakeMeasurementDevice(
            kind=DeviceKind.NUMBER_DETECTOR,
            port_specs=(),
            observe_action=action,
            detect_action=action,
            postselect_action=action,
        )

        result = dispatch_postselect(
            registry=registry,
            device=device,
            state=state,
            action=action,
            ctx=object(),
        )

        self.assertIs(result, result_object)
