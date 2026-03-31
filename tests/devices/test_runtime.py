from __future__ import annotations

import unittest

from symop.devices.action import DeviceAction
from symop.devices.ports import PortSpec
from symop.devices.registry import KernelRegistry, MeasurementKernelRegistry
from symop.devices.runtime import DeviceRuntime
from symop.devices.types.device_kind import DeviceKind
from tests.devices.support.fakes import (
    FakeDevice,
    FakeEditableState,
    FakeMeasurementAction,
    FakeMeasurementDevice,
    FakePath,
    FakeState,
    RecordingKernel,
)


class TestDeviceRuntimeApply(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = DeviceRuntime(
            device_registry=KernelRegistry(),
            measurement_registry=MeasurementKernelRegistry(),
        )

    def test_apply_raises_for_unknown_ports(self) -> None:
        device = FakeDevice(
            kind=DeviceKind.BEAMSPLITTER,
            port_specs=(PortSpec(name="in", direction="in"),),
            action=DeviceAction(
                ports={"in": FakePath("p0")},
                kind=DeviceKind.BEAMSPLITTER,
                params={},
                edits=(),
            ),
        )

        with self.assertRaises(KeyError):
            self.runtime.apply(
                device=device,
                state=FakeState(),
                ports={"unknown": FakePath("p0")},
            )

    def test_apply_raises_for_missing_required_ports(self) -> None:
        device = FakeDevice(
            kind=DeviceKind.BEAMSPLITTER,
            port_specs=(
                PortSpec(name="in", direction="in", required=True),
                PortSpec(name="out", direction="out", required=True),
            ),
            action=DeviceAction(
                ports={"in": FakePath("p0"), "out": FakePath("p1")},
                kind=DeviceKind.BEAMSPLITTER,
                params={},
                edits=(),
            ),
        )

        with self.assertRaises(KeyError):
            self.runtime.apply(
                device=device,
                state=FakeState(),
                ports={"in": FakePath("p0")},
            )

    def test_apply_dispatches_kernel_when_required(self) -> None:
        input_state = FakeState(rep_kind="poly", state_kind="ket")
        output_state = FakeState(rep_kind="poly", state_kind="ket")
        action = DeviceAction(
            ports={"in": FakePath("p0")},
            kind=DeviceKind.BEAMSPLITTER,
            params={"theta": 0.5},
            edits=(),
            requires_kernel=True,
        )
        device = FakeDevice(
            kind=DeviceKind.BEAMSPLITTER,
            port_specs=(PortSpec(name="in", direction="in"),),
            action=action,
        )
        kernel = RecordingKernel(result=output_state)

        self.runtime.device_registry.register(
            device_kind=DeviceKind.BEAMSPLITTER,
            rep="poly",
            in_kind="ket",
            out_kind="ket",
            fn=kernel,
        )

        result = self.runtime.apply(
            device=device,
            state=input_state,
            ports={"in": FakePath("p0")},
            selection={"x": 1},
        )

        self.assertIs(result, output_state)
        self.assertEqual(len(device.plan_calls), 1)
        self.assertIs(device.plan_calls[0]["state"], input_state)
        self.assertEqual(device.plan_calls[0]["selection"], {"x": 1})
        self.assertEqual(len(kernel.calls), 1)
        self.assertIs(kernel.calls[0]["state"], input_state)
        self.assertIs(kernel.calls[0]["action"], action)

    def test_apply_returns_original_state_for_edit_only_action_without_edits(self) -> None:
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = DeviceAction(
            ports={"in": FakePath("p0")},
            kind=DeviceKind.DELAY,
            params={},
            edits=(),
            requires_kernel=False,
        )
        device = FakeDevice(
            kind=DeviceKind.DELAY,
            port_specs=(PortSpec(name="in", direction="in"),),
            action=action,
        )

        result = self.runtime.apply(
            device=device,
            state=state,
            ports={"in": FakePath("p0")},
        )

        self.assertIs(result, state)

    def test_apply_raises_for_edit_only_action_with_changed_out_kind(self) -> None:
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = DeviceAction(
            ports={"in": FakePath("p0")},
            kind=DeviceKind.DELAY,
            params={},
            edits=(),
            requires_kernel=False,
        )
        device = FakeDevice(
            kind=DeviceKind.DELAY,
            port_specs=(PortSpec(name="in", direction="in"),),
            action=action,
        )

        with self.assertRaises(ValueError):
            self.runtime.apply(
                device=device,
                state=state,
                ports={"in": FakePath("p0")},
                out_kind="density",
            )

    def test_apply_raises_type_error_when_edits_are_present_but_state_is_not_editable(self) -> None:
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = DeviceAction(
            ports={"in": FakePath("p0")},
            kind=DeviceKind.DELAY,
            params={},
            edits=(object(),),
            requires_kernel=False,
        )
        device = FakeDevice(
            kind=DeviceKind.DELAY,
            port_specs=(PortSpec(name="in", direction="in"),),
            action=action,
        )

        with self.assertRaises(TypeError):
            self.runtime.apply(
                device=device,
                state=state,
                ports={"in": FakePath("p0")},
            )

    def test_apply_applies_label_edits_when_state_is_editable(self) -> None:
        state = FakeEditableState(rep_kind="poly", state_kind="ket")
        edit = object()
        action = DeviceAction(
            ports={"in": FakePath("p0")},
            kind=DeviceKind.DELAY,
            params={},
            edits=(edit,),
            requires_kernel=False,
        )
        device = FakeDevice(
            kind=DeviceKind.DELAY,
            port_specs=(PortSpec(name="in", direction="in"),),
            action=action,
        )

        result = self.runtime.apply(
            device=device,
            state=state,
            ports={"in": FakePath("p0")},
        )

        self.assertIsInstance(result, FakeEditableState)
        self.assertEqual(result.applied_edits, [edit])


class TestDeviceRuntimeMeasurement(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = DeviceRuntime(
            device_registry=KernelRegistry(),
            measurement_registry=MeasurementKernelRegistry(),
        )

    def test_observe_plans_and_dispatches(self) -> None:
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = FakeMeasurementAction(intent="observe")
        result_object = object()
        kernel = RecordingKernel(result=result_object)

        device = FakeMeasurementDevice(
            kind=DeviceKind.NUMBER_DETECTOR,
            port_specs=(PortSpec(name="in", direction="in"),),
            observe_action=action,
            detect_action=action,
            postselect_action=action,
        )

        self.runtime.measurement_registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="observe",
            rep="poly",
            in_kind="ket",
            fn=kernel,
        )

        result = self.runtime.observe(
            device=device,
            state=state,
            ports={"in": FakePath("p0")},
            selection={"window": 1},
        )

        self.assertIs(result, result_object)
        self.assertEqual(len(device.observe_calls), 1)
        self.assertIs(device.observe_calls[0]["state"], state)
        self.assertEqual(device.observe_calls[0]["selection"], {"window": 1})
        self.assertEqual(len(kernel.calls), 1)

    def test_detect_plans_and_dispatches(self) -> None:
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = FakeMeasurementAction(intent="detect", destructive=True)
        result_object = object()
        kernel = RecordingKernel(result=result_object)

        device = FakeMeasurementDevice(
            kind=DeviceKind.NUMBER_DETECTOR,
            port_specs=(PortSpec(name="in", direction="in"),),
            observe_action=action,
            detect_action=action,
            postselect_action=action,
        )

        self.runtime.measurement_registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="detect",
            rep="poly",
            in_kind="ket",
            fn=kernel,
        )

        result = self.runtime.detect(
            device=device,
            state=state,
            ports={"in": FakePath("p0")},
        )

        self.assertIs(result, result_object)
        self.assertEqual(len(device.detect_calls), 1)
        self.assertEqual(len(kernel.calls), 1)

    def test_postselect_plans_and_dispatches(self) -> None:
        state = FakeState(rep_kind="poly", state_kind="ket")
        action = FakeMeasurementAction(intent="postselect", outcome="keep")
        result_object = object()
        kernel = RecordingKernel(result=result_object)

        device = FakeMeasurementDevice(
            kind=DeviceKind.NUMBER_DETECTOR,
            port_specs=(PortSpec(name="in", direction="in"),),
            observe_action=action,
            detect_action=action,
            postselect_action=action,
        )

        self.runtime.measurement_registry.register(
            device_kind=DeviceKind.NUMBER_DETECTOR,
            intent="postselect",
            rep="poly",
            in_kind="ket",
            fn=kernel,
        )

        result = self.runtime.postselect(
            device=device,
            state=state,
            outcome="keep",
            ports={"in": FakePath("p0")},
        )

        self.assertIs(result, result_object)
        self.assertEqual(len(device.postselect_calls), 1)
        self.assertEqual(device.postselect_calls[0]["outcome"], "keep")
        self.assertEqual(len(kernel.calls), 1)
