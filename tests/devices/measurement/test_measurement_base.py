from __future__ import annotations

import unittest
from unittest.mock import patch

from symop.devices.measurement.action import (
    DetectAction,
    ObserveAction,
    PostselectAction,
)
from symop.devices.measurement.base import MeasurementDeviceBase
from symop.devices.measurement.outcomes import NumberOutcome
from symop.devices.measurement.specs import ProjectiveNumberMeasurementSpec
from symop.devices.measurement.target import MeasurementSelection, MeasurementTarget
from symop.devices.ports import PortSpec
from symop.devices.types.device_kind import DeviceKind

from tests.devices.support.fakes import (
    FakeMeasurementRuntime,
    FakePath,
    FakeState,
)


def make_spec() -> ProjectiveNumberMeasurementSpec:
    return ProjectiveNumberMeasurementSpec(
        target=MeasurementTarget(
            selections=(
                MeasurementSelection(
                    port_name="in",
                    paths=(FakePath("p0"),),
                ),
            )
        )
    )


class DummyMeasurementDevice(MeasurementDeviceBase):
    def __init__(self) -> None:
        self.observe_action = ObserveAction(measurement_spec=make_spec())
        self.detect_action = DetectAction(measurement_spec=make_spec())
        self.postselect_action = PostselectAction(
            measurement_spec=make_spec(),
            selected_outcome=NumberOutcome(1),
        )

    @property
    def kind(self) -> DeviceKind:
        return DeviceKind.NUMBER_DETECTOR

    @property
    def port_specs(self) -> tuple[PortSpec, ...]:
        return (PortSpec("in", "in"),)

    def plan_observe(self, *, state, ports, selection=None, ctx=None):
        return self.observe_action

    def plan_detect(self, *, state, ports, selection=None, ctx=None):
        return self.detect_action

    def plan_postselect(self, *, state, outcome, ports, selection=None, ctx=None):
        return self.postselect_action


class TestMeasurementDeviceBase(unittest.TestCase):
    def test_observe_uses_explicit_runtime(self) -> None:
        device = DummyMeasurementDevice()
        state = FakeState()
        ports = {"in": FakePath("p0")}
        runtime = FakeMeasurementRuntime(observe_result="observed")

        result = device.observe(
            state,
            ports=ports,
            selection="sel",
            runtime=runtime,
            ctx="ctx",
        )

        self.assertEqual(result, "observed")
        self.assertEqual(len(runtime.observe_calls), 1)
        call = runtime.observe_calls[0]
        self.assertIs(call["device"], device)
        self.assertIs(call["state"], state)
        self.assertEqual(call["ports"], ports)
        self.assertEqual(call["selection"], "sel")
        self.assertEqual(call["ctx"], "ctx")

    def test_detect_uses_default_runtime_when_not_provided(self) -> None:
        device = DummyMeasurementDevice()
        state = FakeState()
        ports = {"in": FakePath("p0")}
        runtime = FakeMeasurementRuntime(detect_result="detected")

        with patch("symop.devices.measurement.base.get_default_runtime", return_value=runtime):
            result = device.detect(
                state,
                ports=ports,
                selection="sel",
                ctx="ctx",
            )

        self.assertEqual(result, "detected")
        self.assertEqual(len(runtime.detect_calls), 1)

    def test_postselect_forwards_outcome_to_runtime(self) -> None:
        device = DummyMeasurementDevice()
        state = FakeState()
        ports = {"in": FakePath("p0")}
        outcome = NumberOutcome(2)
        runtime = FakeMeasurementRuntime(postselect_result="postselected")

        result = device.postselect(
            state,
            outcome=outcome,
            ports=ports,
            selection="sel",
            runtime=runtime,
            ctx="ctx",
        )

        self.assertEqual(result, "postselected")
        self.assertEqual(len(runtime.postselect_calls), 1)
        call = runtime.postselect_calls[0]
        self.assertIs(call["outcome"], outcome)
