from __future__ import annotations

import unittest

from symop.devices.measurement.action import (
    DetectAction,
    MeasurementAction,
    ObserveAction,
    PostselectAction,
)
from symop.devices.measurement.outcomes import NumberOutcome
from symop.devices.measurement.resolution import MeasurementResolution
from symop.devices.measurement.specs import ProjectiveNumberMeasurementSpec
from symop.devices.measurement.target import MeasurementSelection, MeasurementTarget

from tests.devices.support.fakes import FakePath


def make_spec() -> ProjectiveNumberMeasurementSpec:
    target = MeasurementTarget(
        selections=(
            MeasurementSelection(
                port_name="in",
                paths=(FakePath("p0"),),
            ),
        )
    )
    return ProjectiveNumberMeasurementSpec(
        target=target,
        resolution=MeasurementResolution(),
    )


class TestMeasurementAction(unittest.TestCase):
    def test_base_action_exposes_target_and_resolution(self) -> None:
        spec = make_spec()
        action = MeasurementAction(measurement_spec=spec)

        self.assertIs(action.target, spec.target)
        self.assertIs(action.resolution, spec.resolution)
        self.assertIsNone(action.outcome)

    def test_base_action_intent_raises(self) -> None:
        action = MeasurementAction(measurement_spec=make_spec())

        with self.assertRaises(NotImplementedError):
            _ = action.intent

    def test_observe_action_has_observe_intent_and_is_non_destructive(self) -> None:
        action = ObserveAction(measurement_spec=make_spec())

        self.assertEqual(action.intent, "observe")
        self.assertFalse(action.destructive)
        self.assertIsNone(action.outcome)

    def test_detect_action_has_detect_intent_and_is_destructive_by_default(self) -> None:
        action = DetectAction(measurement_spec=make_spec())

        self.assertEqual(action.intent, "detect")
        self.assertTrue(action.destructive)
        self.assertIsNone(action.outcome)

    def test_postselect_action_has_selected_outcome(self) -> None:
        outcome = NumberOutcome(2)
        action = PostselectAction(
            measurement_spec=make_spec(),
            selected_outcome=outcome,
        )

        self.assertEqual(action.intent, "postselect")
        self.assertTrue(action.destructive)
        self.assertIs(action.outcome, outcome)
