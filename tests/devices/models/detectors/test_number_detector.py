from __future__ import annotations

import unittest

from symop.devices.measurement.action import (
    DetectAction,
    ObserveAction,
    PostselectAction,
)
from symop.devices.measurement.outcomes import NumberOutcome
from symop.devices.models.detectors.number_detector import NumberDetector

from tests.devices.support.fakes import FakePath, FakeState


class TestNumberDetector(unittest.TestCase):
    def test_port_specs_expose_single_input(self) -> None:
        detector = NumberDetector()

        self.assertEqual(len(detector.port_specs), 1)
        self.assertEqual(detector.port_specs[0].name, "in")
        self.assertEqual(detector.port_specs[0].direction, "in")

    def test_plan_observe_builds_number_measurement_action(self) -> None:
        detector = NumberDetector()
        action = detector.plan_observe(
            state=FakeState(),
            ports={"in": FakePath("p0")},
        )

        self.assertIsInstance(action, ObserveAction)
        self.assertFalse(action.destructive)
        self.assertEqual(action.intent, "observe")
        self.assertEqual(action.target.selections[0].port_name, "in")
        self.assertEqual(action.target.paths, (FakePath("p0"),))
        self.assertEqual(action.resolution.axes, ("path",))
        self.assertEqual(action.resolution.grouping, "total")
        self.assertEqual(action.resolution.readout, "number")

    def test_plan_detect_uses_detector_destructive_flag(self) -> None:
        detector = NumberDetector(destructive=False)

        action = detector.plan_detect(
            state=FakeState(),
            ports={"in": FakePath("p0")},
        )

        self.assertIsInstance(action, DetectAction)
        self.assertFalse(action.destructive)
        self.assertEqual(action.intent, "detect")

    def test_plan_postselect_includes_selected_outcome(self) -> None:
        detector = NumberDetector()
        outcome = NumberOutcome(1)

        action = detector.plan_postselect(
            state=FakeState(),
            outcome=outcome,
            ports={"in": FakePath("p0")},
        )

        self.assertIsInstance(action, PostselectAction)
        self.assertIs(action.outcome, outcome)
        self.assertTrue(action.destructive)
        self.assertEqual(action.intent, "postselect")

    def test_missing_input_port_raises_key_error(self) -> None:
        detector = NumberDetector()

        with self.assertRaises(KeyError):
            detector.plan_observe(
                state=FakeState(),
                ports={},
            )
