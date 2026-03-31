from __future__ import annotations

import unittest

from symop.devices.measurement.action import (
    DetectAction,
    ObserveAction,
    PostselectAction,
)
from symop.devices.measurement.outcomes import NumberOutcome
from symop.devices.measurement.result import (
    DetectionResult,
    ObservationResult,
    PostselectionResult,
)
from symop.devices.measurement.specs import ProjectiveNumberMeasurementSpec
from symop.devices.measurement.target import MeasurementSelection, MeasurementTarget

from tests.devices.support.fakes import FakePath, FakeState


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


class TestMeasurementResult(unittest.TestCase):
    def test_observation_result_stores_probabilities_and_expectation(self) -> None:
        action = ObserveAction(measurement_spec=make_spec())
        result = ObservationResult(
            action=action,
            probabilities={
                NumberOutcome(0): 0.4,
                NumberOutcome(1): 0.6,
            },
            expectation=0.6,
            metadata={"backend": "fake"},
        )

        self.assertIs(result.action, action)
        self.assertEqual(result.expectation, 0.6)
        self.assertEqual(result.metadata["backend"], "fake")

    def test_detection_result_stores_record_outcome_probability_and_state(self) -> None:
        action = DetectAction(measurement_spec=make_spec())
        state = FakeState()
        outcome = NumberOutcome(2)

        result = DetectionResult(
            action=action,
            record={"raw": 2},
            outcome=outcome,
            probability=0.25,
            state=state,
        )

        self.assertEqual(result.record, {"raw": 2})
        self.assertIs(result.outcome, outcome)
        self.assertEqual(result.probability, 0.25)
        self.assertIs(result.state, state)

    def test_postselection_result_stores_selected_branch(self) -> None:
        outcome = NumberOutcome(1)
        action = PostselectAction(
            measurement_spec=make_spec(),
            selected_outcome=outcome,
        )
        state = FakeState()

        result = PostselectionResult(
            action=action,
            outcome=outcome,
            probability=0.75,
            state=state,
        )

        self.assertIs(result.outcome, outcome)
        self.assertEqual(result.probability, 0.75)
        self.assertIs(result.state, state)
