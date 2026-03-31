from __future__ import annotations

import unittest

from symop.devices.measurement.outcomes import NumberOutcome
from symop.devices.measurement.resolved import (
    ResolvedOutcomeMeasurement,
    ResolvedProjectiveNumberMeasurement,
)
from symop.devices.measurement.specs import ProjectiveNumberMeasurementSpec
from symop.devices.measurement.target import MeasurementSelection, MeasurementTarget

from tests.devices.support.fakes import FakePath


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


class TestResolvedMeasurement(unittest.TestCase):
    def test_resolved_outcome_measurement_exposes_outcomes_in_mapping_order(self) -> None:
        spec = make_spec()
        outcome0 = NumberOutcome(0)
        outcome1 = NumberOutcome(1)

        resolved = ResolvedOutcomeMeasurement(
            spec=spec,
            probabilities={
                outcome0: 0.25,
                outcome1: 0.75,
            },
        )

        self.assertEqual(resolved.outcomes, (outcome0, outcome1))

    def test_resolved_projective_number_measurement_computes_expectation(self) -> None:
        spec = make_spec()
        resolved = ResolvedProjectiveNumberMeasurement(
            spec=spec,
            probabilities={
                NumberOutcome(0): 0.2,
                NumberOutcome(1): 0.3,
                NumberOutcome(2): 0.5,
            },
        )

        self.assertAlmostEqual(resolved.expectation, 1.3)
