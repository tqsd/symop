from __future__ import annotations

import unittest

from symop.devices.measurement.outcomes import NumberOutcome, ThresholdOutcome
from symop.devices.measurement.specs import InstrumentSpec, POVMSpec
from symop.devices.measurement.target import MeasurementSelection, MeasurementTarget

from tests.devices.support.fakes import FakePath


def make_target() -> MeasurementTarget:
    return MeasurementTarget(
        selections=(
            MeasurementSelection(
                port_name="in",
                paths=(FakePath("p0"),),
            ),
        )
    )


class TestMeasurementSpecs(unittest.TestCase):
    def test_povm_spec_requires_non_empty_unique_outcomes_and_matching_effects(self) -> None:
        outcome0 = NumberOutcome(0)
        outcome1 = NumberOutcome(1)
        target = make_target()

        spec = POVMSpec(
            target=target,
            outcomes=(outcome0, outcome1),
            effects={
                outcome0: object(),
                outcome1: object(),
            },
        )

        self.assertEqual(spec.outcomes, (outcome0, outcome1))
        self.assertEqual(set(spec.effects.keys()), {outcome0, outcome1})

        with self.assertRaises(ValueError):
            POVMSpec(target=target, outcomes=(), effects={})

        with self.assertRaises(ValueError):
            POVMSpec(
                target=target,
                outcomes=(outcome0, outcome0),
                effects={outcome0: object()},
            )

        with self.assertRaises(ValueError):
            POVMSpec(
                target=target,
                outcomes=(outcome0, outcome1),
                effects={outcome0: object()},
            )

    def test_instrument_spec_requires_operations_for_all_outcomes(self) -> None:
        outcome0 = ThresholdOutcome(False)
        outcome1 = ThresholdOutcome(True)
        target = make_target()

        spec = InstrumentSpec(
            target=target,
            outcomes=(outcome0, outcome1),
            effects={
                outcome0: object(),
                outcome1: object(),
            },
            operations={
                outcome0: (object(),),
                outcome1: (object(), object()),
            },
        )

        self.assertEqual(set(spec.operations.keys()), {outcome0, outcome1})

        with self.assertRaises(ValueError):
            InstrumentSpec(
                target=target,
                outcomes=(outcome0, outcome1),
                effects={
                    outcome0: object(),
                    outcome1: object(),
                },
                operations={
                    outcome0: (object(),),
                },
            )

        with self.assertRaises(ValueError):
            InstrumentSpec(
                target=target,
                outcomes=(outcome0, outcome1),
                effects={
                    outcome0: object(),
                    outcome1: object(),
                },
                operations={
                    outcome0: (object(),),
                    outcome1: (),
                },
            )
