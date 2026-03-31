from __future__ import annotations

import unittest

from symop.devices.measurement.outcomes import (
    JointOutcome,
    NumberOutcome,
    ParityOutcome,
    ThresholdOutcome,
)


class TestMeasurementOutcomes(unittest.TestCase):
    def test_number_outcome_validates_and_exposes_key_and_label(self) -> None:
        outcome = NumberOutcome(3)

        self.assertEqual(outcome.key, ("number", 3))
        self.assertEqual(outcome.label, "3")

        with self.assertRaises(ValueError):
            NumberOutcome(-1)

    def test_threshold_outcome_key_and_label(self) -> None:
        click = ThresholdOutcome(True)
        no_click = ThresholdOutcome(False)

        self.assertEqual(click.key, ("threshold", True))
        self.assertEqual(click.label, "click")
        self.assertEqual(no_click.key, ("threshold", False))
        self.assertEqual(no_click.label, "no-click")

    def test_parity_outcome_validates_and_exposes_key_and_label(self) -> None:
        even = ParityOutcome("even")
        odd = ParityOutcome("odd")

        self.assertEqual(even.key, ("parity", "even"))
        self.assertEqual(even.label, "even")
        self.assertEqual(odd.key, ("parity", "odd"))
        self.assertEqual(odd.label, "odd")

        with self.assertRaises(ValueError):
            ParityOutcome("bad")

    def test_joint_outcome_builds_joint_key_and_label(self) -> None:
        outcome = JointOutcome(
            outcomes_by_port=(
                ("in0", NumberOutcome(1)),
                ("in1", ThresholdOutcome(True)),
            )
        )

        self.assertEqual(
            outcome.key,
            (
                "joint",
                (
                    ("in0", ("number", 1)),
                    ("in1", ("threshold", True)),
                ),
            ),
        )
        self.assertEqual(outcome.label, "in0=1, in1=click")
