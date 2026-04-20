from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from symop.devices.measurement.outcomes import JointOutcome, NumberOutcome
from symop.polynomial.kernels.measurements.number.observe import (
    observe_number_detector_poly_density,
    observe_number_detector_poly_ket,
)


class TestObserveNumberMeasurementPolyKet(unittest.TestCase):
    @patch("symop.polynomial.kernels.measurements.number.observe.resolve_number_stats_poly_ket")
    def test_observe_standard_ket(self, mock_resolve: MagicMock) -> None:
        action = MagicMock()
        action.resolution.grouping = "total"
        action.target = MagicMock()

        stats = MagicMock()
        stats.probabilities = {
            NumberOutcome(count=0): 0.25,
            NumberOutcome(count=1): 0.75,
        }
        stats.expectation = 0.75
        mock_resolve.return_value = stats

        result = observe_number_detector_poly_ket(
            state=MagicMock(),
            action=action,
            ctx=MagicMock(),
        )

        self.assertEqual(
            result.probabilities,
            {
                NumberOutcome(count=0): 0.25,
                NumberOutcome(count=1): 0.75,
            },
        )
        self.assertEqual(result.expectation, 0.75)

    @patch("symop.polynomial.kernels.measurements.number.observe.resolve_joint_number_stats_poly_ket")
    def test_observe_joint_ket(self, mock_resolve_joint: MagicMock) -> None:
        action = MagicMock()
        action.resolution.grouping = "joint_ports"
        action.target = MagicMock()

        outcome = JointOutcome(
            outcomes_by_port=(("a", NumberOutcome(count=1)),)
        )
        mock_resolve_joint.return_value = {outcome: 1.0}

        result = observe_number_detector_poly_ket(
            state=MagicMock(),
            action=action,
            ctx=MagicMock(),
        )

        self.assertEqual(result.probabilities, {outcome: 1.0})
        self.assertIsNone(result.expectation)


class TestObserveNumberMeasurementPolyDensity(unittest.TestCase):
    @patch("symop.polynomial.kernels.measurements.number.observe.resolve_number_stats_poly_density")
    def test_observe_standard_density(self, mock_resolve: MagicMock) -> None:
        action = MagicMock()
        action.resolution.grouping = "total"
        action.target = MagicMock()

        stats = MagicMock()
        stats.probabilities = {
            NumberOutcome(count=0): 0.4,
            NumberOutcome(count=2): 0.6,
        }
        stats.expectation = 1.2
        mock_resolve.return_value = stats

        result = observe_number_detector_poly_density(
            state=MagicMock(),
            action=action,
            ctx=MagicMock(),
        )

        self.assertEqual(
            result.probabilities,
            {
                NumberOutcome(count=0): 0.4,
                NumberOutcome(count=2): 0.6,
            },
        )
        self.assertEqual(result.expectation, 1.2)

    @patch("symop.polynomial.kernels.measurements.number.observe.resolve_joint_number_stats_poly_density")
    def test_observe_joint_density(self, mock_resolve_joint: MagicMock) -> None:
        action = MagicMock()
        action.resolution.grouping = "joint_ports"
        action.target = MagicMock()

        outcome = JointOutcome(
            outcomes_by_port=(("a", NumberOutcome(count=0)),)
        )
        mock_resolve_joint.return_value = {outcome: 1.0}

        result = observe_number_detector_poly_density(
            state=MagicMock(),
            action=action,
            ctx=MagicMock(),
        )

        self.assertEqual(result.probabilities, {outcome: 1.0})
        self.assertIsNone(result.expectation)
