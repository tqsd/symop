from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from symop.devices.measurement.outcomes import JointOutcome, NumberOutcome
from symop.polynomial.kernels.measurements.number.postselect import (
    postselect_number_detector_poly_density,
)


class TestPostselectNumberMeasurementPoly(unittest.TestCase):
    @patch("symop.polynomial.kernels.measurements.number.postselect.project_onto_number_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.postselect.require_number_outcome")
    def test_postselect_standard_non_destructive(
        self,
        mock_require: MagicMock,
        mock_project: MagicMock,
    ) -> None:
        action = MagicMock()
        action.resolution.grouping = "total"
        action.target = MagicMock()
        action.destructive = False
        action.outcome = NumberOutcome(count=1)

        outcome = NumberOutcome(count=1)
        mock_require.return_value = outcome

        projected = MagicMock()
        normalized = MagicMock()
        projected.trace.return_value = 0.6 + 0.0j
        projected.normalize_trace.return_value = normalized
        mock_project.return_value = projected

        result = postselect_number_detector_poly_density(
            state=MagicMock(),
            action=action,
            ctx=MagicMock(),
        )

        self.assertEqual(result.outcome, outcome)
        self.assertEqual(result.probability, 0.6)
        self.assertIs(result.state, normalized)
        mock_require.assert_called_once_with(action.outcome)
        mock_project.assert_called_once()

    @patch("symop.polynomial.kernels.measurements.number.postselect.discard_measured_modes_number_density")
    @patch("symop.polynomial.kernels.measurements.number.postselect.project_onto_number_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.postselect.require_number_outcome")
    def test_postselect_standard_destructive(
        self,
        mock_require: MagicMock,
        mock_project: MagicMock,
        mock_discard: MagicMock,
    ) -> None:
        action = MagicMock()
        action.resolution.grouping = "total"
        action.target = MagicMock()
        action.destructive = True
        action.outcome = NumberOutcome(count=2)

        outcome = NumberOutcome(count=2)
        mock_require.return_value = outcome

        projected = MagicMock()
        normalized = MagicMock()
        reduced = MagicMock()
        projected.trace.return_value = 1.0 + 0.0j
        projected.normalize_trace.return_value = normalized
        mock_project.return_value = projected
        mock_discard.return_value = reduced

        result = postselect_number_detector_poly_density(
            state=MagicMock(),
            action=action,
            ctx=MagicMock(),
        )

        self.assertEqual(result.outcome, outcome)
        self.assertEqual(result.probability, 1.0)
        self.assertIs(result.state, reduced)
        mock_discard.assert_called_once_with(normalized, action.target)

    @patch("symop.polynomial.kernels.measurements.number.postselect.project_onto_joint_number_poly_density")
    def test_postselect_joint_non_destructive(self, mock_project_joint: MagicMock) -> None:
        action = MagicMock()
        action.resolution.grouping = "joint_ports"
        action.target = MagicMock()
        action.destructive = False
        action.outcome = JointOutcome(
            outcomes_by_port=(("a", NumberOutcome(count=1)),)
        )

        projected = MagicMock()
        normalized = MagicMock()
        projected.trace.return_value = 0.4 + 0.0j
        projected.normalize_trace.return_value = normalized
        mock_project_joint.return_value = projected

        result = postselect_number_detector_poly_density(
            state=MagicMock(),
            action=action,
            ctx=MagicMock(),
        )

        self.assertEqual(result.outcome, action.outcome)
        self.assertEqual(result.probability, 0.4)
        self.assertIs(result.state, normalized)

    def test_postselect_joint_rejects_non_joint_outcome(self) -> None:
        action = MagicMock()
        action.resolution.grouping = "joint_ports"
        action.target = MagicMock()
        action.destructive = False
        action.outcome = NumberOutcome(count=1)

        with self.assertRaises(TypeError):
            postselect_number_detector_poly_density(
                state=MagicMock(),
                action=action,
                ctx=MagicMock(),
            )

    @patch("symop.polynomial.kernels.measurements.number.postselect.project_onto_number_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.postselect.require_number_outcome")
    def test_postselect_returns_none_state_for_zero_probability(
        self,
        mock_require: MagicMock,
        mock_project: MagicMock,
    ) -> None:
        action = MagicMock()
        action.resolution.grouping = "total"
        action.target = MagicMock()
        action.destructive = False
        action.outcome = NumberOutcome(count=0)

        outcome = NumberOutcome(count=0)
        mock_require.return_value = outcome

        projected = MagicMock()
        projected.trace.return_value = 0.0 + 0.0j
        mock_project.return_value = projected

        result = postselect_number_detector_poly_density(
            state=MagicMock(),
            action=action,
            ctx=MagicMock(),
        )

        self.assertEqual(result.outcome, outcome)
        self.assertEqual(result.probability, 0.0)
        self.assertIsNone(result.state)
