from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from symop.devices.measurement.outcomes import JointOutcome, NumberOutcome
from symop.polynomial.kernels.measurements.number.detect import (
    detect_number_detector_poly_density,
)


class TestDetectNumberMeasurementPoly(unittest.TestCase):
    @patch("symop.polynomial.kernels.measurements.number.detect.project_onto_number_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.detect.resolve_number_stats_poly_density")
    def test_detect_standard_number_non_destructive(
        self,
        mock_resolve: MagicMock,
        mock_project: MagicMock,
    ) -> None:
        action = MagicMock()
        action.resolution.grouping = "total"
        action.target = MagicMock()
        action.destructive = False

        outcome0 = NumberOutcome(count=0)
        outcome1 = NumberOutcome(count=1)

        stats = MagicMock()
        stats.probabilities = {outcome0: 0.25, outcome1: 0.75}
        mock_resolve.return_value = stats

        projected = MagicMock()
        normalized = MagicMock()
        projected.trace.return_value = 0.75 + 0.0j
        projected.normalize_trace.return_value = normalized
        mock_project.return_value = projected

        rng = MagicMock()
        rng.choices.return_value = [outcome1]

        SupportsRngFake = type("SupportsRngFake", (), {})
        ctx = SupportsRngFake()
        ctx.rng = rng

        with patch(
            "symop.polynomial.kernels.measurements.number.detect.SupportsRng",
            new=SupportsRngFake,
        ):
            result = detect_number_detector_poly_density(
                state=MagicMock(),
                action=action,
                ctx=ctx,
            )

        self.assertEqual(result.outcome, outcome1)
        self.assertEqual(result.record, outcome1)
        self.assertEqual(result.probability, 0.75)
        self.assertIs(result.state, normalized)
        rng.choices.assert_called_once()
        mock_project.assert_called_once()
    @patch("symop.polynomial.kernels.measurements.number.detect.discard_measured_modes_number_density")
    @patch("symop.polynomial.kernels.measurements.number.detect.project_onto_number_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.detect.resolve_number_stats_poly_density")
    def test_detect_standard_number_destructive(
        self,
        mock_resolve: MagicMock,
        mock_project: MagicMock,
        mock_discard: MagicMock,
    ) -> None:
        action = MagicMock()
        action.resolution.grouping = "total"
        action.target = MagicMock()
        action.destructive = True

        outcome = NumberOutcome(count=2)
        stats = MagicMock()
        stats.probabilities = {outcome: 1.0}
        mock_resolve.return_value = stats

        projected = MagicMock()
        normalized = MagicMock()
        reduced = MagicMock()
        projected.trace.return_value = 1.0 + 0.0j
        projected.normalize_trace.return_value = normalized
        mock_project.return_value = projected
        mock_discard.return_value = reduced

        rng = MagicMock()
        rng.choices.return_value = [outcome]
        ctx = MagicMock()
        ctx.rng = rng

        with patch(
            "symop.polynomial.kernels.measurements.number.detect.SupportsRng",
            new=type("SupportsRng", (), {}),
        ):
            result = detect_number_detector_poly_density(
                state=MagicMock(),
                action=action,
                ctx=ctx,
            )

        self.assertEqual(result.outcome, outcome)
        self.assertIs(result.state, reduced)
        mock_discard.assert_called_once_with(normalized, action.target)

    @patch("symop.polynomial.kernels.measurements.number.detect.project_onto_joint_number_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.detect.resolve_joint_number_stats_poly_density")
    def test_detect_joint_number(
        self,
        mock_resolve_joint: MagicMock,
        mock_project_joint: MagicMock,
    ) -> None:
        action = MagicMock()
        action.resolution.grouping = "joint_ports"
        action.target = MagicMock()
        action.destructive = False

        outcome = JointOutcome(
            outcomes_by_port=(("a", NumberOutcome(count=1)),)
        )
        mock_resolve_joint.return_value = {outcome: 1.0}

        projected = MagicMock()
        normalized = MagicMock()
        projected.trace.return_value = 1.0 + 0.0j
        projected.normalize_trace.return_value = normalized
        mock_project_joint.return_value = projected

        rng = MagicMock()
        rng.choices.return_value = [outcome]
        ctx = MagicMock()
        ctx.rng = rng

        with patch(
            "symop.polynomial.kernels.measurements.number.detect.SupportsRng",
            new=type("SupportsRng", (), {}),
        ):
            result = detect_number_detector_poly_density(
                state=MagicMock(),
                action=action,
                ctx=ctx,
            )

        self.assertEqual(result.outcome, outcome)
        self.assertEqual(result.record, outcome)
        self.assertEqual(result.probability, 1.0)
        self.assertIs(result.state, normalized)
        mock_project_joint.assert_called_once()

    @patch("symop.polynomial.kernels.measurements.number.detect.project_onto_number_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.detect.resolve_number_stats_poly_density")
    def test_detect_returns_none_state_for_non_positive_probability(
        self,
        mock_resolve: MagicMock,
        mock_project: MagicMock,
    ) -> None:
        action = MagicMock()
        action.resolution.grouping = "total"
        action.target = MagicMock()
        action.destructive = False

        outcome = NumberOutcome(count=1)
        stats = MagicMock()
        stats.probabilities = {outcome: 1.0}
        mock_resolve.return_value = stats

        projected = MagicMock()
        projected.trace.return_value = 0.0 + 0.0j
        mock_project.return_value = projected

        rng = MagicMock()
        rng.choices.return_value = [outcome]
        ctx = MagicMock()
        ctx.rng = rng

        with patch(
            "symop.polynomial.kernels.measurements.number.detect.SupportsRng",
            new=type("SupportsRng", (), {}),
        ):
            result = detect_number_detector_poly_density(
                state=MagicMock(),
                action=action,
                ctx=ctx,
            )

        self.assertEqual(result.probability, 0.0)
        self.assertIsNone(result.state)

    @patch("symop.polynomial.kernels.measurements.number.detect.resolve_number_stats_poly_density")
    def test_detect_raises_on_empty_standard_distribution(self, mock_resolve: MagicMock) -> None:
        action = MagicMock()
        action.resolution.grouping = "total"
        action.target = MagicMock()
        action.destructive = False

        stats = MagicMock()
        stats.probabilities = {}
        mock_resolve.return_value = stats

        with self.assertRaises(ValueError):
            detect_number_detector_poly_density(
                state=MagicMock(),
                action=action,
                ctx=MagicMock(),
            )

    @patch("symop.polynomial.kernels.measurements.number.detect.resolve_joint_number_stats_poly_density")
    def test_detect_raises_on_empty_joint_distribution(self, mock_resolve_joint: MagicMock) -> None:
        action = MagicMock()
        action.resolution.grouping = "joint_ports"
        action.target = MagicMock()
        action.destructive = False

        mock_resolve_joint.return_value = {}

        with self.assertRaises(ValueError):
            detect_number_detector_poly_density(
                state=MagicMock(),
                action=action,
                ctx=MagicMock(),
            )
