from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from symop.devices.measurement.outcomes import JointOutcome, NumberOutcome
from symop.polynomial.kernels.measurements.number.common import (
    NumberMeasurementError,
    NumberMeasurementStats,
    count_selected_creators_by_port,
    count_selected_creators_in_monomial,
    count_selected_quanta_ket_term,
    count_selected_quanta_left,
    count_selected_quanta_right,
    counts_from_joint_outcome,
    discard_measured_modes_number_density,
    discard_measured_modes_number_ket,
    iter_joint_number_support_poly_density,
    iter_joint_number_support_poly_ket,
    iter_number_support_poly_density,
    iter_number_support_poly_ket,
    joint_number_counts_for_ket_term,
    joint_number_counts_left,
    joint_number_counts_right,
    joint_outcome_from_counts,
    project_onto_joint_number_poly_density,
    project_onto_joint_number_poly_ket,
    project_onto_number_poly_density,
    project_onto_number_poly_ket,
    require_number_outcome,
    resolve_joint_number_stats_poly_density,
    resolve_joint_number_stats_poly_ket,
    resolve_number_stats_poly_density,
    resolve_number_stats_poly_ket,
    selected_mode_signatures,
    selected_mode_signatures_by_port,
    selected_modes,
)


class FakeMode:
    def __init__(self, signature: str) -> None:
        self.signature = signature


class FakeCreator:
    def __init__(self, mode: FakeMode) -> None:
        self.mode = mode


class FakeMonomial:
    def __init__(self, creators, has_annihilators: bool = False) -> None:
        self.creators = tuple(creators)
        self.has_annihilators = has_annihilators


class FakeKetTerm:
    def __init__(self, monomial) -> None:
        self.monomial = monomial


class FakeDensityTerm:
    def __init__(self, left, right) -> None:
        self.left = left
        self.right = right


class TestNumberMeasurementStats(unittest.TestCase):
    def test_expectation(self) -> None:
        stats = NumberMeasurementStats(
            probabilities={
                NumberOutcome(count=0): 0.25,
                NumberOutcome(count=2): 0.75,
            }
        )
        self.assertAlmostEqual(stats.expectation, 1.5)


class TestTargetHelpers(unittest.TestCase):
    def test_selected_modes_delegates(self) -> None:
        state = MagicMock()
        target = MagicMock()
        mode0 = MagicMock()
        mode1 = MagicMock()
        state.resolve_target_modes.return_value = (mode0, mode1)

        out = selected_modes(state, target)

        self.assertEqual(out, (mode0, mode1))
        state.resolve_target_modes.assert_called_once_with(target)

    def test_selected_mode_signatures(self) -> None:
        state = MagicMock()
        target = MagicMock()
        mode0 = MagicMock()
        mode0.signature = "m0"
        mode1 = MagicMock()
        mode1.signature = "m1"
        state.resolve_target_modes.return_value = (mode0, mode1)

        out = selected_mode_signatures(state, target)

        self.assertEqual(out, frozenset({"m0", "m1"}))

    def test_selected_mode_signatures_by_port(self) -> None:
        mode0 = MagicMock()
        mode0.signature = "m0"
        mode1 = MagicMock()
        mode1.signature = "m1"
        mode2 = MagicMock()
        mode2.signature = "m2"

        state = MagicMock()
        state.modes_on_path.side_effect = [
            [mode0, mode1],
            [mode2],
        ]
        state.mode_by_signature = {
            "m1": mode1,
            "m2": mode2,
        }

        target = SimpleNamespace(
            selections=[
                SimpleNamespace(
                    port_name="p0",
                    paths=("path0",),
                    mode_sigs=("m1",),
                ),
                SimpleNamespace(
                    port_name="p1",
                    paths=("path1",),
                    mode_sigs=("m2",),
                ),
            ]
        )

        out = selected_mode_signatures_by_port(state, target)

        self.assertEqual(
            out,
            {
                "p0": frozenset({"m0", "m1"}),
                "p1": frozenset({"m2"}),
            },
        )


class TestOutcomeHelpers(unittest.TestCase):
    def test_require_number_outcome_accepts_number_outcome(self) -> None:
        outcome = NumberOutcome(count=2)
        self.assertIs(require_number_outcome(outcome), outcome)

    def test_require_number_outcome_rejects_other_object(self) -> None:
        with self.assertRaises(TypeError):
            require_number_outcome(object())

    def test_joint_outcome_from_counts(self) -> None:
        outcome = joint_outcome_from_counts((("a", 2), ("b", 1)))

        self.assertEqual(
            outcome,
            JointOutcome(
                outcomes_by_port=(
                    ("a", NumberOutcome(count=2)),
                    ("b", NumberOutcome(count=1)),
                )
            ),
        )

    def test_counts_from_joint_outcome(self) -> None:
        outcome = JointOutcome(
            outcomes_by_port=(
                ("a", NumberOutcome(count=2)),
                ("b", NumberOutcome(count=1)),
            )
        )

        counts = counts_from_joint_outcome(outcome)

        self.assertEqual(counts, (("a", 2), ("b", 1)))

    def test_counts_from_joint_outcome_rejects_non_number_port_outcome(self) -> None:
        outcome = JointOutcome(
            outcomes_by_port=(
                ("a", object()),
            )
        )

        with self.assertRaises(TypeError):
            counts_from_joint_outcome(outcome)


class TestCountingHelpers(unittest.TestCase):
    def test_count_selected_creators_in_monomial(self) -> None:
        m0 = FakeMode("m0")
        m1 = FakeMode("m1")
        monomial = FakeMonomial(
            creators=(
                FakeCreator(m0),
                FakeCreator(m1),
                FakeCreator(m0),
            )
        )

        count = count_selected_creators_in_monomial(frozenset({"m0"}), monomial)

        self.assertEqual(count, 2)

    def test_count_selected_creators_by_port(self) -> None:
        m0 = FakeMode("m0")
        m1 = FakeMode("m1")
        m2 = FakeMode("m2")
        monomial = FakeMonomial(
            creators=(
                FakeCreator(m0),
                FakeCreator(m1),
                FakeCreator(m0),
                FakeCreator(m2),
            )
        )

        counts = count_selected_creators_by_port(
            monomial,
            {
                "p0": frozenset({"m0", "m2"}),
                "p1": frozenset({"m1"}),
            },
        )

        self.assertEqual(counts, (("p0", 3), ("p1", 1)))

    def test_count_selected_quanta_ket_term(self) -> None:
        m0 = FakeMode("m0")
        m1 = FakeMode("m1")
        term = FakeKetTerm(
            FakeMonomial(
                creators=(
                    FakeCreator(m0),
                    FakeCreator(m1),
                    FakeCreator(m0),
                )
            )
        )

        count = count_selected_quanta_ket_term(term, frozenset({"m0"}))

        self.assertEqual(count, 2)

    def test_count_selected_quanta_left_raises_when_annihilators_present(self) -> None:
        term = FakeDensityTerm(
            left=FakeMonomial((), has_annihilators=True),
            right=FakeMonomial(()),
        )

        with self.assertRaises(NumberMeasurementError):
            count_selected_quanta_left(term, frozenset({"m0"}))

    def test_count_selected_quanta_right_raises_when_annihilators_present(self) -> None:
        term = FakeDensityTerm(
            left=FakeMonomial(()),
            right=FakeMonomial((), has_annihilators=True),
        )

        with self.assertRaises(NumberMeasurementError):
            count_selected_quanta_right(term, frozenset({"m0"}))

    def test_joint_number_counts_for_ket_term(self) -> None:
        m0 = FakeMode("m0")
        m1 = FakeMode("m1")
        term = FakeKetTerm(
            FakeMonomial(
                creators=(
                    FakeCreator(m0),
                    FakeCreator(m1),
                    FakeCreator(m0),
                )
            )
        )

        counts = joint_number_counts_for_ket_term(
            term,
            {
                "p0": frozenset({"m0"}),
                "p1": frozenset({"m1"}),
            },
        )

        self.assertEqual(counts, (("p0", 2), ("p1", 1)))

    def test_joint_number_counts_left_raises_when_annihilators_present(self) -> None:
        term = FakeDensityTerm(
            left=FakeMonomial((), has_annihilators=True),
            right=FakeMonomial(()),
        )

        with self.assertRaises(NumberMeasurementError):
            joint_number_counts_left(term, {"p0": frozenset({"m0"})})

    def test_joint_number_counts_right_raises_when_annihilators_present(self) -> None:
        term = FakeDensityTerm(
            left=FakeMonomial(()),
            right=FakeMonomial((), has_annihilators=True),
        )

        with self.assertRaises(NumberMeasurementError):
            joint_number_counts_right(term, {"p0": frozenset({"m0"})})


class TestKetSupportAndProjection(unittest.TestCase):
    @patch("symop.polynomial.kernels.measurements.number.common.selected_mode_signatures")
    def test_iter_number_support_poly_ket(self, mock_selected_mode_signatures: MagicMock) -> None:
        mock_selected_mode_signatures.return_value = frozenset({"m0"})
        m0 = FakeMode("m0")
        m1 = FakeMode("m1")

        state = MagicMock()
        state.ket.terms = (
            FakeKetTerm(FakeMonomial(())),
            FakeKetTerm(FakeMonomial((FakeCreator(m0),))),
            FakeKetTerm(FakeMonomial((FakeCreator(m0), FakeCreator(m0)))),
            FakeKetTerm(FakeMonomial((FakeCreator(m1),))),
        )

        support = iter_number_support_poly_ket(state, MagicMock())

        self.assertEqual(
            support,
            (
                NumberOutcome(count=0),
                NumberOutcome(count=1),
                NumberOutcome(count=2),
            ),
        )

    @patch("symop.polynomial.kernels.measurements.number.common.selected_mode_signatures")
    @patch("symop.polynomial.kernels.measurements.number.common.KetPolyState.from_ketpoly")
    def test_project_onto_number_poly_ket(self, mock_from_ketpoly: MagicMock, mock_selected_mode_signatures: MagicMock) -> None:
        mock_selected_mode_signatures.return_value = frozenset({"m0"})

        term_keep = MagicMock()
        term_drop = MagicMock()

        ket_obj = MagicMock()
        ket_obj.terms = (term_keep, term_drop)

        ket_cls = MagicMock()
        ket_obj.__class__ = ket_cls

        combined = MagicMock()
        ket_cls.return_value.combine_like_terms.return_value = combined

        state = MagicMock()
        state.ket = ket_obj

        with patch(
            "symop.polynomial.kernels.measurements.number.common.count_selected_quanta_ket_term",
            side_effect=[1, 0],
        ):
            out = project_onto_number_poly_ket(
                state,
                MagicMock(),
                NumberOutcome(count=1),
            )

        self.assertIs(out, mock_from_ketpoly.return_value)
        ket_cls.assert_called_once_with((term_keep,))
        ket_cls.return_value.combine_like_terms.assert_called_once_with(eps=1e-12)
        mock_from_ketpoly.assert_called_once_with(combined)

    @patch("symop.polynomial.kernels.measurements.number.common.iter_number_support_poly_ket")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_number_poly_ket")
    def test_resolve_number_stats_poly_ket(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out0 = NumberOutcome(count=0)
        out1 = NumberOutcome(count=1)
        mock_support.return_value = (out0, out1)

        proj0 = MagicMock()
        proj0.norm2 = 0.25
        proj1 = MagicMock()
        proj1.norm2 = 0.75
        mock_project.side_effect = [proj0, proj1]

        stats = resolve_number_stats_poly_ket(MagicMock(), MagicMock())

        self.assertEqual(stats.probabilities, {out0: 0.25, out1: 0.75})
        self.assertAlmostEqual(stats.expectation, 0.75)

    @patch("symop.polynomial.kernels.measurements.number.common.iter_number_support_poly_ket")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_number_poly_ket")
    def test_resolve_number_stats_poly_ket_clips_small_negative_to_zero(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out0 = NumberOutcome(count=0)
        mock_support.return_value = (out0,)

        proj = MagicMock()
        proj.norm2 = -1e-13
        mock_project.return_value = proj

        stats = resolve_number_stats_poly_ket(MagicMock(), MagicMock(), eps=1e-12)

        self.assertEqual(stats.probabilities, {})

    @patch("symop.polynomial.kernels.measurements.number.common.iter_number_support_poly_ket")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_number_poly_ket")
    def test_resolve_number_stats_poly_ket_raises_on_negative_probability(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out1 = NumberOutcome(count=1)
        mock_support.return_value = (out1,)

        proj = MagicMock()
        proj.norm2 = -1e-3
        mock_project.return_value = proj

        with self.assertRaises(NumberMeasurementError):
            resolve_number_stats_poly_ket(MagicMock(), MagicMock(), eps=1e-12)

    def test_discard_measured_modes_number_ket_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            discard_measured_modes_number_ket(MagicMock(), MagicMock())


class TestKetJointSupportAndProjection(unittest.TestCase):
    @patch("symop.polynomial.kernels.measurements.number.common.selected_mode_signatures_by_port")
    def test_iter_joint_number_support_poly_ket(self, mock_selected_mode_signatures_by_port: MagicMock) -> None:
        mock_selected_mode_signatures_by_port.return_value = {
            "a": frozenset({"m0"}),
            "b": frozenset({"m1"}),
        }

        m0 = FakeMode("m0")
        m1 = FakeMode("m1")

        state = MagicMock()
        state.ket.terms = (
            FakeKetTerm(FakeMonomial((FakeCreator(m0),))),
            FakeKetTerm(FakeMonomial((FakeCreator(m0), FakeCreator(m1)))),
        )

        out = iter_joint_number_support_poly_ket(state, MagicMock())

        self.assertEqual(
            out,
            (
                JointOutcome(
                    outcomes_by_port=(
                        ("a", NumberOutcome(count=1)),
                        ("b", NumberOutcome(count=0)),
                    )
                ),
                JointOutcome(
                    outcomes_by_port=(
                        ("a", NumberOutcome(count=1)),
                        ("b", NumberOutcome(count=1)),
                    )
                ),
            ),
        )

    @patch("symop.polynomial.kernels.measurements.number.common.selected_mode_signatures_by_port")
    @patch("symop.polynomial.kernels.measurements.number.common.KetPolyState.from_ketpoly")
    def test_project_onto_joint_number_poly_ket(self, mock_from_ketpoly: MagicMock, mock_selected_mode_signatures_by_port: MagicMock) -> None:
        mock_selected_mode_signatures_by_port.return_value = {
            "a": frozenset({"m0"}),
        }

        term_keep = MagicMock()
        term_drop = MagicMock()

        ket_obj = MagicMock()
        ket_obj.terms = (term_keep, term_drop)

        ket_cls = MagicMock()
        ket_obj.__class__ = ket_cls

        combined = MagicMock()
        ket_cls.return_value.combine_like_terms.return_value = combined

        state = MagicMock()
        state.ket = ket_obj

        expected_outcome = JointOutcome(
            outcomes_by_port=(("a", NumberOutcome(count=1)),)
        )

        with patch(
            "symop.polynomial.kernels.measurements.number.common.joint_number_counts_for_ket_term",
            side_effect=[(("a", 1),), (("a", 0),)],
        ):
            out = project_onto_joint_number_poly_ket(
                state,
                MagicMock(),
                expected_outcome,
            )

        self.assertIs(out, mock_from_ketpoly.return_value)
        ket_cls.assert_called_once_with((term_keep,))
        ket_cls.return_value.combine_like_terms.assert_called_once_with(eps=1e-12)
        mock_from_ketpoly.assert_called_once_with(combined)

    @patch("symop.polynomial.kernels.measurements.number.common.iter_joint_number_support_poly_ket")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_joint_number_poly_ket")
    def test_resolve_joint_number_stats_poly_ket(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out0 = JointOutcome(outcomes_by_port=(("a", NumberOutcome(count=0)),))
        out1 = JointOutcome(outcomes_by_port=(("a", NumberOutcome(count=1)),))
        mock_support.return_value = (out0, out1)

        proj0 = MagicMock()
        proj0.norm2 = 0.2
        proj1 = MagicMock()
        proj1.norm2 = 0.8
        mock_project.side_effect = [proj0, proj1]

        probs = resolve_joint_number_stats_poly_ket(MagicMock(), MagicMock())

        self.assertEqual(probs, {out0: 0.2, out1: 0.8})

    @patch("symop.polynomial.kernels.measurements.number.common.iter_joint_number_support_poly_ket")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_joint_number_poly_ket")
    def test_resolve_joint_number_stats_poly_ket_raises_on_negative_probability(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out1 = JointOutcome(outcomes_by_port=(("a", NumberOutcome(count=1)),))
        mock_support.return_value = (out1,)

        proj = MagicMock()
        proj.norm2 = -1e-3
        mock_project.return_value = proj

        with self.assertRaises(NumberMeasurementError):
            resolve_joint_number_stats_poly_ket(MagicMock(), MagicMock(), eps=1e-12)


class TestDensitySupportAndProjection(unittest.TestCase):
    @patch("symop.polynomial.kernels.measurements.number.common.selected_mode_signatures")
    def test_iter_number_support_poly_density(self, mock_selected_mode_signatures: MagicMock) -> None:
        mock_selected_mode_signatures.return_value = frozenset({"m0"})
        m0 = FakeMode("m0")

        state = MagicMock()
        state.rho.terms = (
            FakeDensityTerm(FakeMonomial(()), FakeMonomial(())),
            FakeDensityTerm(
                FakeMonomial((FakeCreator(m0),)),
                FakeMonomial((FakeCreator(m0),)),
            ),
            FakeDensityTerm(
                FakeMonomial((FakeCreator(m0),)),
                FakeMonomial(()),
            ),
        )

        support = iter_number_support_poly_density(state, MagicMock())

        self.assertEqual(
            support,
            (
                NumberOutcome(count=0),
                NumberOutcome(count=1),
            ),
        )

    @patch("symop.polynomial.kernels.measurements.number.common.selected_mode_signatures")
    @patch("symop.polynomial.kernels.measurements.number.common.DensityPolyState.from_densitypoly")
    def test_project_onto_number_poly_density(self, mock_from_densitypoly: MagicMock, mock_selected_mode_signatures: MagicMock) -> None:
        mock_selected_mode_signatures.return_value = frozenset({"m0"})

        term_keep = MagicMock()
        term_drop = MagicMock()

        rho_obj = MagicMock()
        rho_obj.terms = (term_keep, term_drop)

        rho_cls = MagicMock()
        rho_obj.__class__ = rho_cls

        combined = MagicMock()
        rho_cls.return_value.combine_like_terms.return_value = combined

        state = MagicMock()
        state.rho = rho_obj

        with patch(
            "symop.polynomial.kernels.measurements.number.common.count_selected_quanta_left",
            side_effect=[1, 0],
        ), patch(
            "symop.polynomial.kernels.measurements.number.common.count_selected_quanta_right",
            side_effect=[1, 0],
        ):
            out = project_onto_number_poly_density(
                state,
                MagicMock(),
                NumberOutcome(count=1),
            )

        self.assertIs(out, mock_from_densitypoly.return_value)
        rho_cls.assert_called_once_with((term_keep,))
        rho_cls.return_value.combine_like_terms.assert_called_once_with(eps=1e-12)
        mock_from_densitypoly.assert_called_once_with(combined)

    @patch("symop.polynomial.kernels.measurements.number.common.iter_number_support_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_number_poly_density")
    def test_resolve_number_stats_poly_density(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out0 = NumberOutcome(count=0)
        out1 = NumberOutcome(count=1)
        mock_support.return_value = (out0, out1)

        proj0 = MagicMock()
        proj0.trace.return_value = 0.3 + 0.0j
        proj1 = MagicMock()
        proj1.trace.return_value = 0.7 + 0.0j
        mock_project.side_effect = [proj0, proj1]

        stats = resolve_number_stats_poly_density(MagicMock(), MagicMock())

        self.assertEqual(stats.probabilities, {out0: 0.3, out1: 0.7})
        self.assertAlmostEqual(stats.expectation, 0.7)

    @patch("symop.polynomial.kernels.measurements.number.common.iter_number_support_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_number_poly_density")
    def test_resolve_number_stats_poly_density_raises_on_imaginary_probability(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out0 = NumberOutcome(count=0)
        mock_support.return_value = (out0,)

        proj = MagicMock()
        proj.trace.return_value = 0.3 + 1e-3j
        mock_project.return_value = proj

        with self.assertRaises(NumberMeasurementError):
            resolve_number_stats_poly_density(MagicMock(), MagicMock(), eps=1e-12)

    @patch("symop.polynomial.kernels.measurements.number.common.iter_number_support_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_number_poly_density")
    def test_resolve_number_stats_poly_density_raises_on_negative_probability(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out0 = NumberOutcome(count=0)
        mock_support.return_value = (out0,)

        proj = MagicMock()
        proj.trace.return_value = -1e-3 + 0.0j
        mock_project.return_value = proj

        with self.assertRaises(NumberMeasurementError):
            resolve_number_stats_poly_density(MagicMock(), MagicMock(), eps=1e-12)

    def test_discard_measured_modes_number_density_delegates_to_trace_out_modes(self) -> None:
        state = MagicMock()
        target = MagicMock()
        measured_modes = (MagicMock(),)
        state.resolve_target_modes.return_value = measured_modes
        state.trace_out_modes.return_value = "reduced"

        out = discard_measured_modes_number_density(state, target)

        self.assertEqual(out, "reduced")
        state.resolve_target_modes.assert_called_once_with(target)
        state.trace_out_modes.assert_called_once_with(measured_modes)


class TestDensityJointSupportAndProjection(unittest.TestCase):
    @patch("symop.polynomial.kernels.measurements.number.common.selected_mode_signatures_by_port")
    def test_iter_joint_number_support_poly_density(self, mock_selected_mode_signatures_by_port: MagicMock) -> None:
        mock_selected_mode_signatures_by_port.return_value = {
            "a": frozenset({"m0"}),
            "b": frozenset({"m1"}),
        }

        m0 = FakeMode("m0")
        m1 = FakeMode("m1")

        state = MagicMock()
        state.rho.terms = (
            FakeDensityTerm(
                FakeMonomial((FakeCreator(m0),)),
                FakeMonomial((FakeCreator(m0),)),
            ),
            FakeDensityTerm(
                FakeMonomial((FakeCreator(m0), FakeCreator(m1))),
                FakeMonomial((FakeCreator(m0), FakeCreator(m1))),
            ),
            FakeDensityTerm(
                FakeMonomial((FakeCreator(m0),)),
                FakeMonomial(()),
            ),
        )

        out = iter_joint_number_support_poly_density(state, MagicMock())

        self.assertEqual(
            out,
            (
                JointOutcome(
                    outcomes_by_port=(
                        ("a", NumberOutcome(count=1)),
                        ("b", NumberOutcome(count=0)),
                    )
                ),
                JointOutcome(
                    outcomes_by_port=(
                        ("a", NumberOutcome(count=1)),
                        ("b", NumberOutcome(count=1)),
                    )
                ),
            ),
        )

    @patch("symop.polynomial.kernels.measurements.number.common.selected_mode_signatures_by_port")
    @patch("symop.polynomial.kernels.measurements.number.common.DensityPolyState.from_densitypoly")
    def test_project_onto_joint_number_poly_density(self, mock_from_densitypoly: MagicMock, mock_selected_mode_signatures_by_port: MagicMock) -> None:
        mock_selected_mode_signatures_by_port.return_value = {
            "a": frozenset({"m0"}),
        }

        term_keep = MagicMock()
        term_drop = MagicMock()

        rho_obj = MagicMock()
        rho_obj.terms = (term_keep, term_drop)

        rho_cls = MagicMock()
        rho_obj.__class__ = rho_cls

        combined = MagicMock()
        rho_cls.return_value.combine_like_terms.return_value = combined

        state = MagicMock()
        state.rho = rho_obj

        expected_outcome = JointOutcome(
            outcomes_by_port=(("a", NumberOutcome(count=1)),)
        )

        with patch(
            "symop.polynomial.kernels.measurements.number.common.joint_number_counts_left",
            side_effect=[(("a", 1),), (("a", 0),)],
        ), patch(
            "symop.polynomial.kernels.measurements.number.common.joint_number_counts_right",
            side_effect=[(("a", 1),), (("a", 0),)],
        ):
            out = project_onto_joint_number_poly_density(
                state,
                MagicMock(),
                expected_outcome,
            )

        self.assertIs(out, mock_from_densitypoly.return_value)
        rho_cls.assert_called_once_with((term_keep,))
        rho_cls.return_value.combine_like_terms.assert_called_once_with(eps=1e-12)
        mock_from_densitypoly.assert_called_once_with(combined)

    @patch("symop.polynomial.kernels.measurements.number.common.iter_joint_number_support_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_joint_number_poly_density")
    def test_resolve_joint_number_stats_poly_density(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out0 = JointOutcome(outcomes_by_port=(("a", NumberOutcome(count=0)),))
        out1 = JointOutcome(outcomes_by_port=(("a", NumberOutcome(count=1)),))
        mock_support.return_value = (out0, out1)

        proj0 = MagicMock()
        proj0.trace.return_value = 0.2 + 0.0j
        proj1 = MagicMock()
        proj1.trace.return_value = 0.8 + 0.0j
        mock_project.side_effect = [proj0, proj1]

        probs = resolve_joint_number_stats_poly_density(MagicMock(), MagicMock())

        self.assertEqual(probs, {out0: 0.2, out1: 0.8})

    @patch("symop.polynomial.kernels.measurements.number.common.iter_joint_number_support_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_joint_number_poly_density")
    def test_resolve_joint_number_stats_poly_density_raises_on_imaginary_probability(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out0 = JointOutcome(outcomes_by_port=(("a", NumberOutcome(count=0)),))
        mock_support.return_value = (out0,)

        proj = MagicMock()
        proj.trace.return_value = 0.2 + 1e-3j
        mock_project.return_value = proj

        with self.assertRaises(NumberMeasurementError):
            resolve_joint_number_stats_poly_density(MagicMock(), MagicMock(), eps=1e-12)

    @patch("symop.polynomial.kernels.measurements.number.common.iter_joint_number_support_poly_density")
    @patch("symop.polynomial.kernels.measurements.number.common.project_onto_joint_number_poly_density")
    def test_resolve_joint_number_stats_poly_density_raises_on_negative_probability(self, mock_project: MagicMock, mock_support: MagicMock) -> None:
        out0 = JointOutcome(outcomes_by_port=(("a", NumberOutcome(count=0)),))
        mock_support.return_value = (out0,)

        proj = MagicMock()
        proj.trace.return_value = -1e-3 + 0.0j
        mock_project.return_value = proj

        with self.assertRaises(NumberMeasurementError):
            resolve_joint_number_stats_poly_density(MagicMock(), MagicMock(), eps=1e-12)
