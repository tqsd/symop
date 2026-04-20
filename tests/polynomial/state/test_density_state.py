from __future__ import annotations

import unittest

from symop.ccr.algebra.density.poly import DensityPoly
from symop.core.protocols.devices.label_edit import DeleteModeLabel, SetModeLabel
from symop.devices.measurement.target import MeasurementSelection, MeasurementTarget
from symop.modes.labels import Path
from symop.polynomial.state.density import DensityPolyState

from tests.polynomial.state._builders import (
    make_single_photon_ket,
    make_test_mode,
    make_two_mode_ket,
)


class UnsupportedEdit:
    pass


class TestDensityPolyStateConstruction(unittest.TestCase):
    def test_vacuum_has_expected_basic_properties(self):
        state = DensityPolyState.vacuum()

        self.assertEqual(state.rep_kind, "poly")
        self.assertEqual(state.state_kind, "density")
        self.assertEqual(state.modes, ())
        self.assertEqual(state.mode_by_signature, {})
        self.assertEqual(state.mode_labels, {})
        self.assertEqual(state.modes_by_path, {})

    def test_pure_builds_density_state_from_ket(self):
        ket = make_single_photon_ket(name="a", path="p0")

        state = DensityPolyState.pure(ket)

        self.assertEqual(state.rep_kind, "poly")
        self.assertEqual(state.state_kind, "density")
        self.assertEqual(len(state.modes), 1)
        self.assertTrue(state.is_trace_normalized())

    def test_from_densitypoly_combines_like_terms(self):
        ket = make_single_photon_ket(name="a", path="p0")
        rho = DensityPoly.pure(ket.ket) + DensityPoly.pure(ket.ket)

        state = DensityPolyState.from_densitypoly(rho)

        self.assertEqual(len(state.rho.terms), 1)


class TestDensityPolyStateMetadata(unittest.TestCase):
    def test_with_label_returns_updated_copy(self):
        state = DensityPolyState.vacuum()

        updated = state.with_label("rho")

        self.assertEqual(updated.label, "rho")
        self.assertEqual(updated.rho, state.rho)
        self.assertEqual(updated.index, state.index)
        self.assertNotEqual(id(updated), id(state))

    def test_with_index_returns_updated_copy(self):
        state = DensityPolyState.vacuum()

        updated = state.with_index(123)

        self.assertEqual(updated.index, 123)
        self.assertEqual(updated.rho, state.rho)
        self.assertEqual(updated.label, state.label)
        self.assertNotEqual(id(updated), id(state))


class TestDensityPolyStateSemanticAccess(unittest.TestCase):
    def test_modes_and_labels_are_exposed(self):
        ket_state, mode_a, mode_b = make_two_mode_ket()
        state = ket_state.to_density()

        self.assertEqual(set(state.modes), {mode_a, mode_b})
        self.assertEqual(state.mode_by_signature[mode_a.signature], mode_a)
        self.assertEqual(state.mode_by_signature[mode_b.signature], mode_b)
        self.assertEqual(state.mode_labels[mode_a.signature], mode_a.label)
        self.assertEqual(state.mode_labels[mode_b.signature], mode_b.label)

    def test_modes_by_path_groups_modes(self):
        ket_state, mode_a, mode_b = make_two_mode_ket()
        state = ket_state.to_density()

        self.assertEqual(state.modes_on_path(mode_a.label.path), (mode_a,))
        self.assertEqual(state.modes_on_path(mode_b.label.path), (mode_b,))
        self.assertEqual(state.modes_on_path(Path("missing")), ())

    def test_label_for_mode_returns_label(self):
        ket_state, mode_a, _ = make_two_mode_ket()
        state = ket_state.to_density()

        self.assertEqual(state.label_for_mode(mode_a.signature), mode_a.label)

    def test_label_for_mode_raises_for_missing_signature(self):
        state = DensityPolyState.vacuum()

        with self.assertRaises(KeyError):
            state.label_for_mode("missing-signature")

    def test_labels_on_path_returns_only_matching_labels(self):
        ket_state, mode_a, mode_b = make_two_mode_ket()
        state = ket_state.to_density()

        labels = state.labels_on_path(mode_a.label.path)

        self.assertEqual(labels, {mode_a.signature: mode_a.label})
        self.assertNotIn(mode_b.signature, labels)


class TestDensityPolyStateRelabeling(unittest.TestCase):
    def test_relabel_modes_replaces_selected_mode(self):
        ket_state, mode_a, mode_b = make_two_mode_ket()
        state = ket_state.to_density()
        new_mode_a = mode_a.with_path(Path("out0"))

        updated = state.relabel_modes({mode_a.signature: new_mode_a})

        self.assertIn(new_mode_a, updated.modes)
        self.assertIn(mode_b, updated.modes)
        self.assertNotIn(mode_a, updated.modes)

    def test_relabel_paths_updates_matching_paths_only(self):
        ket_state, mode_a, mode_b = make_two_mode_ket()
        state = ket_state.to_density()

        updated = state.relabel_paths({mode_a.label.path: Path("out0")})

        updated_paths = {mode.label.path for mode in updated.modes}
        self.assertIn(Path("out0"), updated_paths)
        self.assertIn(mode_b.label.path, updated_paths)

    def test_relabel_paths_returns_self_when_nothing_matches(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()

        updated = state.relabel_paths({Path("does-not-exist"): Path("out0")})

        self.assertIs(updated, state)

    def test_relabel_labels_updates_selected_label(self):
        ket_state, mode_a, mode_b = make_two_mode_ket()
        state = ket_state.to_density()
        new_label = mode_a.label.with_path(Path("out0"))
        expected_mode_a = mode_a.with_label(new_label)

        updated = state.relabel_labels({mode_a.signature: new_label})

        self.assertIn(expected_mode_a, updated.modes)
        self.assertIn(mode_b, updated.modes)
        self.assertNotIn(mode_a, updated.modes)
        self.assertEqual(updated.label_for_mode(expected_mode_a.signature), new_label)
        self.assertEqual(updated.label_for_mode(mode_b.signature), mode_b.label)

    def test_relabel_labels_raises_for_missing_signature(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()
        mode = make_test_mode(name="x", path="p9")

        with self.assertRaises(KeyError):
            state.relabel_labels({mode.signature: mode.label})

    def test_relabel_labels_changes_mode_signature(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()
        mode = state.modes[0]
        new_label = mode.label.with_path(Path("out0"))
        expected_mode = mode.with_label(new_label)

        updated = state.relabel_labels({mode.signature: new_label})

        self.assertNotIn(mode.signature, updated.mode_by_signature)
        self.assertIn(expected_mode.signature, updated.mode_by_signature)


class TestDensityPolyStateLabelEdits(unittest.TestCase):
    def test_apply_label_edits_applies_set_mode_label(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()
        mode = state.modes[0]
        new_label = mode.label.with_path(Path("out0"))
        expected_mode = mode.with_label(new_label)

        updated = state.apply_label_edits(
            [SetModeLabel(mode_sig=mode.signature, label=new_label)]
        )

        self.assertIn(expected_mode, updated.modes)
        self.assertNotIn(mode, updated.modes)
        self.assertEqual(updated.label_for_mode(expected_mode.signature), new_label)

    def test_apply_label_edits_raises_for_missing_mode(self):
        state = DensityPolyState.vacuum()
        mode = make_test_mode(name="a", path="p0")

        with self.assertRaises(KeyError):
            state.apply_label_edits(
                [SetModeLabel(mode_sig=mode.signature, label=mode.label)]
            )

    def test_apply_label_edits_rejects_delete(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()
        mode = state.modes[0]

        with self.assertRaises(NotImplementedError):
            state.apply_label_edits([DeleteModeLabel(mode_sig=mode.signature)])

    def test_apply_label_edits_rejects_unknown_edit_type(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()

        with self.assertRaises(TypeError):
            state.apply_label_edits([UnsupportedEdit()])


class TestDensityPolyStateFilteredOnPath(unittest.TestCase):
    def test_filtered_on_path_updates_only_selected_path(self):
        ket_state, mode_a, mode_b = make_two_mode_ket()
        state = ket_state.to_density()

        def update_label(label):
            return label.with_path(label.path)

        expected_label_a = update_label(mode_a.label).with_path(Path("out0"))
        expected_mode_a = mode_a.with_label(expected_label_a)

        updated = state.filtered_on_path(
            in_path=mode_a.label.path,
            out_path=Path("out0"),
            update_label=update_label,
        )

        self.assertIn(expected_mode_a, updated.modes)
        self.assertIn(mode_b, updated.modes)
        self.assertNotIn(mode_a, updated.modes)
        self.assertEqual(updated.label_for_mode(expected_mode_a.signature).path, Path("out0"))
        self.assertEqual(updated.label_for_mode(mode_b.signature), mode_b.label)

    def test_filtered_on_path_returns_self_when_no_modes_match(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()

        updated = state.filtered_on_path(
            in_path=Path("missing"),
            out_path=Path("out0"),
            update_label=lambda label: label,
        )

        self.assertIs(updated, state)


class TestDensityPolyStateTraceBehavior(unittest.TestCase):
    def test_trace_returns_one_for_pure_normalized_state(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()

        self.assertEqual(state.trace(), 1.0)
        self.assertTrue(state.is_trace_normalized())

    def test_normalize_trace_normalizes_scaled_state(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()
        scaled = DensityPolyState(state.rho.scaled(3.0))

        self.assertFalse(scaled.is_trace_normalized())

        normalized = scaled.normalize_trace()

        self.assertTrue(normalized.is_trace_normalized())
        self.assertEqual(normalized.trace(), 1.0)

    def test_normalize_trace_raises_for_near_zero_trace(self):
        zero_trace_state = DensityPolyState(DensityPoly.identity().scaled(0.0))

        with self.assertRaises(ValueError):
            zero_trace_state.normalize_trace()

    def test_is_trace_normalized_respects_tolerance(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()
        perturbed = DensityPolyState(state.rho.scaled(1.0 + 1e-15))

        self.assertTrue(perturbed.is_trace_normalized())
        self.assertFalse(perturbed.is_trace_normalized(eps=1e-16))


class TestDensityPolyStateComposition(unittest.TestCase):
    def test_multiply_returns_product_state(self):
        left = make_single_photon_ket(name="a", path="p0").to_density()
        right = make_single_photon_ket(name="b", path="p1").to_density()

        product = left.multiply(right)

        self.assertIsInstance(product, DensityPolyState)
        self.assertEqual(product.rep_kind, "poly")
        self.assertEqual(product.state_kind, "density")
    def test_join_is_alias_for_multiply(self):
        left = make_single_photon_ket(name="a", path="p0").to_density()
        right = make_single_photon_ket(name="b", path="p1").to_density()

        product = left.multiply(right)
        joined = left.join(right)

        self.assertEqual(joined.rho, product.rho)


class TestDensityPolyStateTraceOut(unittest.TestCase):
    def test_trace_out_modes_removes_selected_mode(self):
        ket_state, mode_a, mode_b = make_two_mode_ket()
        state = ket_state.to_density()

        reduced = state.trace_out_modes([mode_a])

        remaining_sigs = {mode.signature for mode in reduced.modes}
        self.assertNotIn(mode_a.signature, remaining_sigs)
        self.assertIn(mode_b.signature, remaining_sigs)

    def test_trace_out_signatures_removes_selected_mode(self):
        ket_state, mode_a, mode_b = make_two_mode_ket()
        state = ket_state.to_density()

        reduced = state.trace_out_signatures([mode_a.signature])

        remaining_sigs = {mode.signature for mode in reduced.modes}
        self.assertNotIn(mode_a.signature, remaining_sigs)
        self.assertIn(mode_b.signature, remaining_sigs)

    def test_trace_out_signatures_ignores_missing_signatures(self):
        state = make_single_photon_ket(name="a", path="p0").to_density()
        original_rho = state.rho

        reduced = state.trace_out_signatures(["missing-signature"])

        self.assertEqual(reduced.rho, original_rho)


class TestDensityPolyStateResolveTargetModes(unittest.TestCase):
    def test_resolve_target_modes_by_path(self):
        ket_state, mode_a, _ = make_two_mode_ket()
        state = ket_state.to_density()
        target = MeasurementTarget(
            selections=(
                MeasurementSelection(
                    port_name="d0",
                    paths=(mode_a.label.path,),
                ),
            )
        )

        resolved = state.resolve_target_modes(target)

        self.assertEqual(resolved, (mode_a,))

    def test_resolve_target_modes_by_signature(self):
        ket_state, mode_a, _ = make_two_mode_ket()
        state = ket_state.to_density()
        target = MeasurementTarget(
            selections=(
                MeasurementSelection(
                    port_name="d0",
                    mode_sigs=(mode_a.signature,),
                ),
            )
        )

        resolved = state.resolve_target_modes(target)

        self.assertEqual(resolved, (mode_a,))

    def test_resolve_target_modes_deduplicates(self):
        ket_state, mode_a, _ = make_two_mode_ket()
        state = ket_state.to_density()
        target = MeasurementTarget(
            selections=(
                MeasurementSelection(
                    port_name="d0",
                    paths=(mode_a.label.path,),
                    mode_sigs=(mode_a.signature,),
                ),
            )
        )

        resolved = state.resolve_target_modes(target)

        self.assertEqual(resolved, (mode_a,))

    def test_resolve_target_modes_ignores_missing_signature(self):
        ket_state, mode_a, _ = make_two_mode_ket()
        state = ket_state.to_density()
        target = MeasurementTarget(
            selections=(
                MeasurementSelection(
                    port_name="d0",
                    paths=(mode_a.label.path,),
                    mode_sigs=("missing-signature",),
                ),
            )
        )

        resolved = state.resolve_target_modes(target)

        self.assertEqual(resolved, (mode_a,))
