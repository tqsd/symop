from __future__ import annotations

import unittest

from symop.ccr.algebra.ket.poly import KetPoly
from symop.core.protocols.devices.label_edit import DeleteModeLabel, SetModeLabel
from symop.devices.measurement.target import MeasurementSelection, MeasurementTarget
from symop.modes.labels import Path
from symop.polynomial.state.ket import KetPolyState

from tests.polynomial.state._builders import (
    make_single_photon_ket,
    make_test_mode,
    make_two_mode_ket,
)


class UnsupportedEdit:
    pass


class TestKetPolyStateConstruction(unittest.TestCase):
    def test_vacuum_has_expected_basic_properties(self):
        state = KetPolyState.vacuum()

        self.assertEqual(state.rep_kind, "poly")
        self.assertEqual(state.state_kind, "ket")
        self.assertEqual(state.modes, ())
        self.assertEqual(state.mode_by_signature, {})
        self.assertEqual(state.mode_labels, {})
        self.assertEqual(state.modes_by_path, {})

    def test_from_creators_builds_state(self):
        mode = make_test_mode(name="a", path="p0")
        state = KetPolyState.from_creators([mode.cre])

        self.assertEqual(state.rep_kind, "poly")
        self.assertEqual(state.state_kind, "ket")
        self.assertEqual(len(state.modes), 1)
        self.assertIn(mode.signature, state.mode_by_signature)

    def test_from_creators_rejects_annihilator(self):
        mode = make_test_mode(name="a", path="p0")

        with self.assertRaises(ValueError):
            KetPolyState.from_creators([mode.ann])

    def test_post_init_rejects_non_creator_only_ket(self):
        mode = make_test_mode(name="a", path="p0")
        bad_ket = KetPoly.from_ops(
            creators=(),
            annihilators=(mode.ann,),
            coeff=1.0,
        )

        with self.assertRaises(ValueError):
            KetPolyState(bad_ket)

    def test_from_ketpoly_combines_like_terms(self):
        mode = make_test_mode(name="a", path="p0")
        ket = (
            KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=1.0)
            + KetPoly.from_ops(creators=(mode.cre,), annihilators=(), coeff=2.0)
        )
        state = KetPolyState.from_ketpoly(ket)

        self.assertEqual(len(state.ket.terms), 1)


class TestKetPolyStateMetadata(unittest.TestCase):
    def test_with_label_returns_updated_copy(self):
        state = KetPolyState.vacuum()
        updated = state.with_label("psi")

        self.assertEqual(updated.label, "psi")
        self.assertEqual(updated.ket, state.ket)
        self.assertEqual(updated.index, state.index)
        self.assertNotEqual(id(updated), id(state))

    def test_with_index_returns_updated_copy(self):
        state = KetPolyState.vacuum()
        updated = state.with_index(123)

        self.assertEqual(updated.index, 123)
        self.assertEqual(updated.ket, state.ket)
        self.assertEqual(updated.label, state.label)
        self.assertNotEqual(id(updated), id(state))


class TestKetPolyStateSemanticAccess(unittest.TestCase):
    def test_modes_and_labels_are_exposed(self):
        state, mode_a, mode_b = make_two_mode_ket()

        self.assertEqual(set(state.modes), {mode_a, mode_b})
        self.assertEqual(state.mode_by_signature[mode_a.signature], mode_a)
        self.assertEqual(state.mode_by_signature[mode_b.signature], mode_b)
        self.assertEqual(state.mode_labels[mode_a.signature], mode_a.label)
        self.assertEqual(state.mode_labels[mode_b.signature], mode_b.label)

    def test_modes_by_path_groups_modes(self):
        state, mode_a, mode_b = make_two_mode_ket()

        self.assertEqual(state.modes_on_path(mode_a.label.path), (mode_a,))
        self.assertEqual(state.modes_on_path(mode_b.label.path), (mode_b,))
        self.assertEqual(state.modes_on_path(Path("missing")), ())

    def test_label_for_mode_returns_label(self):
        state, mode_a, _ = make_two_mode_ket()

        self.assertEqual(state.label_for_mode(mode_a.signature), mode_a.label)

    def test_label_for_mode_raises_for_missing_signature(self):
        state = KetPolyState.vacuum()

        with self.assertRaises(KeyError):
            state.label_for_mode("missing-signature")

    def test_labels_on_path_returns_only_matching_labels(self):
        state, mode_a, mode_b = make_two_mode_ket()

        labels = state.labels_on_path(mode_a.label.path)

        self.assertEqual(labels, {mode_a.signature: mode_a.label})
        self.assertNotIn(mode_b.signature, labels)


class TestKetPolyStateRelabeling(unittest.TestCase):
    def test_relabel_modes_replaces_selected_mode(self):
        state, mode_a, mode_b = make_two_mode_ket()
        new_mode_a = mode_a.with_path(Path("out0"))

        updated = state.relabel_modes({mode_a.signature: new_mode_a})

        updated_mode_a = next(
            mode for mode in updated.modes if mode.user_label == mode_a.user_label
        )
        self.assertEqual(updated_mode_a.label.path, Path("out0"))
        self.assertIn(mode_b, updated.modes)

    def test_relabel_paths_updates_matching_paths_only(self):
        state, mode_a, mode_b = make_two_mode_ket()

        updated = state.relabel_paths({mode_a.label.path: Path("out0")})

        updated_paths = {mode.label.path for mode in updated.modes}
        self.assertIn(Path("out0"), updated_paths)
        self.assertIn(mode_b.label.path, updated_paths)

    def test_relabel_paths_returns_self_when_nothing_matches(self):
        state, _, _ = make_two_mode_ket()

        updated = state.relabel_paths({Path("does-not-exist"): Path("out0")})

        self.assertIs(updated, state)

    def test_relabel_labels_updates_selected_label(self):
        state, mode_a, mode_b = make_two_mode_ket()
        new_label = mode_a.label.with_path(Path("out0"))
        expected_mode_a = mode_a.with_label(new_label)

        updated = state.relabel_labels({mode_a.signature: new_label})

        self.assertIn(expected_mode_a, updated.modes)
        self.assertIn(mode_b, updated.modes)
        self.assertNotIn(mode_a, updated.modes)
        self.assertEqual(updated.label_for_mode(expected_mode_a.signature), new_label)

        self.assertEqual(updated.label_for_mode(mode_b.signature), mode_b.label)

    def test_relabel_labels_raises_for_missing_signature(self):
        state = make_single_photon_ket()

        mode = make_test_mode(name="x", path="p9")
        with self.assertRaises(KeyError):
            state.relabel_labels({mode.signature: mode.label})


class TestKetPolyStateLabelEdits(unittest.TestCase):
    def test_apply_label_edits_applies_set_mode_label(self):
        state = make_single_photon_ket(name="a", path="p0")
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
        state = KetPolyState.vacuum()
        mode = make_test_mode(name="a", path="p0")

        with self.assertRaises(KeyError):
            state.apply_label_edits(
                [SetModeLabel(mode_sig=mode.signature, label=mode.label)]
            )

    def test_apply_label_edits_rejects_delete(self):
        state = make_single_photon_ket(name="a", path="p0")
        mode = state.modes[0]

        with self.assertRaises(NotImplementedError):
            state.apply_label_edits([DeleteModeLabel(mode_sig=mode.signature)])

    def test_apply_label_edits_rejects_unknown_edit_type(self):
        state = make_single_photon_ket(name="a", path="p0")

        with self.assertRaises(TypeError):
            state.apply_label_edits([UnsupportedEdit()])


class TestKetPolyStateFilteredOnPath(unittest.TestCase):
    def test_filtered_on_path_updates_only_selected_path(self):
        state, mode_a, mode_b = make_two_mode_ket()

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
        state, _, _ = make_two_mode_ket()

        updated = state.filtered_on_path(
            in_path=Path("missing"),
            out_path=Path("out0"),
            update_label=lambda label: label,
        )

        self.assertIs(updated, state)


class TestKetPolyStateNormAndConversion(unittest.TestCase):
    def test_norm2_and_normalized(self):
        mode = make_test_mode(name="a", path="p0")
        state = KetPolyState.from_creators([mode.cre], coeff=2.0)

        self.assertFalse(state.is_normalized())
        self.assertGreater(state.norm2, 1.0)

        normalized = state.normalized()

        self.assertTrue(normalized.is_normalized())

    def test_to_density_returns_density_state(self):
        state = make_single_photon_ket()
        rho = state.to_density()

        self.assertEqual(rho.rep_kind, "poly")
        self.assertEqual(rho.state_kind, "density")
        self.assertTrue(rho.is_trace_normalized())


class TestKetPolyStateComposition(unittest.TestCase):
    def test_multiply_returns_product_state(self):
        left = make_single_photon_ket(name="a", path="p0")
        right = make_single_photon_ket(name="b", path="p1")

        product = left.multiply(right)

        self.assertEqual(product.state_kind, "ket")
        self.assertEqual(len(product.modes), 2)

    def test_join_is_alias_for_multiply(self):
        left = make_single_photon_ket(name="a", path="p0")
        right = make_single_photon_ket(name="b", path="p1")

        product = left.multiply(right)
        joined = left.join(right)

        self.assertEqual(joined.ket, product.ket)


class TestKetPolyStateResolveTargetModes(unittest.TestCase):
    def test_resolve_target_modes_by_path(self):
        state, mode_a, _ = make_two_mode_ket()
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
        state, mode_a, _ = make_two_mode_ket()
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
        state, mode_a, _ = make_two_mode_ket()
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
        state, mode_a, _ = make_two_mode_ket()
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
