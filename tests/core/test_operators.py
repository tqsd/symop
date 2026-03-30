import unittest

from symop.core.operators import LadderOp, ModeOp
from symop.core.types import OperatorKind

from tests.core.support.fakes import (
    FakeComponentLabel,
    make_mode,
    make_mode_label,
    set_symmetric_overlap,
)


class TestModeOp(unittest.TestCase):
    def test_post_init_creates_cached_ann_and_cre(self) -> None:
        mode = make_mode()

        self.assertIsInstance(mode.ann, LadderOp)
        self.assertIsInstance(mode.cre, LadderOp)

        self.assertIs(mode.ann, mode.ann)
        self.assertIs(mode.cre, mode.cre)

        self.assertIs(mode.ann.mode, mode)
        self.assertIs(mode.cre.mode, mode)

        self.assertEqual(mode.ann.kind, OperatorKind.ANN)
        self.assertEqual(mode.cre.kind, OperatorKind.CRE)

    def test_aliases_return_same_objects(self) -> None:
        mode = make_mode()

        self.assertIs(mode.annihilate, mode.ann)
        self.assertIs(mode.create, mode.cre)

    def test_with_user_label_returns_updated_copy(self) -> None:
        mode = make_mode(user_label="old")
        updated = mode.with_user_label("new")

        self.assertEqual(mode.user_label, "old")
        self.assertEqual(updated.user_label, "new")
        self.assertEqual(updated.label, mode.label)
        self.assertNotEqual(id(updated), id(mode))

    def test_with_index_returns_updated_copy(self) -> None:
        mode = make_mode(display_index=1)
        updated = mode.with_index(7)

        self.assertEqual(mode.display_index, 1)
        self.assertEqual(updated.display_index, 7)
        self.assertEqual(updated.label, mode.label)

    def test_with_label_returns_updated_copy(self) -> None:
        mode = make_mode(path="a")
        new_label = make_mode_label(path="b")
        updated = mode.with_label(new_label)

        self.assertEqual(updated.label, new_label)
        self.assertEqual(updated.user_label, mode.user_label)
        self.assertEqual(updated.display_index, mode.display_index)

    def test_with_path_returns_updated_copy(self) -> None:
        mode = make_mode(path="a")
        new_path = FakeComponentLabel("path", "b")
        updated = mode.with_path(new_path)

        self.assertEqual(updated.label.path, new_path)
        self.assertEqual(updated.label.polarization, mode.label.polarization)
        self.assertEqual(updated.label.envelope, mode.label.envelope)

    def test_with_polarization_returns_updated_copy(self) -> None:
        mode = make_mode(polarization="h")
        new_pol = FakeComponentLabel("polarization", "v")
        updated = mode.with_polarization(new_pol)

        self.assertEqual(updated.label.polarization, new_pol)
        self.assertEqual(updated.label.path, mode.label.path)
        self.assertEqual(updated.label.envelope, mode.label.envelope)

    def test_with_envelope_returns_updated_copy(self) -> None:
        mode = make_mode(envelope="e1")
        new_env = FakeComponentLabel("envelope", "e2")
        updated = mode.with_envelope(new_env)

        self.assertEqual(updated.label.envelope, new_env)
        self.assertEqual(updated.label.path, mode.label.path)
        self.assertEqual(updated.label.polarization, mode.label.polarization)

    def test_signature(self) -> None:
        mode = make_mode(path="p1", polarization="h", envelope="env1")

        self.assertEqual(
            mode.signature,
            ("mode", mode.label.signature),
        )

    def test_approx_signature(self) -> None:
        mode = make_mode(path="p1", polarization="h", envelope="env1")

        self.assertEqual(
            mode.approx_signature(decimals=7, ignore_global_phase=True),
            (
                "mode_approx",
                mode.label.approx_signature(
                    decimals=7,
                    ignore_global_phase=True,
                ),
            ),
        )


class TestLadderOp(unittest.TestCase):
    def test_is_annihilation_and_is_creation(self) -> None:
        mode = make_mode()

        self.assertTrue(mode.ann.is_annihilation)
        self.assertFalse(mode.ann.is_creation)

        self.assertTrue(mode.cre.is_creation)
        self.assertFalse(mode.cre.is_annihilation)

    def test_dagger_of_annihilation_is_creation(self) -> None:
        mode = make_mode()

        self.assertIs(mode.ann.dagger(), mode.cre)

    def test_dagger_of_creation_is_annihilation(self) -> None:
        mode = make_mode()

        self.assertIs(mode.cre.dagger(), mode.ann)

    def test_commutator_ann_cre_same_mode_is_one(self) -> None:
        mode = make_mode()

        self.assertEqual(mode.ann.commutator(mode.cre), 1.0 + 0.0j)

    def test_commutator_cre_ann_same_mode_is_minus_one(self) -> None:
        mode = make_mode()

        self.assertEqual(mode.cre.commutator(mode.ann), -1.0 + 0.0j)

    def test_commutator_ann_ann_is_zero(self) -> None:
        mode = make_mode()

        self.assertEqual(mode.ann.commutator(mode.ann), 0.0 + 0.0j)

    def test_commutator_cre_cre_is_zero(self) -> None:
        mode = make_mode()

        self.assertEqual(mode.cre.commutator(mode.cre), 0.0 + 0.0j)

    def test_commutator_uses_mode_overlap(self) -> None:
        envelope_table = {}

        left = make_mode(
            path="p",
            polarization="h",
            envelope="e1",
            envelope_table=envelope_table,
        )
        right = make_mode(
            path="p",
            polarization="h",
            envelope="e2",
            envelope_table=envelope_table,
        )

        set_symmetric_overlap(
            envelope_table,
            left.label.envelope,
            right.label.envelope,
            0.25 + 0.5j,
        )

        self.assertEqual(left.ann.commutator(right.cre), 0.25 + 0.5j)
        self.assertEqual(left.cre.commutator(right.ann), -0.25 - 0.5j)

    def test_commutator_zero_if_overlap_below_threshold(self) -> None:
        envelope_table = {}

        left = make_mode(
            path="p",
            polarization="h",
            envelope="e1",
            envelope_table=envelope_table,
        )
        right = make_mode(
            path="p",
            polarization="h",
            envelope="e2",
            envelope_table=envelope_table,
        )

        set_symmetric_overlap(
            envelope_table,
            left.label.envelope,
            right.label.envelope,
            1e-16 + 0.0j,
        )

        self.assertEqual(left.ann.commutator(right.cre), 0.0 + 0.0j)
        self.assertEqual(left.cre.commutator(right.ann), 0.0 + 0.0j)

    def test_commutator_nonzero_at_threshold_or_above(self) -> None:
        envelope_table = {}

        left = make_mode(
            path="p",
            polarization="h",
            envelope="e1",
            envelope_table=envelope_table,
        )
        right = make_mode(
            path="p",
            polarization="h",
            envelope="e2",
            envelope_table=envelope_table,
        )

        value = 1e-15 + 0.0j
        set_symmetric_overlap(
            envelope_table,
            left.label.envelope,
            right.label.envelope,
            value,
        )

        self.assertEqual(left.ann.commutator(right.cre), value)
        self.assertEqual(left.cre.commutator(right.ann), -value)

    def test_signature(self) -> None:
        mode = make_mode(path="p1", polarization="h", envelope="env1")

        self.assertEqual(
            mode.ann.signature,
            ("lop", OperatorKind.ANN.value, mode.signature),
        )
        self.assertEqual(
            mode.cre.signature,
            ("lop", OperatorKind.CRE.value, mode.signature),
        )

    def test_approx_signature(self) -> None:
        mode = make_mode(path="p1", polarization="h", envelope="env1")

        self.assertEqual(
            mode.ann.approx_signature(decimals=9, ignore_global_phase=True),
            (
                "lop",
                OperatorKind.ANN.value,
                mode.approx_signature(
                    decimals=9,
                    ignore_global_phase=True,
                ),
            ),
        )
        self.assertEqual(
            mode.cre.approx_signature(decimals=9, ignore_global_phase=True),
            (
                "lop",
                OperatorKind.CRE.value,
                mode.approx_signature(
                    decimals=9,
                    ignore_global_phase=True,
                ),
            ),
        )

    def test_with_label_rebuilds_cached_ladder_ops_for_new_mode(self) -> None:
        mode = make_mode(path="a")
        updated = mode.with_label(make_mode_label(path="b"))

        self.assertIs(updated.ann.mode, updated)
        self.assertIs(updated.cre.mode, updated)
        self.assertIsNot(updated.ann, mode.ann)
        self.assertIsNot(updated.cre, mode.cre)
if __name__ == "__main__":
    unittest.main()
