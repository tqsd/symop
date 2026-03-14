from __future__ import annotations

import unittest
from itertools import count

import symop.core.operators as opmod


class FakeEnvelope:
    def __init__(
        self, name: str, overlaps: dict[str, complex] | None = None
    ) -> None:
        self.name = name
        self._overlaps = overlaps or {}

    @property
    def signature(self):
        return ("env", self.name)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        return ("env_approx", self.name, decimals, ignore_global_phase)

    def overlap(self, other: FakeEnvelope) -> complex:
        if other.name == self.name:
            return 1.0 + 0.0j
        key = f"{self.name}|{other.name}"
        if key in self._overlaps:
            return self._overlaps[key]
        key_rev = f"{other.name}|{self.name}"
        if key_rev in self._overlaps:
            return self._overlaps[key_rev]
        return 0.0 + 0.0j


class FakeLabel:
    def __init__(
        self,
        name: str,
        *,
        pol: str | None = None,
        path: str | None = None,
        overlaps: dict[str, complex] | None = None,
    ) -> None:
        self.name = name
        self.pol = pol
        self.path = path
        self._overlaps = overlaps or {}

    @property
    def signature(self):
        return ("label", self.name, self.pol, self.path)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        return (
            "label_approx",
            self.name,
            self.pol,
            self.path,
            decimals,
            ignore_global_phase,
        )

    def overlap(self, other: FakeLabel) -> complex:
        if other.signature == self.signature:
            return 1.0 + 0.0j
        key = f"{self.signature}|{other.signature}"
        if key in self._overlaps:
            return self._overlaps[key]
        key_rev = f"{other.signature}|{self.signature}"
        if key_rev in self._overlaps:
            return self._overlaps[key_rev]
        return 0.0 + 0.0j

    def with_polarization(self, pol):
        return FakeLabel(
            self.name, pol=str(pol), path=self.path, overlaps=self._overlaps
        )

    def with_path(self, path):
        return FakeLabel(
            self.name, pol=self.pol, path=str(path), overlaps=self._overlaps
        )


class FakeModeLabel:
    """
    Minimal composite label for operator tests.

    It mimics the real ModeLabel behavior:
    overlap factorizes into base_label_overlap * envelope_overlap.
    """

    def __init__(self, *, base: FakeLabel, envelope: FakeEnvelope) -> None:
        self.base = base
        self.envelope = envelope

    @property
    def signature(self):
        return ("mode_label", self.base.signature, self.envelope.signature)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        return (
            "mode_label_approx",
            self.base.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            ),
            self.envelope.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            ),
        )

    def overlap(self, other: FakeModeLabel) -> complex:
        return self.base.overlap(other.base) * self.envelope.overlap(
            other.envelope
        )

    def with_polarization(self, pol):
        return FakeModeLabel(
            base=self.base.with_polarization(pol), envelope=self.envelope
        )

    def with_path(self, path):
        return FakeModeLabel(
            base=self.base.with_path(path), envelope=self.envelope
        )

    def with_envelope(self, envelope: FakeEnvelope):
        return FakeModeLabel(base=self.base, envelope=envelope)

    @property
    def pol(self):
        return self.base.pol

    @property
    def path(self):
        return self.base.path


class TestOperatorKind(unittest.TestCase):
    def test_values(self):
        self.assertEqual(opmod.OperatorKind.ANN.value, "a")
        self.assertEqual(opmod.OperatorKind.CREATE.value, "adag")


class TestModeOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        opmod._mode_display_counter = count(1)

    def test_post_init_creates_cached_ladder_ops(self):
        env = FakeEnvelope("e1")
        base = FakeLabel("l1")
        label = FakeModeLabel(base=base, envelope=env)

        m = opmod.ModeOp(label=label)

        self.assertIsNotNone(m.display_index)
        self.assertIsNotNone(m.ann)
        self.assertIsNotNone(m.create)

        self.assertTrue(m.ann.is_annihilation)
        self.assertFalse(m.ann.is_creation)

        self.assertTrue(m.create.is_creation)
        self.assertFalse(m.create.is_annihilation)

        self.assertIs(m.ann.mode, m)
        self.assertIs(m.create.mode, m)

        self.assertEqual(m.ann.kind.value, "a")
        self.assertEqual(m.create.kind.value, "adag")

    def test_with_helpers_return_new_instances(self):
        env1 = FakeEnvelope("e1")
        env2 = FakeEnvelope("e2")

        base1 = FakeLabel("l1", pol="H", path="p1")
        base2 = FakeLabel("l2", pol="V", path="p2")

        label1 = FakeModeLabel(base=base1, envelope=env1)
        label2 = FakeModeLabel(base=base2, envelope=env2)

        m = opmod.ModeOp(label=label1)

        m2 = m.with_user_label("tag")
        self.assertIsNot(m2, m)
        self.assertEqual(m2.user_label, "tag")
        self.assertEqual(m.user_label, None)

        m3 = m.with_index(123)
        self.assertEqual(m3.display_index, 123)
        self.assertNotEqual(m.display_index, 123)

        m4 = m.with_envelope(env2)
        self.assertIs(m4.label.envelope, env2)
        self.assertIs(m.label.envelope, env1)

        m5 = m.with_label(label2)
        self.assertIs(m5.label, label2)
        self.assertIs(m.label, label1)

        m6 = m.with_polarization("D")
        self.assertEqual(m6.label.pol, "D")
        self.assertEqual(m6.label.path, "p1")
        self.assertEqual(m.label.pol, "H")

        m7 = m.with_path("pX")
        self.assertEqual(m7.label.path, "pX")
        self.assertEqual(m7.label.pol, "H")
        self.assertEqual(m.label.path, "p1")

    def test_signature_and_approx_signature(self):
        env = FakeEnvelope("e1")
        base = FakeLabel("l1", pol="H", path="p1")
        label = FakeModeLabel(base=base, envelope=env)
        m = opmod.ModeOp(label=label)

        self.assertEqual(m.signature, ("mode", label.signature))

        approx = m.approx_signature(decimals=7, ignore_global_phase=True)
        self.assertEqual(
            approx,
            (
                "mode_approx",
                label.approx_signature(decimals=7, ignore_global_phase=True),
            ),
        )


class TestLadderOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        opmod._mode_display_counter = count(1)

    def make_modes_for_commutator(
        self, *, env_overlap: complex, label_overlap: complex
    ):
        env_overlaps = {"e1|e2": env_overlap}

        base1 = FakeLabel("l1")
        base2 = FakeLabel("l2")

        label_overlaps = {}
        key = f"{base1.signature}|{base2.signature}"
        label_overlaps[key] = label_overlap

        base1 = FakeLabel("l1", overlaps=label_overlaps)
        base2 = FakeLabel("l2", overlaps=label_overlaps)

        env1 = FakeEnvelope("e1", overlaps=env_overlaps)
        env2 = FakeEnvelope("e2", overlaps=env_overlaps)

        label1 = FakeModeLabel(base=base1, envelope=env1)
        label2 = FakeModeLabel(base=base2, envelope=env2)

        m1 = opmod.ModeOp(label=label1)
        m2 = opmod.ModeOp(label=label2)
        return m1, m2

    def test_is_annihilation_is_creation(self):
        env = FakeEnvelope("e1")
        base = FakeLabel("l1")
        label = FakeModeLabel(base=base, envelope=env)
        m = opmod.ModeOp(label=label)

        self.assertTrue(m.ann.is_annihilation)
        self.assertFalse(m.ann.is_creation)

        self.assertTrue(m.create.is_creation)
        self.assertFalse(m.create.is_annihilation)

    def test_dagger_involution_and_identity(self):
        env = FakeEnvelope("e1")
        base = FakeLabel("l1")
        label = FakeModeLabel(base=base, envelope=env)
        m = opmod.ModeOp(label=label)

        self.assertIs(m.ann.dagger(), m.create)
        self.assertIs(m.create.dagger(), m.ann)

        self.assertIs(m.ann.dagger().dagger(), m.ann)
        self.assertIs(m.create.dagger().dagger(), m.create)

    def test_commutator_zero_if_overlap_small(self):
        m1, m2 = self.make_modes_for_commutator(
            env_overlap=0.7 + 0.2j,
            label_overlap=0.0 + 0.0j,
        )

        self.assertEqual(m1.ann.commutator(m2.create), 0.0 + 0.0j)
        self.assertEqual(m1.create.commutator(m2.ann), 0.0 + 0.0j)

    def test_commutator_ann_with_create_positive(self):
        env_overlap = 0.3 - 0.1j
        label_overlap = 0.5 + 0.0j
        m1, m2 = self.make_modes_for_commutator(
            env_overlap=env_overlap,
            label_overlap=label_overlap,
        )

        out = m1.ann.commutator(m2.create)
        self.assertEqual(out, env_overlap * label_overlap)

    def test_commutator_create_with_ann_negative(self):
        env_overlap = 0.3 - 0.1j
        label_overlap = 0.5 + 0.0j
        m1, m2 = self.make_modes_for_commutator(
            env_overlap=env_overlap,
            label_overlap=label_overlap,
        )

        out = m1.create.commutator(m2.ann)
        self.assertEqual(out, -(env_overlap * label_overlap))

    def test_commutator_same_kind_is_zero(self):
        m1, m2 = self.make_modes_for_commutator(
            env_overlap=1.0 + 0.0j,
            label_overlap=1.0 + 0.0j,
        )

        self.assertEqual(m1.ann.commutator(m2.ann), 0.0 + 0.0j)
        self.assertEqual(m1.create.commutator(m2.create), 0.0 + 0.0j)

    def test_signature_and_approx_signature(self):
        env = FakeEnvelope("e1")
        base = FakeLabel("l1")
        label = FakeModeLabel(base=base, envelope=env)
        m = opmod.ModeOp(label=label)
        a = m.ann

        self.assertEqual(a.signature, ("lop", "a", m.signature))

        approx = a.approx_signature(decimals=9, ignore_global_phase=True)
        self.assertEqual(
            approx,
            (
                "lop",
                "a",
                m.approx_signature(decimals=9, ignore_global_phase=True),
            ),
        )


if __name__ == "__main__":
    unittest.main()
