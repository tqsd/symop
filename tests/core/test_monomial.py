import unittest


class FakeMode:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def signature(self):
        return ("mode", self._name)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        return ("mode_approx", self._name, decimals, ignore_global_phase)


class FakeLadderOp:
    """Minimal stand-in for LadderOpProto used by Monomial.

    It supports:
      - .mode.signature
      - .signature
      - .approx_signature(decimals=..., ignore_global_phase=...)
      - .dagger()
    """

    def __init__(self, op_id: str, mode: FakeMode) -> None:
        self._op_id = op_id
        self.mode = mode
        self._dagger = None
        self.last_approx_args = None

    def set_dagger(self, other: "FakeLadderOp") -> None:
        self._dagger = other

    def dagger(self) -> "FakeLadderOp":
        if self._dagger is None:
            raise RuntimeError("Dagger not configured for FakeLadderOp")
        return self._dagger

    @property
    def signature(self):
        return ("lop", self._op_id, self.mode.signature)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        self.last_approx_args = (decimals, ignore_global_phase)
        return (
            "lop_approx",
            self._op_id,
            self.mode.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
        )


# Import after fakes so test failures are easier to interpret if import-time
# issues occur in the module under test.
from symop.core.monomial import Monomial  # noqa: E402


class TestMonomial(unittest.TestCase):
    def make_ops(self):
        mode_a = FakeMode("A")
        mode_b = FakeMode("B")
        mode_c = FakeMode("C")

        a1 = FakeLadderOp("a1", mode_a)
        a1_dag = FakeLadderOp("a1_dag", mode_a)
        a1.set_dagger(a1_dag)
        a1_dag.set_dagger(a1)

        b1 = FakeLadderOp("b1", mode_b)
        b1_dag = FakeLadderOp("b1_dag", mode_b)
        b1.set_dagger(b1_dag)
        b1_dag.set_dagger(b1)

        c1 = FakeLadderOp("c1", mode_c)
        c1_dag = FakeLadderOp("c1_dag", mode_c)
        c1.set_dagger(c1_dag)
        c1_dag.set_dagger(c1)

        return {
            "modes": (mode_a, mode_b, mode_c),
            "ops": (a1, a1_dag, b1, b1_dag, c1, c1_dag),
            "a1": a1,
            "a1_dag": a1_dag,
            "b1": b1,
            "b1_dag": b1_dag,
            "c1": c1,
            "c1_dag": c1_dag,
        }

    def test_identity_flags(self):
        m = Monomial()
        self.assertTrue(m.is_identity)
        self.assertFalse(m.is_creator_only)
        self.assertFalse(m.is_annihilator_only)
        self.assertFalse(m.has_creators)
        self.assertFalse(m.has_annihilators)
        self.assertEqual(m.creators, ())
        self.assertEqual(m.annihilators, ())

    def test_post_init_coerces_iterables_to_tuples(self):
        data = self.make_ops()
        a1 = data["a1"]
        b1 = data["b1"]
        m = Monomial(creators=[a1], annihilators=[b1])
        self.assertIsInstance(m.creators, tuple)
        self.assertIsInstance(m.annihilators, tuple)
        self.assertEqual(m.creators, (a1,))
        self.assertEqual(m.annihilators, (b1,))

    def test_mode_ops_unique_and_ordered_by_first_appearance(self):
        data = self.make_ops()
        a1 = data["a1"]
        a1_dag = data["a1_dag"]
        b1 = data["b1"]
        c1 = data["c1"]

        # Order of first appearance:
        # creators: A then B
        # annihilators: A again (should not duplicate), then C
        m = Monomial(creators=(a1_dag, b1), annihilators=(a1, c1))
        mode_ops = m.mode_ops

        self.assertEqual(len(mode_ops), 3)
        self.assertEqual(
            [mo.signature for mo in mode_ops],
            [
                ("mode", "A"),
                ("mode", "B"),
                ("mode", "C"),
            ],
        )

    def test_adjoint_swaps_sides_and_daggers_each_operator(self):
        data = self.make_ops()
        a1 = data["a1"]
        a1_dag = data["a1_dag"]
        b1 = data["b1"]
        b1_dag = data["b1_dag"]
        c1 = data["c1"]
        c1_dag = data["c1_dag"]

        m = Monomial(creators=(a1_dag, b1_dag), annihilators=(c1, a1))
        m_adj = m.adjoint()

        # adjoint() does:
        # dag_creators = dagger(annihilators) in the same order as annihilators
        # dag_annihilators = dagger(creators) in the same order as creators
        self.assertEqual(m_adj.creators, (c1_dag, a1_dag))
        self.assertEqual(m_adj.annihilators, (a1, b1))

        # Adjoint of adjoint should return the original (with these fakes).
        m_adj_adj = m_adj.adjoint()
        self.assertEqual(m_adj_adj.creators, m.creators)
        self.assertEqual(m_adj_adj.annihilators, m.annihilators)

    def test_signature_is_order_insensitive_within_creators_and_annihilators(
        self,
    ):
        data = self.make_ops()
        a1 = data["a1"]
        b1 = data["b1"]
        c1 = data["c1"]
        a1_dag = data["a1_dag"]
        b1_dag = data["b1_dag"]
        c1_dag = data["c1_dag"]

        m1 = Monomial(creators=(a1_dag, b1_dag, c1_dag), annihilators=(a1, b1, c1))
        m2 = Monomial(creators=(c1_dag, a1_dag, b1_dag), annihilators=(b1, c1, a1))

        self.assertEqual(m1.signature, m2.signature)

        sig = m1.signature
        self.assertEqual(sig[0], "cre")
        self.assertEqual(sig[2], "ann")
        self.assertIsInstance(sig[1], tuple)
        self.assertIsInstance(sig[3], tuple)

    def test_approx_signature_passes_parameters_to_ops(self):
        data = self.make_ops()
        a1 = data["a1"]
        b1 = data["b1"]
        a1_dag = data["a1_dag"]

        m = Monomial(creators=(a1_dag,), annihilators=(a1, b1))
        out = m.approx_signature(decimals=7, ignore_global_phase=True)

        self.assertEqual(out[0], "cre")
        self.assertEqual(out[2], "ann")

        # Ensure args were propagated to each op used.
        self.assertEqual(a1_dag.last_approx_args, (7, True))
        self.assertEqual(a1.last_approx_args, (7, True))
        self.assertEqual(b1.last_approx_args, (7, True))

    def test_creator_annihilator_identity_flags(self):
        data = self.make_ops()
        a1 = data["a1"]
        b1 = data["b1"]
        a1_dag = data["a1_dag"]

        only_creators = Monomial(creators=(a1_dag,), annihilators=())
        self.assertTrue(only_creators.is_creator_only)
        self.assertFalse(only_creators.is_annihilator_only)
        self.assertFalse(only_creators.is_identity)
        self.assertTrue(only_creators.has_creators)
        self.assertFalse(only_creators.has_annihilators)

        only_ann = Monomial(creators=(), annihilators=(a1, b1))
        self.assertFalse(only_ann.is_creator_only)
        self.assertTrue(only_ann.is_annihilator_only)
        self.assertFalse(only_ann.is_identity)
        self.assertFalse(only_ann.has_creators)
        self.assertTrue(only_ann.has_annihilators)

        mixed = Monomial(creators=(a1_dag,), annihilators=(a1,))
        self.assertFalse(mixed.is_creator_only)
        self.assertFalse(mixed.is_annihilator_only)
        self.assertFalse(mixed.is_identity)
        self.assertTrue(mixed.has_creators)
        self.assertTrue(mixed.has_annihilators)


if __name__ == "__main__":
    unittest.main()
