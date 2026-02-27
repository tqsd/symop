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
    """Minimal LadderOp stand-in for ket_from_ops tests."""

    def __init__(self, op_id: str, mode: FakeMode, *, is_creation: bool) -> None:
        self._op_id = op_id
        self.mode = mode
        self._is_creation = is_creation
        self._dagger = None
        self.last_approx_args = None

    @property
    def is_creation(self) -> bool:
        return self._is_creation

    @property
    def is_annihilation(self) -> bool:
        return not self._is_creation

    def set_dagger(self, other: "FakeLadderOp") -> None:
        self._dagger = other

    def dagger(self) -> "FakeLadderOp":
        if self._dagger is None:
            raise RuntimeError("Dagger not configured for FakeLadderOp")
        return self._dagger

    @property
    def signature(self):
        kind = "adag" if self.is_creation else "a"
        return ("lop", kind, self._op_id, self.mode.signature)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        self.last_approx_args = (decimals, ignore_global_phase)
        kind = "adag" if self.is_creation else "a"
        return (
            "lop_approx",
            kind,
            self._op_id,
            self.mode.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
        )


from symop.ccr.algebra.ket.from_ops import ket_from_ops  # noqa: E402
from symop.core.monomial import Monomial  # noqa: E402


class TestKetFromOps(unittest.TestCase):
    def make_ops(self):
        mode_a = FakeMode("A")
        mode_b = FakeMode("B")

        a = FakeLadderOp("a", mode_a, is_creation=False)
        adag = FakeLadderOp("adag", mode_a, is_creation=True)
        a.set_dagger(adag)
        adag.set_dagger(a)

        b = FakeLadderOp("b", mode_b, is_creation=False)
        bdag = FakeLadderOp("bdag", mode_b, is_creation=True)
        b.set_dagger(bdag)
        bdag.set_dagger(b)

        return {"a": a, "adag": adag, "b": b, "bdag": bdag}

    def test_builds_single_term_with_expected_monomial(self):
        data = self.make_ops()
        adag = data["adag"]
        bdag = data["bdag"]
        a = data["a"]
        b = data["b"]

        terms = ket_from_ops(
            creators=(adag, bdag),
            annihilators=(a, b),
            coeff=2.5 - 1.0j,
        )
        self.assertEqual(len(terms), 1)
        t0 = terms[0]
        self.assertEqual(t0.coeff, 2.5 - 1.0j)

        expected = Monomial(creators=(adag, bdag), annihilators=(a, b))
        self.assertEqual(t0.monomial.signature, expected.signature)

    def test_rejects_non_creation_in_creators(self):
        data = self.make_ops()
        a = data["a"]
        bdag = data["bdag"]
        with self.assertRaises(ValueError):
            ket_from_ops(creators=(a, bdag))

    def test_rejects_non_annihilation_in_annihilators(self):
        data = self.make_ops()
        adag = data["adag"]
        b = data["b"]
        with self.assertRaises(ValueError):
            ket_from_ops(annihilators=(adag, b))


if __name__ == "__main__":
    unittest.main()
