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
    """Minimal LadderOp stand-in for ket_inner tests.

    Supports:
      - .is_annihilation / .is_creation
      - .dagger()
      - .commutator(other) -> scalar
      - .mode.signature (used by Monomial.mode_ops in some contexts)
      - .signature / .approx_signature(...)
    """

    def __init__(
        self,
        op_id: str,
        mode: FakeMode,
        *,
        is_creation: bool,
        overlaps: dict[tuple[str, str], complex],
    ) -> None:
        self._op_id = op_id
        self.mode = mode
        self._is_creation = is_creation
        self._dagger = None
        self._overlaps = overlaps

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

    def commutator(self, other: "FakeLadderOp") -> complex:
        if self.is_annihilation and other.is_creation:
            key = (self.mode.signature[1], other.mode.signature[1])
            return self._overlaps.get(key, 0.0 + 0.0j)
        return 0.0 + 0.0j

    @property
    def signature(self):
        kind = "adag" if self.is_creation else "a"
        return ("lop", kind, self._op_id, self.mode.signature)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
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


from symop.ccr.algebra.ket.inner import ket_inner  # noqa: E402
from symop.core.monomial import Monomial  # noqa: E402
from symop.core.terms import KetTerm  # noqa: E402


class TestKetInner(unittest.TestCase):
    def make_ops(self):
        overlaps: dict[tuple[str, str], complex] = {
            ("A", "A"): 1.0 + 0.0j,
            ("B", "B"): 1.0 + 0.0j,
            ("A", "B"): 0.25 + 0.0j,
            ("B", "A"): 0.25 + 0.0j,
        }

        mode_a = FakeMode("A")
        mode_b = FakeMode("B")

        a = FakeLadderOp("a", mode_a, is_creation=False, overlaps=overlaps)
        adag = FakeLadderOp("adag", mode_a, is_creation=True, overlaps=overlaps)
        a.set_dagger(adag)
        adag.set_dagger(a)

        b = FakeLadderOp("b", mode_b, is_creation=False, overlaps=overlaps)
        bdag = FakeLadderOp("bdag", mode_b, is_creation=True, overlaps=overlaps)
        b.set_dagger(bdag)
        bdag.set_dagger(b)

        return {
            "a": a,
            "adag": adag,
            "b": b,
            "bdag": bdag,
            "overlaps": overlaps,
        }

    def test_vacuum_inner_vacuum_is_one(self):
        # |0> represented as coefficient * identity monomial
        ket0 = (KetTerm(1.0 + 0.0j, Monomial.identity()),)
        self.assertEqual(ket_inner(ket0, ket0), 1.0 + 0.0j)

    def test_single_photon_same_mode_inner_is_one(self):
        data = self.make_ops()
        adag = data["adag"]

        ket1 = (KetTerm(1.0 + 0.0j, Monomial(creators=(adag,), annihilators=())),)
        self.assertEqual(ket_inner(ket1, ket1), 1.0 + 0.0j)

    def test_single_photon_cross_mode_inner_is_overlap(self):
        data = self.make_ops()
        adag = data["adag"]
        bdag = data["bdag"]
        overlaps = data["overlaps"]

        ket_a = (KetTerm(1.0 + 0.0j, Monomial(creators=(adag,), annihilators=())),)
        ket_b = (KetTerm(1.0 + 0.0j, Monomial(creators=(bdag,), annihilators=())),)

        self.assertEqual(ket_inner(ket_a, ket_b), overlaps[("A", "B")])

    def test_eps_skips_small_input_terms(self):
        data = self.make_ops()
        adag = data["adag"]

        big = KetTerm(1.0 + 0.0j, Monomial(creators=(adag,), annihilators=()))
        small = KetTerm(1.0e-20 + 0.0j, Monomial.identity())

        ket = (big, small)
        out = ket_inner(ket, ket, eps=1.0e-12)

        # Should behave like <big|big> = 1, ignoring tiny identity term.
        self.assertEqual(out, 1.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
