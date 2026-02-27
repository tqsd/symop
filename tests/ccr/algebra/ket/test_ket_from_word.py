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
    """Minimal stand-in for LadderOpProto used by ket_from_word.

    Supports:
      - .is_annihilation
      - .is_creation
      - .mode.signature
      - .signature / .approx_signature(...)
      - .dagger()
      - .commutator(other) -> complex scalar
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

    def commutator(self, other: "FakeLadderOp") -> complex:
        # CCR scalar: [a_i, a_j^dag] = <m_i|m_j>, otherwise 0.
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


from symop.ccr.algebra.ket.from_word import ket_from_word  # noqa: E402
from symop.core.monomial import Monomial  # noqa: E402


class TestKetFromWord(unittest.TestCase):
    def make_mode_ops(self):
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
            "overlaps": overlaps,
            "mode_a": mode_a,
            "mode_b": mode_b,
            "a": a,
            "adag": adag,
            "b": b,
            "bdag": bdag,
        }

    def terms_to_sig_coeff(self, terms):
        out: dict[tuple, complex] = {}
        for t in terms:
            out[t.monomial.signature] = (
                out.get(t.monomial.signature, 0.0 + 0.0j) + t.coeff
            )
        return out

    def test_empty_word_is_identity(self):
        terms = ket_from_word(ops=[])
        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0].coeff, 1.0 + 0.0j)
        self.assertTrue(terms[0].monomial.is_identity)

    def test_a_adag_expands_to_adag_a_plus_identity(self):
        data = self.make_mode_ops()
        a = data["a"]
        adag = data["adag"]

        terms = ket_from_word(ops=[a, adag])
        got = self.terms_to_sig_coeff(terms)

        m_adag_a = Monomial(creators=(adag,), annihilators=(a,))
        m_id = Monomial.identity()

        self.assertIn(m_adag_a.signature, got)
        self.assertIn(m_id.signature, got)
        self.assertEqual(got[m_adag_a.signature], 1.0 + 0.0j)
        self.assertEqual(got[m_id.signature], 1.0 + 0.0j)

    def test_nonorthogonal_commutator_inserts_overlap_scalar(self):
        data = self.make_mode_ops()
        a = data["a"]
        bdag = data["bdag"]
        overlaps = data["overlaps"]

        terms = ket_from_word(ops=[a, bdag])
        got = self.terms_to_sig_coeff(terms)

        m_bdag_a = Monomial(creators=(bdag,), annihilators=(a,))
        m_id = Monomial.identity()

        self.assertEqual(got[m_bdag_a.signature], 1.0 + 0.0j)
        self.assertEqual(got[m_id.signature], overlaps[("A", "B")])

    def test_a_adag_adag_gives_adag_adag_a_plus_2_adag(self):
        data = self.make_mode_ops()
        a = data["a"]
        adag = data["adag"]

        terms = ket_from_word(ops=[a, adag, adag])
        got = self.terms_to_sig_coeff(terms)

        m_2adag_a = Monomial(creators=(adag, adag), annihilators=(a,))
        m_adag = Monomial(creators=(adag,), annihilators=())

        self.assertEqual(got[m_2adag_a.signature], 1.0 + 0.0j)
        self.assertEqual(got[m_adag.signature], 2.0 + 0.0j)

    def test_eps_drops_small_terms(self):
        data = self.make_mode_ops()
        a = data["a"]
        bdag = data["bdag"]

        # Overlap is 0.25, so with eps > 0.25 it should drop the identity term.
        terms = ket_from_word(ops=[a, bdag], eps=0.3)
        got = self.terms_to_sig_coeff(terms)

        m_bdag_a = Monomial(creators=(bdag,), annihilators=(a,))
        m_id = Monomial.identity()

        self.assertIn(m_bdag_a.signature, got)
        self.assertNotIn(m_id.signature, got)


if __name__ == "__main__":
    unittest.main()
