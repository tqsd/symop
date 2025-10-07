from __future__ import annotations

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.ket.from_word import ket_from_word
from symop_proto.core.monomial import Monomial
from symop_proto.core.operators import ModeOp
from symop_proto.core.terms import KetTerm
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel


def make_mode(
    path: str = "A",
    *,
    omega: float = 1.0,
    sigma: float = 0.3,
    tau: float = 0.0,
    phi: float = 0.0,
    pol: PolarizationLabel | None = None,
) -> ModeOp:
    pol = pol or PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


class TestKetFromWord(ExtendedTestCase):
    def test_empty_word_gives_identity(self):
        terms = ket_from_word(ops=())
        self.assertEqual(len(terms), 1)
        t = terms[0]
        self.assertIsInstance(t, KetTerm)
        self.assertComplexAlmostEqual(t.coeff, 1.0 + 0j)
        self.assertEqual(t.monomial.creators, ())
        self.assertEqual(t.monomial.annihilators, ())

    def test_single_creation(self):
        m = make_mode("A")
        terms = ket_from_word(ops=(m.create,))
        self.assertEqual(len(terms), 1)
        t = terms[0]
        self.assertEqual(t.monomial.creators, (m.create,))
        self.assertEqual(t.monomial.annihilators, ())
        self.assertComplexAlmostEqual(t.coeff, 1.0 + 0j)

    def test_single_annihilation(self):
        m = make_mode("A")
        terms = ket_from_word(ops=(m.ann,))
        self.assertEqual(len(terms), 1)
        t = terms[0]
        self.assertEqual(t.monomial.creators, ())
        self.assertEqual(t.monomial.annihilators, (m.ann,))
        self.assertComplexAlmostEqual(t.coeff, 1.0 + 0j)

    def test_two_ops_same_mode_a_then_adag(self):
        # Expect two terms: identity (coeff 1) and a_dag a (coeff 1)
        m = make_mode("A")
        terms = ket_from_word(ops=(m.ann, m.create))
        self.assertEqual(len(terms), 2)
        # sorted by monomial signature  identity first
        id_term = terms[0]
        aa_term = terms[1]
        # identity
        self.assertEqual(id_term.monomial.creators, ())
        self.assertEqual(id_term.monomial.annihilators, ())
        self.assertComplexAlmostEqual(id_term.coeff, 1.0 + 0j)
        # a_dag a
        self.assertEqual(aa_term.monomial.creators, (m.create,))
        self.assertEqual(aa_term.monomial.annihilators, (m.ann,))
        self.assertComplexAlmostEqual(aa_term.coeff, 1.0 + 0j)

    def test_two_ops_same_mode_adag_then_a(self):
        # Implementation returns the same normal-ordered polynomial as above
        m = make_mode("A")
        terms = ket_from_word(ops=(m.create, m.ann))
        self.assertEqual(len(terms), 2)
        # identity + a_dag a
        sigs = [t.monomial.signature for t in terms]
        id_sig = Monomial((), ()).signature
        aa_sig = Monomial((m.create,), (m.ann,)).signature
        self.assertEqual(set(sigs), {id_sig, aa_sig})
        # check coefficients are both +1
        coeff_by_sig = {t.monomial.signature: t.coeff for t in terms}
        self.assertComplexAlmostEqual(coeff_by_sig[id_sig], 1.0 + 0j)
        self.assertComplexAlmostEqual(coeff_by_sig[aa_sig], 1.0 + 0j)

    def test_orthogonal_paths_no_contraction(self):
        a = make_mode("A", pol=PolarizationLabel.H())
        b = make_mode("B", pol=PolarizationLabel.H())  # path-orthogonal
        terms = ket_from_word(ops=(a.ann, b.create))
        # Only pass-through term: b_dag with a to the right (no identity)
        self.assertEqual(len(terms), 1)
        t = terms[0]
        self.assertEqual(t.monomial.creators, (b.create,))
        self.assertEqual(t.monomial.annihilators, (a.ann,))
        self.assertComplexAlmostEqual(t.coeff, 1.0 + 0j)

    def test_nonorthogonal_polarization_contraction_complex_coeff(self):
        # <V|R> = -i / sqrt(2) -> identity term picks up that factor
        aV = make_mode("A", pol=PolarizationLabel.V())
        aR = make_mode("A", pol=PolarizationLabel.R())
        terms = ket_from_word(ops=(aV.ann, aR.create))
        self.assertEqual(len(terms), 2)
        id_sig = Monomial((), ()).signature
        arav_sig = Monomial((aR.create,), (aV.ann,)).signature
        by_sig = {t.monomial.signature: t for t in terms}
        self.assertIn(id_sig, by_sig)
        self.assertIn(arav_sig, by_sig)
        self.assertComplexAlmostEqual(by_sig[arav_sig].coeff, 1.0 + 0j)

        expected = (-1j) * (2**-0.5)
        self.assertComplexAlmostEqual(by_sig[id_sig].coeff, expected)

    def test_sorted_by_signature_identity_first(self):
        a = make_mode("A")
        terms = ket_from_word(ops=(a.ann, a.create))
        self.assertGreaterEqual(
            terms[1].monomial.signature, terms[0].monomial.signature
        )
        # identity first
        self.assertEqual(
            terms[0].monomial.signature, Monomial((), ()).signature
        )
