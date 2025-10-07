from __future__ import annotations

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.ket.multiply import ket_multiply
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import KetTerm
from symop_proto.core.operators import ModeOp
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


class TestKetMultiply(ExtendedTestCase):
    def test_identity_times_identity(self):
        I = (KetTerm(2.0 + 0.5j, Monomial((), ())),)
        J = (KetTerm(-1.0j, Monomial((), ())),)
        out = ket_multiply(I, J)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].monomial.creators, ())
        self.assertEqual(out[0].monomial.annihilators, ())
        self.assertComplexAlmostEqual(out[0].coeff, (2.0 + 0.5j) * (-1.0j))

    def test_left_identity_and_right_identity_behave(self):
        m = make_mode("A")
        I = (KetTerm(1.0, Monomial((), ())),)
        A = (KetTerm(3.0, Monomial((m.create,), ())),)
        out1 = ket_multiply(I, A)
        out2 = ket_multiply(A, I)
        self.assertEqual(len(out1), 1)
        self.assertEqual(len(out2), 1)
        self.assertEqual(
            out1[0].monomial.signature, Monomial((m.create,), ()).signature
        )
        self.assertEqual(
            out2[0].monomial.signature, Monomial((m.create,), ()).signature
        )
        self.assertComplexAlmostEqual(out1[0].coeff, 3.0)
        self.assertComplexAlmostEqual(out2[0].coeff, 3.0)

    def test_a_then_adag_same_mode_produces_identity_and_number_term(self):
        m = make_mode("A")
        a = (KetTerm(1.0, Monomial((), (m.ann,))),)
        adag = (KetTerm(1.0, Monomial((m.create,), ())),)
        out = ket_multiply(a, adag)
        # Expect two terms: identity (coeff 1) and a_dag a (coeff 1)
        self.assertEqual(len(out), 2)
        sigs = {t.monomial.signature: t.coeff for t in out}
        id_sig = Monomial((), ()).signature
        num_sig = Monomial((m.create,), (m.ann,)).signature
        self.assertIn(id_sig, sigs)
        self.assertIn(num_sig, sigs)
        self.assertComplexAlmostEqual(sigs[id_sig], 1.0 + 0j)
        self.assertComplexAlmostEqual(sigs[num_sig], 1.0 + 0j)

    def test_annihilation_creation_orthogonal_paths_no_contraction(self):
        aA = make_mode("A", pol=PolarizationLabel.H())
        aB = make_mode("B", pol=PolarizationLabel.H())  # orthogonal by path
        a = (KetTerm(2.0, Monomial((), (aA.ann,))),)
        adagB = (KetTerm(5.0, Monomial((aB.create,), ())),)
        out = ket_multiply(a, adagB)
        self.assertEqual(len(out), 1)
        t = out[0]
        self.assertEqual(t.monomial.creators, (aB.create,))
        self.assertEqual(t.monomial.annihilators, (aA.ann,))
        self.assertComplexAlmostEqual(t.coeff, 10.0)

    def test_nonorthogonal_polarization_contraction_carries_complex_coeff(
        self,
    ):
        # <V|R> = -i / sqrt(2)
        aV = make_mode("A", pol=PolarizationLabel.V())
        aR = make_mode("A", pol=PolarizationLabel.R())
        a = (KetTerm(1.0, Monomial((), (aV.ann,))),)
        adag = (KetTerm(1.0, Monomial((aR.create,), ())),)
        out = ket_multiply(a, adag)
        self.assertEqual(len(out), 2)
        by_sig = {t.monomial.signature: t for t in out}
        id_sig = Monomial((), ()).signature
        arav_sig = Monomial((aR.create,), (aV.ann,)).signature
        self.assertIn(id_sig, by_sig)
        self.assertIn(arav_sig, by_sig)
        self.assertComplexAlmostEqual(by_sig[arav_sig].coeff, 1.0 + 0j)
        self.assertComplexAlmostEqual(by_sig[id_sig].coeff, (-1j) * (2**-0.5))

    def test_linearity_and_combination(self):
        m = make_mode("A")
        A = Monomial((m.create,), ())
        I = Monomial((), ())
        p = (KetTerm(1.0, I), KetTerm(1.0, A))
        q = (KetTerm(1.0, I), KetTerm(2.0, A))
        out = ket_multiply(p, q)
        self.assertEqual(len(out), 3)
        coeff_by_sig = {t.monomial.signature: t.coeff for t in out}
        self.assertComplexAlmostEqual(coeff_by_sig[I.signature], 1.0)
        self.assertComplexAlmostEqual(coeff_by_sig[A.signature], 3.0)
        A2_sig = Monomial((m.create, m.create), ()).signature
        self.assertComplexAlmostEqual(coeff_by_sig[A2_sig], 2.0)

    def test_combines_duplicate_terms(self):
        # (I + I) * I = 2 I
        I = Monomial((), ())
        p = (KetTerm(1.0, I), KetTerm(1.0, I))
        q = (KetTerm(1.0, I),)
        out = ket_multiply(p, q)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].monomial.signature, I.signature)
        self.assertComplexAlmostEqual(out[0].coeff, 2.0)
