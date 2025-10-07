from __future__ import annotations

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.ket.scale import ket_scale
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


class TestKetScale(ExtendedTestCase):
    def test_empty_returns_empty(self):
        out = ket_scale((), 3.0)
        self.assertEqual(out, ())

    def test_real_scaling(self):
        m = make_mode("A")
        mono = Monomial((m.create,), ())
        t = KetTerm(2.0 + 1.0j, mono)
        out = ket_scale((t,), 3.0)  # real scale
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, (2.0 + 1.0j) * 3.0)
        self.assertEqual(out[0].monomial.signature, mono.signature)

    def test_complex_scaling(self):
        m = make_mode("A")
        mono = Monomial((m.create,), ())
        t1 = KetTerm(1.0, mono)
        t2 = KetTerm(-0.5j, mono)
        out = ket_scale((t1, t2), 1.0 + 2.0j)
        self.assertEqual(len(out), 2)
        self.assertComplexAlmostEqual(
            out[0].coeff, (1.0 + 0.0j) * (1.0 + 2.0j)
        )
        self.assertComplexAlmostEqual(out[1].coeff, (-0.5j) * (1.0 + 2.0j))
        # order preserved
        self.assertEqual(out[0].monomial.signature, t1.monomial.signature)
        self.assertEqual(out[1].monomial.signature, t2.monomial.signature)

    def test_zero_scaling_keeps_term_but_zero_coeff(self):
        m = make_mode("A")
        mono = Monomial((m.create,), ())
        t = KetTerm(3.0 - 4.0j, mono)
        out = ket_scale((t,), 0.0)
        self.assertEqual(len(out), 1)
        self.assertComplexAlmostEqual(out[0].coeff, 0.0 + 0.0j)
        # monomial unchanged
        self.assertEqual(out[0].monomial.signature, mono.signature)

    def test_original_terms_unchanged_and_monomial_reference_preserved(self):
        m = make_mode("A")
        mono = Monomial((m.create,), ())
        t = KetTerm(1.0 + 2.0j, mono)
        out = ket_scale((t,), 2.0)
        # input term unchanged
        self.assertComplexAlmostEqual(t.coeff, 1.0 + 2.0j)
        # returned term is a new KetTerm object
        self.assertIsNot(out[0], t)
        # but the monomial object should be the same reference
        self.assertIs(out[0].monomial, t.monomial)
