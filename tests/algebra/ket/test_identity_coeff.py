from __future__ import annotations

from tests.utils.case import ExtendedTestCase

from symop_proto.algebra.ket.identity_coeff import (
    identity_coeff,
)
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import KetTerm
from symop_proto.core.operators import ModeOp
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel


def make_mode(path: str = "A") -> ModeOp:
    env = GaussianEnvelope(omega0=1.0, sigma=0.3, tau=0.0, phi0=0.0)
    label = ModeLabel(PathLabel(path), PolarizationLabel.H())
    return ModeOp(env=env, label=label)


class TestIdentityCoeff(ExtendedTestCase):
    def test_empty_returns_zero(self):
        self.assertComplexAlmostEqual(identity_coeff(()), 0.0 + 0.0j)

    def test_only_identity_term(self):
        coeff = 2.0 + 3.0j
        terms = (KetTerm(coeff, Monomial((), ())),)
        self.assertComplexAlmostEqual(identity_coeff(terms), coeff)

    def test_no_identity_among_terms_returns_zero(self):
        m = make_mode("A")
        terms = (KetTerm(1.0, Monomial((m.create,), ())),)
        self.assertComplexAlmostEqual(identity_coeff(terms), 0.0 + 0.0j)

    def test_identity_found_among_other_terms(self):
        m = make_mode("A")
        t1 = KetTerm(5.0, Monomial((m.create,), ()))  # non-identity
        t2 = KetTerm(1.0 + 0.5j, Monomial((), ()))  # identity
        t3 = KetTerm(-3.0, Monomial((), (m.ann,)))  # non-identity
        terms = (t1, t2, t3)
        self.assertComplexAlmostEqual(identity_coeff(terms), 1.0 + 0.5j)
