from __future__ import annotations
import unittest
from math import sqrt

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel

from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import (
    Monomial,
)  # used only for signatures in sanity checks
from symop_proto.algebra.operator_polynomial import OpTerm, OpPoly


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


class TestOpTerm(ExtendedTestCase):

    def test_identity_and_scaled_and_adjoint(self):
        mA = make_mode("A")
        mB = make_mode("B")

        t_id = OpTerm.identity(2.0)
        self.assertEqual(t_id.ops, ())
        self.assertComplexAlmostEqual(t_id.coeff, 2.0 + 0j)

        t = OpTerm(ops=(mA.create, mB.ann), coeff=3.0 - 4.0j)
        t_star = t.adjoint()
        # reverse order and dagger each: (mB.ann) -> (mB.create), (mA.create) -> (mA.ann)
        self.assertEqual(t_star.ops, (mB.create, mA.ann))
        self.assertComplexAlmostEqual(t_star.coeff, 3.0 + 4.0j)

        t2 = t.scaled(-1j)
        self.assertEqual(t2.ops, t.ops)
        self.assertComplexAlmostEqual(t2.coeff, (-1j) * (3.0 - 4.0j))


class TestOpPolyConstruction(ExtendedTestCase):
    def test_zero_and_identity_flags(self):
        Z = OpPoly.zero()
        I = OpPoly.identity()
        self.assertTrue(Z.is_zero)
        self.assertFalse(I.is_zero)
        self.assertTrue(I.is_identity)
        # mixing identity with a non-empty word makes it non-identity
        m = make_mode("A")
        O = I + OpPoly.from_words([[m.ann]])
        self.assertFalse(O.is_identity)

    def test_from_words_and_scaling(self):
        m = make_mode("A")
        P = OpPoly.from_words([[m.ann], [m.create]], coeffs=[2.0, -3.0j])
        self.assertEqual(len(P.terms), 2)
        # scalar on right
        Q = P * (0.5 + 0.0j)
        # scalar on left
        R = (0.5 + 0.0j) * P
        for X in (Q, R):
            self.assertComplexAlmostEqual(
                X.terms[0].coeff, P.terms[0].coeff * 0.5
            )
            self.assertComplexAlmostEqual(
                X.terms[1].coeff, P.terms[1].coeff * 0.5
            )


class TestOpPolyAlgebra(ExtendedTestCase):
    def test_add_and_multiply_and_combine(self):
        m = make_mode("A")
        A = OpPoly.from_words([[m.ann], [m.create]], coeffs=[1.0, 1.0])
        B = OpPoly.from_words([[m.create]], coeffs=[2.0])
        # multiplication forms all concatenations
        M = A * B
        self.assertEqual(len(M.terms), 2)
        ops_sets = {tuple(t.ops) for t in M.terms}
        self.assertIn((m.ann, m.create), ops_sets)
        self.assertIn((m.create, m.create), ops_sets)
        # combine like terms merges identical words
        D = OpPoly.from_words(
            [[m.ann], [m.ann]], coeffs=[1.0, 2.5]
        ).combine_like_terms()
        self.assertEqual(len(D.terms), 1)
        self.assertEqual(D.terms[0].ops, (m.ann,))
        self.assertComplexAlmostEqual(D.terms[0].coeff, 3.5 + 0j)

    def test_adjoint_distributes_over_sum(self):
        m = make_mode("A")
        O = OpPoly.from_words(
            [[m.create, m.ann], [m.ann]], coeffs=[1.0 + 2.0j, -3.0]
        )
        Odag = O.adjoint()
        self.assertEqual(len(Odag.terms), 2)
        # check first term: ops reversed+daggered, coeff conjugated
        t0 = Odag.terms[0]
        self.assertEqual(t0.ops, (m.ann.dagger(), m.create.dagger()))
        self.assertComplexAlmostEqual(t0.coeff, (1.0 - 2.0j))


class TestConvenienceConstructors(ExtendedTestCase):
    def test_a_adag_n(self):
        m = make_mode("A")
        a = OpPoly.a(m)
        adag = OpPoly.adag(m)
        n = OpPoly.n(m)
        self.assertEqual(a.terms[0].ops, (m.ann,))
        self.assertEqual(adag.terms[0].ops, (m.create,))
        self.assertEqual(n.terms[0].ops, (m.create, m.ann))

    def test_q_p_and_Xtheta(self):
        m = make_mode("A")
        q = OpPoly.q(m).combine_like_terms()
        p = OpPoly.p(m).combine_like_terms()
        # q = (a + adag)/sqrt(2)
        sigs_q = {tuple(t.ops): t.coeff for t in q.terms}
        self.assertComplexAlmostEqual(sigs_q[(m.ann,)], 1.0 / sqrt(2))
        self.assertComplexAlmostEqual(sigs_q[(m.create,)], 1.0 / sqrt(2))

        # p = i adag/sqrt(2) - i a/sqrt(2)
        sigs_p = {tuple(t.ops): t.coeff for t in p.terms}
        self.assertComplexAlmostEqual(sigs_p[(m.create,)], 1j / sqrt(2))
        self.assertComplexAlmostEqual(sigs_p[(m.ann,)], -1j / sqrt(2))

        # X_theta(theta=0) = q
        X0 = OpPoly.X_theta(m, 0.0).combine_like_terms()
        sigs_X0 = {tuple(t.ops): t.coeff for t in X0.terms}
        self.assertComplexAlmostEqual(sigs_X0[(m.ann,)], 1.0 / sqrt(2))
        self.assertComplexAlmostEqual(sigs_X0[(m.create,)], 1.0 / sqrt(2))

    def test_generators_are_accepted(self):
        m = make_mode("A")
        words_gen = ((op,) for op in (m.ann, m.create))  # generator of words
        P = OpPoly.from_words(words_gen)  # coeffs default to 1
        self.assertEqual(
            {tuple(t.ops) for t in P.terms}, {(m.ann,), (m.create,)}
        )
        self.assertTrue(all(abs(t.coeff - 1.0) < 1e-14 for t in P.terms))


if __name__ == "__main__":
    unittest.main()
