from __future__ import annotations
import unittest
from typing import Tuple
from dataclasses import dataclass

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp, LadderOp

from symop_proto.algebra.operator_polynomial import OpPoly, OpTerm


def make_mode(
    path: str = "A",
    *,
    omega=1.0,
    sigma=0.3,
    tau=0.0,
    phi=0.0,
    pol: PolarizationLabel | None = None,
) -> ModeOp:
    pol = pol or PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


@dataclass(frozen=True)
class _StubTerm:
    ops: Tuple[LadderOp, ...]
    coeff: complex

    @property
    def signature(self) -> tuple:
        return (
            "stub",
            tuple(op.mode.label.signature for op in self.ops),
            len(self.ops),
        )

    def approx_signature(self, **env_kw) -> tuple:
        return self.signature


class TestOpPoly(ExtendedTestCase):
    def test_identity_and_zero_flags(self):
        I = OpPoly.identity(2.0)
        Z = OpPoly.zero()
        self.assertTrue(I.is_identity)
        self.assertFalse(I.is_zero)
        self.assertTrue(Z.is_zero)
        self.assertFalse(Z.is_identity)

    def test_from_words_builds_terms(self):
        mA = make_mode("A")
        mB = make_mode("B")
        O = OpPoly.from_words([[mA.create, mA.ann], [mB.ann]])
        self.assertEqual(len(O.terms), 2)
        self.assertEqual(O.terms[0].ops, (mA.create, mA.ann))
        self.assertEqual(O.terms[1].ops, (mB.ann,))

    def test_addition_concatenates_terms(self):
        mA = make_mode("A")
        O1 = OpPoly.from_words([[mA.create]])
        O2 = OpPoly.from_words([[mA.ann]])
        O = O1 + O2
        self.assertEqual(len(O.terms), 2)
        self.assertEqual(O.terms[0].ops, (mA.create,))
        self.assertEqual(O.terms[1].ops, (mA.ann,))

    def test_scalar_multiplication_scales_coeffs(self):
        mA = make_mode("A")
        O = OpPoly.from_words([[mA.create]])
        O2 = 3.0 * O
        self.assertAlmostEqual(O2.terms[0].coeff, 3.0)
        O3 = O * -2.0
        self.assertAlmostEqual(O3.terms[0].coeff, -2.0)

    def test_operator_multiplication_cartesian_and_order(self):
        mA = make_mode("A")
        mB = make_mode("B")
        O1 = OpPoly.from_words([[mA.create], [mA.ann]])
        O2 = OpPoly.from_words([[mB.create]])
        O = O1 * O2
        self.assertEqual(len(O.terms), 2)
        self.assertEqual(O.terms[0].ops, (mA.create, mB.create))
        self.assertEqual(O.terms[1].ops, (mA.ann, mB.create))

    def test_adjoint_reverses_order_and_daggers(self):
        mA = make_mode("A")
        O = OpPoly.from_words([[mA.create, mA.ann]])
        Od = O.adjoint()
        self.assertEqual(len(Od.terms), 1)
        self.assertEqual(Od.terms[0].ops, (mA.create, mA.ann))

    def test_q_p_n_builders(self):
        mA = make_mode("A")
        q = OpPoly.q(mA)
        p = OpPoly.p(mA)
        n = OpPoly.n(mA)
        self.assertEqual(len(q.terms), 2)
        self.assertEqual(len(p.terms), 2)
        self.assertEqual(len(n.terms), 1)
        self.assertEqual(n.terms[0].ops, (mA.create, mA.ann))

    def test_q2_p2_n2_helpers(self):
        mA = make_mode("A")
        q2_a = OpPoly.q2(mA)
        q = OpPoly.q(mA)
        q2_b = (q * q).combine_like_terms()
        self.assertEqual(
            {t.signature for t in q2_a.terms},
            {t.signature for t in q2_b.terms},
        )
        p2_a = OpPoly.p2(mA)
        p = OpPoly.p(mA)
        p2_b = (p * p).combine_like_terms()
        self.assertEqual(
            {t.signature for t in p2_a.terms},
            {t.signature for t in p2_b.terms},
        )
        n2_a = OpPoly.n2(mA)
        n = OpPoly.n(mA)
        n2_b = (n * n).combine_like_terms()
        self.assertEqual(
            {t.signature for t in n2_a.terms},
            {t.signature for t in n2_b.terms},
        )

    def test_combine_like_terms_exact(self):
        mA = make_mode("A")
        t1 = OpTerm((mA.create,), 2.0)
        t2 = OpTerm((mA.create,), 3.5)
        O = OpPoly((t1, t2)).combine_like_terms()
        self.assertEqual(len(O.terms), 1)
        self.assertEqual(O.terms[0].ops, (mA.create,))
        self.assertAlmostEqual(O.terms[0].coeff, 5.5)

    def test_combine_like_terms_plumbing_accepts_kw(self):
        mA = make_mode("A")
        O = OpPoly.from_words([[mA.create], [mA.create]])
        O2 = O.combine_like_terms(approx=False)
        self.assertEqual(len(O2.terms), 1)
        self.assertEqual(O2.terms[0].ops, (mA.create,))


if __name__ == "__main__":
    unittest.main()
