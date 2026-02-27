import unittest
from dataclasses import dataclass

from symop.ccr.algebra.op.combine import combine_like_terms


@dataclass(frozen=True)
class DummyTerm:
    ops: tuple
    coeff: complex

    @property
    def signature(self):
        return ("sig", self.ops)

    def approx_signature(self, **kw):
        return ("approx", self.ops)


def dummy_factory(ops, coeff):
    return DummyTerm(ops, coeff)


class TestCombineLikeTerms(unittest.TestCase):
    def test_exact_combination(self):
        t1 = DummyTerm(("a",), 1.0)
        t2 = DummyTerm(("a",), 2.0)
        t3 = DummyTerm(("b",), 3.0)

        out = combine_like_terms(
            (t1, t2, t3),
            term_factory=dummy_factory,
        )

        self.assertEqual(len(out), 2)

        coeffs = {t.ops: t.coeff for t in out}
        self.assertEqual(coeffs[("a",)], 3.0)
        self.assertEqual(coeffs[("b",)], 3.0)

    def test_zero_bucket_removed(self):
        t1 = DummyTerm(("a",), 1.0)
        t2 = DummyTerm(("a",), -1.0)

        out = combine_like_terms(
            (t1, t2),
            term_factory=dummy_factory,
        )

        self.assertEqual(len(out), 0)

    def test_approx_combination(self):
        class ApproxTerm(DummyTerm):
            def approx_signature(self, **kw):
                return ("approx", "same")

        t1 = ApproxTerm(("a",), 1.0)
        t2 = ApproxTerm(("b",), 2.0)

        out = combine_like_terms(
            (t1, t2),
            approx=True,
            term_factory=dummy_factory,
        )

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].coeff, 3.0)

    def test_factory_used(self):
        t = DummyTerm(("x",), 5.0)

        out = combine_like_terms(
            (t,),
            term_factory=dummy_factory,
        )

        self.assertIsInstance(out[0], DummyTerm)
