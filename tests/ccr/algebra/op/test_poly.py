import unittest
from dataclasses import dataclass

from symop.ccr.algebra.op import OpPoly


@dataclass(frozen=True)
class DummyLadderOp:
    name: str
    is_creation: bool = False

    @property
    def signature(self):
        return ("lop", self.name, self.is_creation)

    def approx_signature(self, decimals=12, ignore_global_phase=False):
        return ("lop_approx", self.name, self.is_creation)

    def dagger(self):
        return DummyLadderOp(self.name, not self.is_creation)


def _canon(poly: OpPoly):
    """Canonical dictionary representation for comparison."""
    out = {}
    for t in poly.normalize().terms:
        out[t.signature] = out.get(t.signature, 0.0j) + t.coeff
    return out


class TestOpPoly(unittest.TestCase):
    def test_from_words_basic(self):
        a = DummyLadderOp("a")
        poly = OpPoly.from_words([[a]], coeffs=[2.0])

        self.assertEqual(len(poly.terms), 1)
        self.assertEqual(poly.terms[0].coeff, 2.0)

    def test_addition(self):
        a = DummyLadderOp("a")
        b = DummyLadderOp("b")

        p1 = OpPoly.from_words([[a]])
        p2 = OpPoly.from_words([[b]])

        s = p1 + p2

        self.assertEqual(len(s.terms), 2)

    def test_scalar_left_and_right(self):
        a = DummyLadderOp("a")
        p = OpPoly.from_words([[a]], coeffs=[3.0])

        left = 2.0 * p
        right = p * 2.0

        self.assertEqual(_canon(left), _canon(right))
        self.assertEqual(list(_canon(left).values())[0], 6.0)

    def test_multiplication(self):
        a = DummyLadderOp("a")
        b = DummyLadderOp("b")

        p1 = OpPoly.from_words([[a]], coeffs=[2.0])
        p2 = OpPoly.from_words([[b]], coeffs=[3.0])

        prod = (p1 * p2).normalize()

        terms = list(prod.terms)
        self.assertEqual(len(terms), 1)
        self.assertEqual(terms[0].coeff, 6.0)
        self.assertEqual(len(terms[0].ops), 2)

    def test_identity(self):
        a = DummyLadderOp("a")
        p = OpPoly.from_words([[a]], coeffs=[4.0])
        I = OpPoly.identity()

        left = (I * p).normalize()
        right = (p * I).normalize()

        self.assertEqual(_canon(left), _canon(p))
        self.assertEqual(_canon(right), _canon(p))

    def test_zero(self):
        z = OpPoly.zero()
        self.assertTrue(z.is_zero)

        a = DummyLadderOp("a")
        p = OpPoly.from_words([[a]])

        prod = (z * p).normalize()
        self.assertTrue(prod.is_zero)

    def test_adjoint_involution(self):
        a = DummyLadderOp("a")
        b = DummyLadderOp("b", is_creation=True)

        p = OpPoly.from_words([[a, b]], coeffs=[1.0 + 2.0j])

        self.assertEqual(
            _canon(p.normalize()),
            _canon(p.adjoint().adjoint().normalize()),
        )

    def test_adjoint_reverses_product(self):
        a = DummyLadderOp("a")
        b = DummyLadderOp("b", is_creation=True)

        p1 = OpPoly.from_words([[a]], coeffs=[2.0])
        p2 = OpPoly.from_words([[b]], coeffs=[3.0])

        left = (p1 * p2).adjoint().normalize()
        right = (p2.adjoint() * p1.adjoint()).normalize()

        self.assertEqual(_canon(left), _canon(right))

    def test_normalize_combines_terms(self):
        a = DummyLadderOp("a")

        p1 = OpPoly.from_words([[a]], coeffs=[2.0])
        p2 = OpPoly.from_words([[a]], coeffs=[3.0])

        s = (p1 + p2).normalize()

        self.assertEqual(len(s.terms), 1)
        self.assertEqual(s.terms[0].coeff, 5.0)

    def test_is_identity_property(self):
        I = OpPoly.identity()
        self.assertTrue(I.is_identity)

        a = DummyLadderOp("a")
        p = OpPoly.from_words([[a]])

        self.assertFalse(p.is_identity)
