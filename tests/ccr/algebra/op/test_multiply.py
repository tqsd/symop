import unittest
from dataclasses import dataclass

from symop.ccr.algebra.op.multiply import multiply


@dataclass(frozen=True)
class DummyTerm:
    ops: tuple
    coeff: complex


def dummy_factory(ops, coeff):
    return DummyTerm(ops, coeff)


class TestMultiply(unittest.TestCase):
    def test_basic_cartesian_product(self):
        a = (
            DummyTerm(("a",), 2.0),
            DummyTerm(("b",), 3.0),
        )
        b = (DummyTerm(("c",), 5.0),)

        out = multiply(a, b, term_factory=dummy_factory)

        self.assertEqual(len(out), 2)

        self.assertEqual(out[0].ops, ("a", "c"))
        self.assertEqual(out[0].coeff, 10.0)

        self.assertEqual(out[1].ops, ("b", "c"))
        self.assertEqual(out[1].coeff, 15.0)

    def test_empty_left(self):
        a = ()
        b = (DummyTerm(("x",), 1.0),)

        out = multiply(a, b, term_factory=dummy_factory)

        self.assertEqual(len(out), 0)

    def test_empty_right(self):
        a = (DummyTerm(("x",), 1.0),)
        b = ()

        out = multiply(a, b, term_factory=dummy_factory)

        self.assertEqual(len(out), 0)

    def test_row_major_order(self):
        a = (
            DummyTerm(("a",), 1.0),
            DummyTerm(("b",), 1.0),
        )
        b = (
            DummyTerm(("c",), 1.0),
            DummyTerm(("d",), 1.0),
        )

        out = multiply(a, b, term_factory=dummy_factory)

        words = [t.ops for t in out]

        self.assertEqual(
            words,
            [
                ("a", "c"),
                ("a", "d"),
                ("b", "c"),
                ("b", "d"),
            ],
        )
