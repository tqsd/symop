import unittest
from dataclasses import dataclass

from symop.ccr.algebra.op.from_words import from_words


@dataclass(frozen=True)
class DummyLadderOp:
    """Minimal ladder-operator stub for testing."""

    name: str


@dataclass(frozen=True)
class DummyTerm:
    """Simple term implementation used as factory output."""

    ops: tuple
    coeff: complex


class TestFromWords(unittest.TestCase):
    def test_unit_coefficients(self):
        w1 = [DummyLadderOp("a")]
        w2 = [DummyLadderOp("b"), DummyLadderOp("c")]

        terms = from_words(
            words=[w1, w2],
            term_factory=DummyTerm,
        )

        self.assertEqual(len(terms), 2)
        self.assertEqual(terms[0].coeff, 1.0)
        self.assertEqual(terms[1].coeff, 1.0)

        self.assertIsInstance(terms[0].ops, tuple)
        self.assertEqual(len(terms[1].ops), 2)

    def test_explicit_coefficients(self):
        w = [[DummyLadderOp("a")]]
        terms = from_words(
            words=w,
            coeffs=[2.5],
            term_factory=DummyTerm,
        )

        self.assertEqual(terms[0].coeff, 2.5)

    def test_length_mismatch_raises(self):
        w = [[DummyLadderOp("a")], [DummyLadderOp("b")]]

        with self.assertRaises(ValueError):
            from_words(
                words=w,
                coeffs=[1.0],
                term_factory=DummyTerm,
            )

    def test_factory_is_used(self):
        w = [[DummyLadderOp("x")]]

        terms = from_words(
            words=w,
            term_factory=DummyTerm,
        )

        self.assertIsInstance(terms[0], DummyTerm)
