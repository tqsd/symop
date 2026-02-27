from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import symop.ccr.algebra.ket.apply as mod


@dataclass(frozen=True)
class DummyKetTerm:
    tag: str


class TestKetApply(unittest.TestCase):
    def test_ket_apply_word_threads_eps_and_calls_from_word_and_multiply(
        self,
    ) -> None:
        ket = (DummyKetTerm("k1"), DummyKetTerm("k2"))
        word = ("op1", "op2")
        eps = 1e-9

        fake_word_terms = (DummyKetTerm("w1"),)

        def fake_from_word(*, ops, eps):
            self.assertEqual(tuple(ops), tuple(word))
            self.assertEqual(eps, 1e-9)
            return fake_word_terms

        def fake_multiply(left, right, *, eps: float = 1e-9):
            self.assertIs(left, fake_word_terms)
            self.assertIs(right, ket)
            self.assertEqual(eps, 1e-9)
            return (DummyKetTerm("out"),)

        with (
            patch.object(mod, "ket_from_word", side_effect=fake_from_word) as p_fw,
            patch.object(mod, "ket_multiply", side_effect=fake_multiply) as p_mul,
        ):
            out = mod.ket_apply_word(ket, word, eps=eps)

        self.assertEqual(p_fw.call_count, 1)
        self.assertEqual(p_mul.call_count, 1)
        self.assertEqual(out, (DummyKetTerm("out"),))

    def test_ket_apply_words_linear_skips_zero_scales_concatenates_and_combines(
        self,
    ) -> None:
        ket = (DummyKetTerm("k"),)
        eps = 1e-8

        def fake_apply_word(k, w, *, eps):
            self.assertIs(k, ket)
            self.assertEqual(eps, 1e-8)
            if tuple(w) == ("w1",):
                return (DummyKetTerm("a"), DummyKetTerm("b"))
            if tuple(w) == ("w2",):
                return (DummyKetTerm("c"),)
            if tuple(w) == ("w3",):
                return (DummyKetTerm("d"),)
            raise AssertionError("unexpected word")

        def fake_scale(term_tuple, c):
            return tuple(DummyKetTerm(t.tag + "*%s" % (c,)) for t in term_tuple)

        def fake_combine(term_tuple, *, eps):
            self.assertEqual(eps, 1e-8)
            return tuple(term_tuple)

        terms = (
            (2.0 + 0.0j, ("w1",)),
            (0.0 + 0.0j, ("w3",)),  # should be skipped
            (1.0 + 0.0j, ("w2",)),  # should not scale
        )

        with (
            patch.object(mod, "ket_apply_word", side_effect=fake_apply_word) as p_aw,
            patch.object(mod, "ket_scale", side_effect=fake_scale) as p_sc,
            patch.object(
                mod, "combine_like_terms_ket", side_effect=fake_combine
            ) as p_cb,
        ):
            out = mod.ket_apply_words_linear(ket, terms, eps=eps)

        # apply_word should be called only for nonzero coeffs (w1 and w2).
        self.assertEqual(p_aw.call_count, 2)

        # ket_scale should be called only for coeff != 1 (only for w1).
        self.assertEqual(p_sc.call_count, 1)

        # combine_like_terms_ket should be called once at the end.
        self.assertEqual(p_cb.call_count, 1)

        expected = (
            DummyKetTerm("a*(2+0j)"),
            DummyKetTerm("b*(2+0j)"),
            DummyKetTerm("c"),
        )
        self.assertEqual(out, expected)

    def test_ket_apply_words_linear_passes_words_through_iterables(
        self,
    ) -> None:
        ket = (DummyKetTerm("k"),)

        class WordIterable:
            def __init__(self, items):
                self._items = items

            def __iter__(self):
                return iter(self._items)

        w = WordIterable(["w1"])

        def fake_apply_word(k, word, *, eps):
            self.assertIs(k, ket)
            self.assertEqual(tuple(word), ("w1",))
            return (DummyKetTerm("x"),)

        def fake_combine(term_tuple, *, eps):
            return tuple(term_tuple)

        terms = ((1.0 + 0.0j, w),)

        with (
            patch.object(mod, "ket_apply_word", side_effect=fake_apply_word) as p_aw,
            patch.object(
                mod, "combine_like_terms_ket", side_effect=fake_combine
            ) as p_cb,
        ):
            out = mod.ket_apply_words_linear(ket, terms)

        self.assertEqual(p_aw.call_count, 1)
        self.assertEqual(p_cb.call_count, 1)
        self.assertEqual(out, (DummyKetTerm("x"),))
