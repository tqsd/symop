from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import symop.ccr.algebra.density.apply_left as mod


@dataclass(frozen=True)
class DummyDensityTerm:
    coeff: complex
    left: object
    right: object


@dataclass(frozen=True)
class DummyKetTerm:
    coeff: complex
    monomial: object


@dataclass(frozen=True)
class DummyDensityTermOut:
    coeff: complex
    left: object
    right: object


class TestApplyLeft(unittest.TestCase):
    def test_apply_left_expands_each_term_and_combines(self) -> None:
        word = ("op1", "op2")

        t1 = DummyDensityTerm(coeff=2.0 + 0.0j, left="L1", right="R1")
        t2 = DummyDensityTerm(coeff=3.0 + 0.0j, left="L2", right="R2")
        terms = (t1, t2)

        expansions = {
            "L1": (
                DummyKetTerm(coeff=10.0 + 0.0j, monomial="NL1a"),
                DummyKetTerm(coeff=20.0 + 0.0j, monomial="NL1b"),
            ),
            "L2": (
                DummyKetTerm(coeff=30.0 + 0.0j, monomial="NL2a"),
                DummyKetTerm(coeff=40.0 + 0.0j, monomial="NL2b"),
            ),
        }

        def fake_expand(w, left):
            self.assertEqual(tuple(w), tuple(word))
            return expansions[left]

        def fake_combine(out_terms):
            # Keep as-is; combine is tested separately.
            return tuple(out_terms)

        def fake_density_term_ctor(*, coeff, left, right):
            return DummyDensityTermOut(coeff=coeff, left=left, right=right)

        with (
            patch.object(
                mod, "expand_word_times_monomial", side_effect=fake_expand
            ) as p_expand,
            patch.object(
                mod, "combine_like_terms_density", side_effect=fake_combine
            ) as p_combine,
            patch.object(mod, "DensityTerm", side_effect=fake_density_term_ctor),
        ):
            out = mod.apply_left(terms, word)

        self.assertEqual(p_expand.call_count, 2)
        self.assertEqual(p_combine.call_count, 1)

        expected = (
            DummyDensityTermOut(
                coeff=(2.0 + 0.0j) * (10.0 + 0.0j), left="NL1a", right="R1"
            ),
            DummyDensityTermOut(
                coeff=(2.0 + 0.0j) * (20.0 + 0.0j), left="NL1b", right="R1"
            ),
            DummyDensityTermOut(
                coeff=(3.0 + 0.0j) * (30.0 + 0.0j), left="NL2a", right="R2"
            ),
            DummyDensityTermOut(
                coeff=(3.0 + 0.0j) * (40.0 + 0.0j), left="NL2b", right="R2"
            ),
        )
        self.assertEqual(out, expected)
