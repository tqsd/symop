import unittest

from symop.ccr.algebra.density.partial_trace import (
    _split_monomial_by_modes,
    _to_mode_signatures,
    density_partial_trace,
)
from symop.core.monomial import Monomial
from symop.core.terms.density_term import DensityTerm

from tests.ccr.support.fakes import make_mode, set_hermitian_overlap


class TestToModeSignatures(unittest.TestCase):
    def test_accepts_mode_ladder_monomial_and_signature(self) -> None:
        mode = make_mode()
        sig = mode.signature
        monomial = Monomial(creators=(mode.cre,), annihilators=())

        result = _to_mode_signatures((mode, mode.ann, monomial, sig))

        self.assertEqual(result, {sig})

    def test_accepts_mixed_modes(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        result = _to_mode_signatures((mode_a, mode_b.ann))

        self.assertEqual(result, {mode_a.signature, mode_b.signature})


class TestSplitMonomialByModes(unittest.TestCase):
    def test_splits_kept_and_traced_parts(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        monomial = Monomial(
            creators=(mode_a.cre, mode_b.cre),
            annihilators=(mode_b.ann, mode_a.ann),
        )

        kept, traced = _split_monomial_by_modes(monomial, {mode_b.signature})

        self.assertEqual(
            kept,
            Monomial(
                creators=(mode_a.cre,),
                annihilators=(mode_a.ann,),
            ),
        )
        self.assertEqual(
            traced,
            Monomial(
                creators=(mode_b.cre,),
                annihilators=(mode_b.ann,),
            ),
        )


class TestDensityPartialTrace(unittest.TestCase):
    def test_empty_input_returns_empty_tuple(self) -> None:
        self.assertEqual(density_partial_trace((), ()), ())

    def test_tracing_nothing_returns_same_density(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )

        result = density_partial_trace(terms, ())

        self.assertEqual(result, terms)

    def test_trace_out_single_mode_identity_density(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )

        result = density_partial_trace(terms, (mode,))

        expected = (
            DensityTerm(
                coeff=2.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)

    def test_trace_out_orthogonal_traced_parts_drops_term(self) -> None:
        mode_a = make_mode(path="a")
        mode_b = make_mode(path="b")

        terms = (
            DensityTerm(
                coeff=1.0,
                left=Monomial(creators=(mode_a.cre,), annihilators=()),
                right=Monomial(creators=(mode_b.cre,), annihilators=()),
            ),
        )

        result = density_partial_trace(terms, (mode_a, mode_b))

        self.assertEqual(result, ())

    def test_partial_trace_keeps_untraced_modes(self) -> None:
        mode_keep = make_mode(path="keep")
        mode_trace = make_mode(path="trace")

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(
                    creators=(mode_keep.cre, mode_trace.cre),
                    annihilators=(),
                ),
                right=Monomial(
                    creators=(mode_keep.cre, mode_trace.cre),
                    annihilators=(),
                ),
            ),
        )

        result = density_partial_trace(terms, (mode_trace,))

        expected = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode_keep.cre,), annihilators=()),
                right=Monomial(creators=(mode_keep.cre,), annihilators=()),
            ),
        )
        self.assertEqual(result, expected)

    def test_partial_trace_uses_overlap_on_traced_subsystem(self) -> None:
        envelope_table = {}

        mode_a = make_mode(
            path="p",
            polarization="h",
            envelope="env_a",
            envelope_table=envelope_table,
        )
        mode_b = make_mode(
            path="p",
            polarization="h",
            envelope="env_b",
            envelope_table=envelope_table,
        )

        overlap = 0.25 + 0.5j
        set_hermitian_overlap(
            envelope_table,
            mode_a.label.envelope,
            mode_b.label.envelope,
            overlap,
        )

        terms = (
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode_b.cre,), annihilators=()),
                right=Monomial(creators=(mode_a.cre,), annihilators=()),
            ),
        )

        result = density_partial_trace(terms, (mode_a, mode_b))

        expected = (
            DensityTerm(
                coeff=3.0 * overlap,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)

    def test_zero_new_coefficient_is_dropped(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=0.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )

        result = density_partial_trace(terms, (mode,))

        self.assertEqual(result, ())

    def test_multiple_terms_are_combined_after_trace(self) -> None:
        mode = make_mode()

        terms = (
            DensityTerm(
                coeff=2.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
            DensityTerm(
                coeff=3.0,
                left=Monomial(creators=(mode.cre,), annihilators=()),
                right=Monomial(creators=(mode.cre,), annihilators=()),
            ),
        )

        result = density_partial_trace(terms, (mode,))

        expected = (
            DensityTerm(
                coeff=5.0,
                left=Monomial.identity(),
                right=Monomial.identity(),
            ),
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
