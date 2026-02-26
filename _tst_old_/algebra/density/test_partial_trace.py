from __future__ import annotations
import unittest

from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel

from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import DensityTerm

from symop_proto.algebra.density.partial_trace import density_partial_trace


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


class TestDensityPartialTrace(ExtendedTestCase):
    def test_no_trace_leaves_terms_unchanged(self):
        a = make_mode("A")
        rho = (
            DensityTerm(
                2.0,
                left=Monomial(creators=(a.create,)),
                right=Monomial(annihilators=(a.ann,)),
            ),
        )
        out = density_partial_trace(rho, trace_over_modes=())
        self.assertEqual(out, rho)

    def test_trace_over_irrelevant_mode_no_change(self):
        a = make_mode("A")
        b = make_mode("B")
        rho = (
            DensityTerm(
                1.0,
                left=Monomial(creators=(a.create,)),
                right=Monomial(creators=(a.create,)),
            ),
        )
        # trace out B which is not present
        out = density_partial_trace(rho, trace_over_modes=(b,))
        self.assertEqual(out, rho)

    def test_trace_simple_creator_kept_removed_and_contracted(self):
        # Left and right each have one creator on A; tracing A removes it and contracts to 1
        a = make_mode("A")
        rho = (
            DensityTerm(
                3.0,
                left=Monomial(creators=(a.create,)),
                right=Monomial(creators=(a.create,)),
            ),
        )
        out = density_partial_trace(rho, trace_over_modes=(a,))
        self.assertEqual(len(out), 1)
        t = out[0]
        # kept parts become identity
        self.assertEqual(t.left.signature, Monomial().signature)
        self.assertEqual(t.right.signature, Monomial().signature)
        # coefficient is unchanged (contraction = 1)
        self.assertComplexAlmostEqual(t.coeff, 3.0 + 0.0j)

    def test_trace_mismatch_yields_zero_and_drops_term(self):
        # Left on A, Right on B -> tracing over A removes left traced part,
        # but right traced part is empty -> contraction <R^T|L^T> = <I|a^dag> = 0
        a = make_mode("A")
        b = make_mode("B")
        rho = (
            DensityTerm(
                1.0,
                left=Monomial(creators=(a.create,)),
                right=Monomial(creators=(b.create,)),
            ),
        )
        out = density_partial_trace(rho, trace_over_modes=(a,))
        self.assertEqual(out, ())

    def test_multiple_terms_merge_after_trace(self):
        a = make_mode("A")
        rho = (
            DensityTerm(
                1.0,
                left=Monomial(creators=(a.create,)),
                right=Monomial(creators=(a.create,)),
            ),
            DensityTerm(
                2.5,
                left=Monomial(creators=(a.create,)),
                right=Monomial(creators=(a.create,)),
            ),
        )
        out = density_partial_trace(rho, trace_over_modes=(a,))
        # both become identity on kept subsystem and should merge
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].left.signature, Monomial().signature)
        self.assertEqual(out[0].right.signature, Monomial().signature)
        self.assertComplexAlmostEqual(out[0].coeff, 3.5 + 0.0j)

    def test_trace_input_accepts_various_kinds(self):
        a = make_mode("A")
        # Build one density term supported only on A
        rho = (
            DensityTerm(
                1.0,
                left=Monomial(creators=(a.create,)),
                right=Monomial(creators=(a.create,)),
            ),
        )
        # Provide the same mode through different accepted kinds
        kinds = [
            a,  # ModeOp
            a.create,  # LadderOp (creator)
            Monomial(creators=(a.create,)),  # Monomial
            a.signature,  # raw signature tuple
        ]
        out = density_partial_trace(rho, trace_over_modes=kinds)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].left.signature, Monomial().signature)
        self.assertEqual(out[0].right.signature, Monomial().signature)
        self.assertComplexAlmostEqual(out[0].coeff, 1.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
