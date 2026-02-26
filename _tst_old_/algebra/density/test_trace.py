from __future__ import annotations
import unittest

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from tests.utils.case import ExtendedTestCase

from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.core.operators import ModeOp
from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import DensityTerm
from symop_proto.algebra.density.trace import density_trace


class TestDensityTrace(ExtendedTestCase):
    def test_identity_term_has_trace_equal_coeff(self):
        m = Monomial()
        rho = (DensityTerm(2.0, left=m, right=m),)
        self.assertComplexAlmostEqual(density_trace(rho), 2.0 + 0.0j)

    def test_multiple_terms_add_up(self):
        m = Monomial()
        rho = (
            DensityTerm(1.0, left=m, right=m),
            DensityTerm(2.0, left=m, right=m),
        )
        self.assertComplexAlmostEqual(density_trace(rho), 3.0 + 0.0j)

    def test_nonmatching_left_right_gives_zero(self):

        mA = ModeOp(
            GaussianEnvelope(omega0=1.0, sigma=0.3, tau=0.0),
            ModeLabel(PathLabel("A"), PolarizationLabel.H()),
        )
        mB = ModeOp(
            GaussianEnvelope(omega0=1.0, sigma=0.3, tau=0.0),
            ModeLabel(PathLabel("B"), PolarizationLabel.H()),
        )

        left = Monomial(creators=(mA.create,))
        right = Monomial(creators=(mB.create,))
        rho = (DensityTerm(1.0, left=left, right=right),)
        val = density_trace(rho)
        self.assertAlmostEqual(abs(val), 0.0, places=12)

    def test_empty_input_returns_zero(self):
        self.assertComplexAlmostEqual(density_trace(()), 0.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
