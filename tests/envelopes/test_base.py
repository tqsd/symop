import numpy as np

from tests.helpers.dummy_envelopes import HookEnv, OpaqueEnv
from tests.helpers.rect_envelope import RectEnvelope
from tests.utils.case import ExtendedTestCase
from symop_proto.envelopes.base import BaseEnvelope, _overlap_numeric


class TestOverlapNumeric(ExtendedTestCase):

    def test_constant_functions_exact_formula(self):
        tmin, tmax, n = 0.0, 1.0, 11

        def f(t):
            return np.ones_like(t, dtype=complex)

        val = _overlap_numeric(f, f, tmin=tmin, tmax=tmax, n=n)

        expected = tmax - tmin
        self.assertAlmostEqual(val.real, expected, places=12)
        self.assertAlmostEqual(val.imag, 0.0, places=12)

    def test_conjugation(self):
        omega = 2.0

        def f(t):
            return np.exp(1j * omega * t)

        val = _overlap_numeric(f, f, tmin=0.0, tmax=1.0, n=4097)
        expected = 1
        self.assertAlmostEqual(val.real, expected, places=9)
        self.assertAlmostEqual(val.imag, 0.0, places=9)


class TestBaseEnvelopeOverlap(ExtendedTestCase):
    def test_self_overlap_is_one(self):
        e = RectEnvelope(center=0.0, width=2.0)
        z = e.overlap(e)
        self.assertComplexAlmostEqual(z, 1.0 + 0j, rtol=5e-5, atol=1e-5)

    def test_shifted_rect_overlap_matches_analytic(self):
        W = 2.0
        e1 = RectEnvelope(center=0.0, width=W)
        for shift in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
            e2 = e1.delayed(shift)
            z = e1.overlap(e2)
            expected = max(0.0, 1.0 - abs(shift) / W)

            self.assertAlmostEqual(z.real, expected, places=3)
            self.assertAlmostEqual(z.imag, 0.0, places=9)

    def test_phase_difference_appears_as_global_phase(self):
        W = 1.0
        phi = 0.4
        e1 = RectEnvelope(center=0.0, width=W, phase=0.0)
        e2 = e1.phased(phi)
        z = e1.overlap(e2)
        self.assertComplexAlmostEqual(
            z, np.exp(1j * phi), rtol=1e-4, atol=1e-6
        )

    def test_conjugate_symmetry(self):
        e1 = RectEnvelope(center=-0.3, width=1.2, phase=0.7)
        e2 = RectEnvelope(center=+0.4, width=1.2, phase=-0.2)
        z12 = e1.overlap(e2)
        z21 = e2.overlap(e1)
        self.assertAlmostEqual((z12.conjugate() - z21).real, 0.0, places=10)
        self.assertAlmostEqual((z12.conjugate() - z21).imag, 0.0, places=10)

    def test_cross_family_hook_is_used(self):
        e = RectEnvelope()
        h = HookEnv()
        z = e.overlap(
            h
        )  # BaseEnvelope should call other.overlap_with_generic(self)
        self.assertAlmostEqual(z.real, 0.123, places=15)
        self.assertAlmostEqual(z.imag, 0.0, places=15)

    def test_raises_when_no_timeeval_and_no_hook(self):
        e = RectEnvelope()
        with self.assertRaises(TypeError):
            _ = e.overlap(OpaqueEnv())
