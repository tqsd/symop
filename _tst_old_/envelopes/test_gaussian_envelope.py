import numpy as np


from symop_proto.envelopes.base import _overlap_numeric
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from tests.utils.case import ExtendedTestCase


def gauss_overlap_closed_form(
    *,
    sigma1: float,
    sigma2: float,
    omega1: float,
    omega2: float,
    tau1: float,
    tau2: float,
) -> complex:
    """
    Closed-form from the paper (no global phase):
    """
    s1, s2 = sigma1, sigma2
    w1, w2 = omega1, omega2
    t1, t2 = tau1, tau2

    S2 = s1**2 + s2**2
    dt = t1 - t2
    dw = w1 - w2

    pref = np.sqrt(2.0 * s1 * s2 / S2)
    mag = np.exp(-(dt**2) / (4.0 * S2) - (s1**2 * s2**2 / S2) * (dw**2))
    ph = np.exp(1j * (dt * (s1**2 * w1 + s2**2 * w2) / S2))
    return pref * mag * ph


class TestGaussianEnvelope(ExtendedTestCase):
    def test_time_eval_normalization(self):
        g = GaussianEnvelope(omega0=1.0, sigma=0.4, tau=0.3, phi0=0.7)
        c, s = g.center_and_scale()
        T = 8.0 * s
        t = np.linspace(c - T, c + T, 1 << 15)
        f = g.time_eval(t)
        val = np.trapezoid(np.abs(f) ** 2, t)
        self.assertAlmostEqual(float(val), 1.0, places=9)

    def test_delayed_and_phased(self):
        g = GaussianEnvelope(1.0, 0.3, 0.2, 0.0)
        self.assertEqual(g.delayed(0.5).tau, g.tau + 0.5)
        self.assertEqual(g.phased(0.25).phi0, g.phi0 + 0.25)

    def test_overlap_self(self):
        g = GaussianEnvelope(1.0, 0.3, 0.0, 0.0)
        z = g.overlap(g)
        self.assertAlmostEqual(z.real, 1.0, places=12)
        self.assertAlmostEqual(z.imag, 0.0, places=12)

    def test_overlap_numeric_matches_analytic(self):
        g1 = GaussianEnvelope(omega0=1.0, sigma=0.35, tau=-0.2, phi0=0.4)
        g2 = GaussianEnvelope(omega0=1.3, sigma=0.5, tau=+0.1, phi0=-0.6)
        # analytic
        z_a = g1.overlap(g2)
        # numeric (same window heuristic as BaseEnvelope)
        c = 0.5 * (g1.tau + g2.tau)
        S = max(g1.sigma, g2.sigma)
        z_n = _overlap_numeric(
            g1.time_eval, g2.time_eval, tmin=c - 8 * S, tmax=c + 8 * S
        )
        self.assertAlmostEqual(z_a.real, z_n.real, places=8)
        self.assertAlmostEqual(z_a.imag, z_n.imag, places=8)

    def test_phase_only(self):
        g = GaussianEnvelope(1.0, 0.3, 0.0, 0.0)
        for phi in (0.1, -0.7, 1.2):
            z = g.overlap(g.phased(phi))
            self.assertAlmostEqual(z.real, np.cos(phi), places=9)
            self.assertAlmostEqual(z.imag, np.sin(phi), places=9)

    def test_signatures(self):
        g = GaussianEnvelope(1.0, 0.3, 0.2, 0.5)
        sig = g.signature
        self.assertEqual(sig[0], "gauss")
        s8 = g.approx_signature(decimals=8)
        s6 = g.approx_signature(decimals=6, ignore_global_phase=True)
        self.assertEqual(s8[0], "gauss_approx")
        # last element should be rounded/zeroed when ignoring global phase
        self.assertEqual(s6[-1], 0.0)


class TestGaussianClosedForm(ExtendedTestCase):
    def test_matches_paper_formula_without_global_phase(self):
        # phi0 = 0 for both => implementation should match the LaTeX exactly
        cases = [
            # (omega1, sigma1, tau1, omega2, sigma2, tau2)
            (1.0, 0.35, -0.2, 1.3, 0.5, 0.1),
            (0.0, 0.20, 0.0, 0.0, 0.20, 0.0),  # identical -> 1
            (2.1, 0.75, 0.8, 1.5, 0.40, -0.3),
            (3.5, 1.00, -0.5, 3.5, 0.60, -0.2),
        ]
        for w1, s1, t1, w2, s2, t2 in cases:
            g1 = GaussianEnvelope(omega0=w1, sigma=s1, tau=t1, phi0=0.0)
            g2 = GaussianEnvelope(omega0=w2, sigma=s2, tau=t2, phi0=0.0)

            z_impl = g1.overlap(g2)
            z_cf = gauss_overlap_closed_form(
                sigma1=s1, sigma2=s2, omega1=w1, omega2=w2, tau1=t1, tau2=t2
            )
            # very tight tolerance (all analytic)
            self.assertComplexAlmostEqual(z_impl, z_cf, rtol=1e-12, atol=1e-12)

    def test_matches_paper_formula_with_global_phase_extension(self):
        # Implementation includes exp(i*(phi2 - phi1)); check that explicitly
        cases = [
            (1.0, 0.35, -0.2, +0.40, 1.3, 0.5, 0.1, -0.10),
            (2.0, 0.25, 0.0, +1.23, 2.0, 0.25, 0.2, +0.77),
        ]
        for w1, s1, t1, p1, w2, s2, t2, p2 in cases:
            g1 = GaussianEnvelope(omega0=w1, sigma=s1, tau=t1, phi0=p1)
            g2 = GaussianEnvelope(omega0=w2, sigma=s2, tau=t2, phi0=p2)

            z_impl = g1.overlap(g2)

            z_cf = gauss_overlap_closed_form(
                sigma1=s1, sigma2=s2, omega1=w1, omega2=w2, tau1=t1, tau2=t2
            ) * np.exp(1j * (p2 - p1))  # global-phase extension
            self.assertComplexAlmostEqual(z_impl, z_cf, rtol=1e-12, atol=1e-12)

    def test_self_overlap_is_one(self):
        g = GaussianEnvelope(omega0=1.2, sigma=0.4, tau=0.3, phi0=0.7)
        z = g.overlap(g)
        self.assertComplexAlmostEqual(z, 1.0 + 0j, rtol=1e-13, atol=1e-13)

    def test_conjugate_symmetry(self):
        g1 = GaussianEnvelope(omega0=1.0, sigma=0.35, tau=-0.2, phi0=0.4)
        g2 = GaussianEnvelope(omega0=1.3, sigma=0.5, tau=+0.1, phi0=-0.6)
        z12 = g1.overlap(g2)
        z21 = g2.overlap(g1)
        self.assertComplexAlmostEqual(z12.conjugate(), z21, rtol=1e-13, atol=1e-13)
