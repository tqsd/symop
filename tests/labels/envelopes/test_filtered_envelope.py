import unittest

import numpy as np

from symop.modes.envelopes.filtered import (
    FilteredEnvelope,
    _estimate_spectral_window,
    _fit_linear_phase_delay,
    _interp_complex_1d,
    _remove_linear_phase,
)
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.transfer.constant_phase import ConstantPhase


class _OnesTransfer:
    @property
    def signature(self):
        return ("ones_tf",)

    def __call__(self, w):
        w = np.asarray(w, dtype=float)
        return np.ones_like(w, dtype=complex)


class _BadTransferNan:
    @property
    def signature(self):
        return ("bad_tf_nan",)

    def __call__(self, w):
        w = np.asarray(w, dtype=float)
        out = np.ones_like(w, dtype=complex)
        if out.size > 0:
            out[out.size // 2] = np.nan + 0.0j
        return out


class _BadEnvelope:
    def __init__(self, mode="nan"):
        self.mode = mode
        self.omega0 = 0.0
        self.sigma = 1.0

    @property
    def signature(self):
        return ("bad_env", self.mode)

    def center_and_scale(self):
        return 0.0, 1.0

    def freq_eval(self, w):
        w = np.asarray(w, dtype=float)
        out = np.ones_like(w, dtype=complex)
        if out.size > 0:
            out[out.size // 2] = (
                (np.nan + 0.0j) if self.mode == "nan" else (np.inf + 0.0j)
            )
        return out


class TestFilteredEnvelopeHelpers(unittest.TestCase):
    def test_interp_complex_1d_interpolates_re_im_and_zeros_outside(self):
        xp = np.array([0.0, 1.0, 2.0], dtype=float)
        fp = np.array([0.0 + 0.0j, 1.0 + 1.0j, 2.0 + 0.0j], dtype=complex)

        x = np.array([-1.0, 0.5, 1.5, 3.0], dtype=float)
        y = _interp_complex_1d(x, xp, fp)

        self.assertTrue(np.iscomplexobj(y))
        self.assertEqual(y.shape, x.shape)

        # outside -> 0
        self.assertEqual(y[0], 0.0 + 0.0j)
        self.assertEqual(y[-1], 0.0 + 0.0j)

        # linear interpolation
        self.assertAlmostEqual(y[1].real, 0.5, places=14)
        self.assertAlmostEqual(y[1].imag, 0.5, places=14)
        self.assertAlmostEqual(y[2].real, 1.5, places=14)
        self.assertAlmostEqual(y[2].imag, 0.5, places=14)

    def test_fit_linear_phase_delay_recovers_delay(self):
        n = 256
        w_rel = np.linspace(-5.0, 5.0, n, dtype=float)
        t0 = 0.37

        amp = np.exp(-0.5 * (w_rel / 2.0) ** 2)
        Z = amp * np.exp(-1.0j * w_rel * t0)

        est = _fit_linear_phase_delay(w_rel, Z, frac=0.3)
        self.assertAlmostEqual(est, t0, places=2)

    def test_fit_linear_phase_delay_small_n_returns_zero(self):
        w_rel = np.linspace(-1.0, 1.0, 6, dtype=float)
        Z = np.exp(-1.0j * w_rel * 0.5)
        self.assertEqual(_fit_linear_phase_delay(w_rel, Z), 0.0)

    def test_remove_linear_phase_cancels_delay(self):
        n = 256
        w_rel = np.linspace(-3.0, 3.0, n, dtype=float)
        t0 = 0.8
        Z = np.exp(-1.0j * w_rel * t0)

        Zc = _remove_linear_phase(w_rel, Z, t0)
        # Should be ~1 everywhere (up to numerical error)
        np.testing.assert_allclose(Zc, np.ones_like(Zc), atol=1e-12, rtol=0.0)

    def test_estimate_spectral_window_prefers_metadata(self):
        base = GaussianEnvelope(omega0=4.0, sigma=0.5, tau=0.0, phi0=0.0)
        w0, sw = _estimate_spectral_window(base, w0_fallback=1.0, sigma_w_fallback=2.0)

        self.assertAlmostEqual(w0, 4.0, places=14)
        # GaussianEnvelope.omega_sigma is ~ 1/sigma
        self.assertAlmostEqual(sw, 1.0 / 0.5, places=12)

    def test_estimate_spectral_window_fallbacks(self):
        class NoMeta:
            signature = ("no_meta",)

            def center_and_scale(self):
                return 0.0, 1.0

            def freq_eval(self, w):
                w = np.asarray(w, dtype=float)
                return np.ones_like(w, dtype=complex)

        env = NoMeta()
        w0, sw = _estimate_spectral_window(env, w0_fallback=3.0, sigma_w_fallback=7.0)
        self.assertEqual(w0, 3.0)
        self.assertEqual(sw, 7.0)


class TestFilteredEnvelope(unittest.TestCase):
    def setUp(self):
        self.base = GaussianEnvelope(omega0=5.0, sigma=0.7, tau=0.3, phi0=0.2)

    def test_signature(self):
        tf = _OnesTransfer()
        env = FilteredEnvelope(base=self.base, transfer=tf, n_fft=256, w_span_sigma=8.0)
        self.assertEqual(env.signature, ("filtered", self.base.signature, tf.signature))

    def test_approx_signature_uses_base_approx_if_available(self):
        tf = _OnesTransfer()
        env = FilteredEnvelope(base=self.base, transfer=tf, n_fft=128, w_span_sigma=9.0)

        got = env.approx_signature(decimals=3, ignore_global_phase=True)
        self.assertEqual(got[0], "filtered_approx")
        self.assertEqual(
            got[1],
            self.base.approx_signature(decimals=3, ignore_global_phase=True),
        )
        self.assertEqual(got[2], tf.signature)
        self.assertEqual(got[3], 128)
        self.assertEqual(got[4], 9.0)

    def test_freq_eval_is_product_and_raises_on_nonfinite_transfer(self):
        w = np.linspace(0.0, 1.0, 17, dtype=float)

        good = FilteredEnvelope(
            base=self.base,
            transfer=_OnesTransfer(),
            n_fft=256,
            w_span_sigma=6.0,
        )
        out = good.freq_eval(w)
        np.testing.assert_allclose(out, self.base.freq_eval(w))

        bad = FilteredEnvelope(
            base=self.base,
            transfer=_BadTransferNan(),
            n_fft=256,
            w_span_sigma=6.0,
        )
        with self.assertRaises(ValueError):
            _ = bad.freq_eval(w)

    def test_freq_eval_raises_on_nonfinite_base(self):
        w = np.linspace(0.0, 1.0, 17, dtype=float)

        bad_base = _BadEnvelope(mode="nan")
        env = FilteredEnvelope(
            base=bad_base,
            transfer=_OnesTransfer(),
            n_fft=256,
            w_span_sigma=6.0,
        )
        with self.assertRaises(ValueError):
            _ = env.freq_eval(w)

    def test_freq_eval_constant_phase_is_exact(self):
        phi = 0.6
        tf = ConstantPhase(phi0=phi)
        env = FilteredEnvelope(
            base=self.base, transfer=tf, n_fft=512, w_span_sigma=10.0
        )

        w = np.linspace(
            self.base.omega0 - 10.0,
            self.base.omega0 + 10.0,
            500,
            dtype=float,
        )

        Z_base = self.base.freq_eval(w)
        Z_filt = env.freq_eval(w)

        np.testing.assert_allclose(
            Z_filt,
            np.exp(1j * phi) * Z_base,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_time_eval_phase_only_preserves_energy(self):
        phi = 0.6
        tf = ConstantPhase(phi0=phi)
        env = FilteredEnvelope(
            base=self.base, transfer=tf, n_fft=1024, w_span_sigma=12.0
        )

        t0, s0 = self.base.center_and_scale()
        t = t0 + np.linspace(-4.0 * s0, 4.0 * s0, 201)

        z_base = self.base.time_eval(t)
        z_filt = env.time_eval(t)

        e_base = float(np.trapezoid(np.abs(z_base) ** 2, t))
        e_filt = float(np.trapezoid(np.abs(z_filt) ** 2, t))

        rel = abs(e_filt - e_base) / max(abs(e_base), 1e-30)
        self.assertLess(rel, 1e-2)

    def test_time_eval_raises_if_n_fft_too_small(self):
        env = FilteredEnvelope(
            base=self.base,
            transfer=_OnesTransfer(),
            n_fft=32,
            w_span_sigma=8.0,
        )
        with self.assertRaises(ValueError):
            _ = env.time_eval([0.0, 1.0])

    def test_delayed_and_phased_forward_to_base_when_available(self):
        tf = _OnesTransfer()
        env = FilteredEnvelope(base=self.base, transfer=tf, n_fft=256, w_span_sigma=8.0)

        env_d = env.delayed(0.4)
        self.assertNotEqual(env_d.base.signature, env.base.signature)
        self.assertAlmostEqual(env_d.base.tau, self.base.tau + 0.4, places=14)

        env_p = env.phased(0.9)
        self.assertNotEqual(env_p.base.signature, env.base.signature)
        self.assertAlmostEqual(env_p.base.phi0, self.base.phi0 + 0.9, places=14)

    def test_latex(self):
        env = FilteredEnvelope(
            base=self.base,
            transfer=_OnesTransfer(),
            n_fft=256,
            w_span_sigma=8.0,
        )
        self.assertIsInstance(env.latex, str)
        self.assertIn("Z_{\\mathrm{out}}", env.latex)

    def test_center_and_scale_returns_finite(self):
        env = FilteredEnvelope(
            base=self.base,
            transfer=_OnesTransfer(),
            n_fft=512,
            w_span_sigma=10.0,
        )
        c, s = env.center_and_scale()
        self.assertTrue(np.isfinite(c))
        self.assertTrue(np.isfinite(s))
        self.assertTrue(s > 0.0)

    def test_overlap_with_generic_self_is_close_to_one(self):
        env = FilteredEnvelope(
            base=self.base,
            transfer=_OnesTransfer(),
            n_fft=1024,
            w_span_sigma=10.0,
        )
        ov = env.overlap_with_generic(self.base)
        self.assertIsInstance(ov, complex)
        self.assertAlmostEqual(abs(ov), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
