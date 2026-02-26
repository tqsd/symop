from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from symop_proto.core.protocols import SignatureProto
from symop_proto.envelopes.base import BaseEnvelope
from symop_proto.envelopes.protocols import EnvelopeProto, FloatArray, RCArray
from symop_proto.envelopes.spectral_filters.transfer import SpectralTransfer


def _interp_complex_1d(x: FloatArray, xp: FloatArray, fp: RCArray) -> RCArray:
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=complex)

    re = np.interp(x, xp, np.real(fp), left=0.0, right=0.0)
    im = np.interp(x, xp, np.imag(fp), left=0.0, right=0.0)
    return (re + 1j * im).astype(complex)


def _fit_linear_phase_delay(
    w_rel: np.ndarray, Z: np.ndarray, *, frac: float = 0.2
) -> float:
    """Estimate delay t0 from the linear phase of Z(w_rel) around w_rel=0.

    If Z(w_rel) ~ A(w_rel) * exp(-i w_rel t0), then unwrap(angle(Z)) has slope -t0.
    """
    n = int(w_rel.size)
    if n < 8:
        return 0.0

    # Use a central window (avoid edges where amplitude is tiny and phase is noisy)
    m = max(4, int(frac * n))
    i0 = n // 2
    sl = slice(i0 - m // 2, i0 + m // 2)

    w = w_rel[sl]
    z = Z[sl]

    a = np.abs(z)
    # avoid points with near-zero magnitude
    mask = a > (np.max(a) * 1e-8)
    if not np.any(mask):
        return 0.0

    w = w[mask]
    phi = np.unwrap(np.angle(z[mask]))

    # Weighted least squares for phi ≈ b + s*w
    w_mean = float(np.mean(w))
    phi_mean = float(np.mean(phi))
    ww = w - w_mean
    pp = phi - phi_mean

    denom = float(np.dot(ww, ww))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0

    slope = float(np.dot(ww, pp) / denom)  # phi slope dphi/dw
    t0 = -slope
    if not np.isfinite(t0):
        return 0.0
    return t0


def _remove_linear_phase(w_rel: np.ndarray, Z: np.ndarray, t0: float) -> np.ndarray:
    return Z * np.exp(1j * w_rel * float(t0))


def _estimate_spectral_window(
    env: EnvelopeProto,
    *,
    w0_fallback: float,
    sigma_w_fallback: float,
) -> tuple[float, float]:
    # Prefer explicit spectral metadata if available.
    w0 = getattr(env, "omega0", None)
    if w0 is not None:
        w0 = float(w0)
    else:
        w0 = float(w0_fallback)

    sigma_w = getattr(env, "omega_sigma", None)
    if sigma_w is not None:
        sigma_w = float(sigma_w)
    else:
        sigma = getattr(env, "sigma", None)
        if sigma is not None:
            sigma_w = float(sigma)
        else:
            sigma_w = float(sigma_w_fallback)

    sigma_w = max(sigma_w, 1e-12)
    return w0, sigma_w


@dataclass(frozen=True)
class FilteredEnvelope(BaseEnvelope):
    r"""Envelope defined by spectral multiplication:

    z_out(w) = H(w) * z_in(w)

    Time-domain is obtained numerically via an inverse Fourier transform
    on a fixed grid.

    Notes
    -----
    - This class is intended to make basis changes "real": ModeBasis overlaps
      may call time_eval() (via numeric overlap fallback), so time_eval must
      be coherent with freq_eval.
    - We implement a practical FFT-based evaluation with interpolation.
    - The absolute Fourier convention is not important as long as overlap and
      plots are internally consistent. If you later care about absolute scaling,
      you can unify the convention across all envelopes.

    """

    base: EnvelopeProto
    transfer: SpectralTransfer

    # FFT grid controls (good defaults; you can expose them later if needed)
    n_fft: int = 2**15
    w_span_sigma: float = 12.0

    @property
    def signature(self) -> SignatureProto:
        return ("filtered", self.base.signature, self.transfer.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
        **kw: Any,
    ) -> SignatureProto:
        return (
            "filtered_approx",
            getattr(self.base, "approx_signature", lambda **_: self.base.signature)(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            ),
            self.transfer.signature,
            int(self.n_fft),
            float(self.w_span_sigma),
        )

    def center_and_scale(self) -> tuple[float, float]:
        w0 = float(getattr(self.base, "omega0", 0.0))
        sigma_w = float(getattr(self.base, "sigma", 1.0))
        sigma_w = max(sigma_w, 1e-12)

        W = float(self.w_span_sigma) * sigma_w
        n = int(self.n_fft)
        if n < 64 or not np.isfinite(W) or W <= 0.0:
            return self.base.center_and_scale()

        dw = 2.0 * W / float(n)
        k = np.arange(n, dtype=float) - (n // 2)
        w = w0 + k * dw

        Z = np.asarray(self.freq_eval(w), dtype=complex)
        if not np.isfinite(Z).all():
            return self.base.center_and_scale()

        # Optional: remove linear phase ramp (same as in time_eval)
        t0 = _fit_linear_phase_delay(k * dw, Z, frac=0.2)
        Z = _remove_linear_phase(k * dw, Z, t0)

        z = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Z))) * (n * dw) / (2.0 * np.pi)

        dt = 2.0 * np.pi / (float(n) * dw)
        t_grid = k * dt

        p = np.abs(z) ** 2
        s = float(np.sum(p))
        if not (s > 0.0) or not np.isfinite(s):
            return self.base.center_and_scale()

        c = float(np.sum(t_grid * p) / s)
        v = float(np.sum(((t_grid - c) ** 2) * p) / s)
        scale = float(np.sqrt(max(v, 1e-30)))
        return c, scale

    def freq_eval(self, w: FloatArray) -> RCArray:
        w = np.asarray(w, dtype=float)

        Hw = np.asarray(self.transfer(w), dtype=complex)
        if not np.isfinite(Hw).all():
            bad = np.argwhere(~np.isfinite(Hw))
            i = int(bad[0, 0])
            raise ValueError(f"Non-finite transfer(w) at w={w[i]}: {Hw[i]!r}")

        Zw = np.asarray(self.base.freq_eval(w), dtype=complex)
        if not np.isfinite(Zw).all():
            bad = np.argwhere(~np.isfinite(Zw))
            i = int(bad[0, 0])
            raise ValueError(f"Non-finite base.freq_eval(w) at w={w[i]}: {Zw[i]!r}")

        out = Hw * Zw
        if not np.isfinite(out).all():
            bad = np.argwhere(~np.isfinite(out))
            i = int(bad[0, 0])
            raise ValueError(
                f"Non-finite product at w={w[i]}: H={Hw[i]!r}, base={Zw[i]!r}"
            )
        return out

    def time_eval(self, t: FloatArray) -> RCArray:
        t = np.asarray(t, dtype=float)

        w0 = float(getattr(self.base, "omega0", 0.0))

        # width-based window (your existing heuristic)
        _, sigma_w = _estimate_spectral_window(
            self.base, w0_fallback=w0, sigma_w_fallback=1.0
        )
        W_width = float(self.w_span_sigma) * float(sigma_w)

        # carrier-resolution window: enforce omega0 * dt <= phase_step
        phase_step = 0.2
        W_carrier = (np.pi * abs(w0)) / max(phase_step, 1e-12)

        W = max(W_width, W_carrier, 1e-12)

        n = int(self.n_fft)
        if n < 64:
            raise ValueError(f"n_fft too small: {n}")
        if not np.isfinite(W) or W <= 0.0:
            raise ValueError(f"Invalid frequency window W={W}")

        # FFT-consistent spacing and centered frequency bins:
        dw = 2.0 * W / float(n)  # bin spacing
        k = np.arange(n, dtype=float) - (n // 2)
        w_rel = k * dw  # centered bins: ... -2dw, -dw, 0, +dw, ...
        w = w0 + w_rel

        Z = np.asarray(self.freq_eval(w), dtype=complex)

        t0 = _fit_linear_phase_delay(w_rel, Z, frac=0.2)
        if abs(t0) > 0.0:
            Z = _remove_linear_phase(w_rel, Z, t0)

        if not np.isfinite(Z).all():
            bad = np.argwhere(~np.isfinite(Z))
            i = int(bad[0, 0])
            raise ValueError(f"Non-finite freq_eval at w={w[i]}: {Z[i]!r}")

        # IFFT: put zero-frequency at index 0
        Z0 = np.fft.ifftshift(Z)

        z_rel = np.fft.ifft(Z0) * (n * dw) / (2.0 * np.pi)
        z_rel = np.fft.fftshift(z_rel)

        # Matching centered time bins:
        dt = 2.0 * np.pi / (float(n) * dw)
        t_grid = k * dt

        z = z_rel * np.exp(1j * w0 * t_grid)

        return _interp_complex_1d(t, t_grid, z)

    def delayed(self, dt: float) -> FilteredEnvelope:
        base_delayed = (
            self.base.delayed(dt) if hasattr(self.base, "delayed") else self.base
        )
        return FilteredEnvelope(
            base=base_delayed,
            transfer=self.transfer,
            n_fft=self.n_fft,
            w_span_sigma=self.w_span_sigma,
        )

    def phased(self, dphi: float) -> FilteredEnvelope:
        base_phased = (
            self.base.phased(dphi) if hasattr(self.base, "phased") else self.base
        )
        return FilteredEnvelope(
            base=base_phased,
            transfer=self.transfer,
            n_fft=self.n_fft,
            w_span_sigma=self.w_span_sigma,
        )

    @property
    def latex(self) -> str | None:
        return r"\zeta_{\mathrm{out}}(\omega)=H(\omega)\,\zeta_{\mathrm{in}}(\omega)"

    def overlap_with_generic(self, other: EnvelopeProto) -> complex:
        # Anchor the window to self (the filtered thing) unless other has omega0 too.
        w0_self, sigma_w_self = _estimate_spectral_window(
            self.base,
            w0_fallback=0.0,
            sigma_w_fallback=1.0,
        )
        w0_other, sigma_w_other = _estimate_spectral_window(
            other,
            w0_fallback=w0_self,  # if other lacks omega0, do NOT drag to 0
            sigma_w_fallback=sigma_w_self,
        )

        # If other had a real omega0 attribute, we can average; otherwise keep self’s anchor.
        other_has_w0 = hasattr(other, "omega0")
        w0 = 0.5 * (w0_self + w0_other) if other_has_w0 else w0_self

        sigma_w = max(sigma_w_self, sigma_w_other)
        W = float(self.w_span_sigma) * sigma_w

        n = int(self.n_fft)
        if n < 64:
            raise ValueError(f"n_fft too small: {n}")
        if not np.isfinite(W) or not (W > 0.0):
            raise ValueError(f"Invalid frequency window W={W}")

        w = np.linspace(w0 - W, w0 + W, n, dtype=float)

        z1 = np.asarray(self.freq_eval(w), dtype=complex)
        z2 = np.asarray(other.freq_eval(w), dtype=complex)

        if not np.isfinite(z1).all():
            bad = np.argwhere(~np.isfinite(z1))
            i = int(bad[0, 0])
            raise ValueError(f"Non-finite self.freq_eval at w={w[i]}: {z1[i]!r}")
        if not np.isfinite(z2).all():
            bad = np.argwhere(~np.isfinite(z2))
            i = int(bad[0, 0])
            raise ValueError(f"Non-finite other.freq_eval at w={w[i]}: {z2[i]!r}")

        ov = np.trapezoid(np.conjugate(z1) * z2, w)
        return complex(ov / (2.0 * np.pi))
