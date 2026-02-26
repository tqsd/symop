"""Spectrally filtered envelopes.

This module defines :class:`FilteredEnvelope`, an envelope constructed
by multiplying a base spectrum with a transfer function in the
frequency domain. The resulting time-domain field is obtained via
FFT-based inverse transformation and interpolation.

It also provides numerical helpers for spectral interpolation,
linear phase (delay) estimation, and window selection for
frequency-domain quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from symop.core.protocols.signature import SignatureProto
from symop.modes.envelopes.base import BaseEnvelope
from symop.modes.protocols import (
    EnvelopeProto,
    HasLatex,
    TransferFunctionProto,
)
from symop.modes.types import FloatArray, RCArray


def _interp_complex_1d(x: FloatArray, xp: FloatArray, fp: RCArray) -> RCArray:
    """Interpolate a complex-valued 1D function on a real grid.

    This interpolates real and imaginary parts separately using
    :func:`numpy.interp`, with values outside the interpolation domain set to
    zero.

    Parameters
    ----------
    x:
        Target points.
    xp:
        Sample grid (must be 1D, increasing).
    fp:
        Complex samples at ``xp``.

    Returns
    -------
    RCArray
        Interpolated complex samples at ``x``.

    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=complex)

    re = np.interp(x, xp, np.real(fp), left=0.0, right=0.0)
    im = np.interp(x, xp, np.imag(fp), left=0.0, right=0.0)
    return cast(RCArray, (re + 1j * im).astype(complex))


def _fit_linear_phase_delay(
    w_rel: np.ndarray, Z: np.ndarray, *, frac: float = 0.2
) -> float:
    r"""Estimate a time delay from the linear phase of a spectrum.

    If, locally around :math:`\omega_\mathrm{rel}=0`, the spectrum behaves like

    .. math::

        Z(\omega_\mathrm{rel}) \approx A(\omega_\mathrm{rel})\,e^{-i\omega_\mathrm{rel} t_0},

    then the unwrapped phase satisfies :math:`\phi(\omega_\mathrm{rel}) \approx -t_0\,\omega_\mathrm{rel} + b`.
    This function estimates :math:`t_0` from a least-squares fit of the phase slope.

    Parameters
    ----------
    w_rel:
        Relative angular-frequency samples centered near 0.
    Z:
        Complex spectrum values at ``w_rel``.
    frac:
        Fraction of the central bins used for fitting (default: 0.2).

    Returns
    -------
    float
        Estimated delay :math:`t_0`. Returns 0.0 if the fit is ill-conditioned.

    """
    n = int(w_rel.size)
    if n < 8:
        return 0.0

    m = max(4, int(frac * n))
    i0 = n // 2
    sl = slice(i0 - m // 2, i0 + m // 2)

    w = w_rel[sl]
    z = Z[sl]

    a = np.abs(z)
    mask = a > (np.max(a) * 1e-8)
    if not np.any(mask):
        return 0.0

    w = w[mask]
    phi = np.unwrap(np.angle(z[mask]))

    w_mean = float(np.mean(w))
    phi_mean = float(np.mean(phi))
    ww = w - w_mean
    pp = phi - phi_mean

    denom = float(np.dot(ww, ww))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0

    slope = float(np.dot(ww, pp) / denom)
    t0 = -slope
    if not np.isfinite(t0):
        return 0.0
    return t0


def _remove_linear_phase(w_rel: np.ndarray, Z: np.ndarray, t0: float) -> np.ndarray:
    r"""Remove a linear phase ramp corresponding to a delay.

    This multiplies

    .. math::

        Z(\omega_\mathrm{rel}) \leftarrow Z(\omega_\mathrm{rel})\,e^{+i\omega_\mathrm{rel} t_0}.

    Parameters
    ----------
    w_rel:
        Relative angular-frequency samples.
    Z:
        Complex spectrum values.
    t0:
        Delay estimate.

    Returns
    -------
    ndarray
        Phase-corrected spectrum.

    """
    return Z * np.exp(1j * w_rel * float(t0))


def _estimate_spectral_window(
    env: EnvelopeProto,
    *,
    w0_fallback: float,
    sigma_w_fallback: float,
) -> tuple[float, float]:
    r"""Estimate a reasonable spectral window center and width for FFT-based evaluation.

    This prefers envelope metadata if available:
    - :math:`\omega_0` from ``env.omega0`` (if present)
    - a spectral width hint from ``env.omega_sigma`` (if present)

    Parameters
    ----------
    env:
        Envelope providing (optional) metadata.
    w0_fallback:
        Fallback center frequency if the envelope does not provide ``omega0``.
    sigma_w_fallback:
        Fallback width hint if the envelope does not provide any width metadata.

    Returns
    -------
    (w0, sigma_w):
        Center and width hint for building a spectral window.

    """
    w0_tmp = getattr(env, "omega0", None)
    w0 = float(w0_tmp) if w0_tmp is not None else float(w0_fallback)

    sigma_w_tmp = getattr(env, "omega_sigma", None)
    if sigma_w_tmp is not None:
        sigma_w = float(sigma_w_tmp)
    else:
        sigma_tmp = getattr(env, "sigma", None)
        sigma_w = float(sigma_tmp) if sigma_tmp is not None else float(sigma_w_fallback)

    sigma_w = max(sigma_w, 1e-12)
    return w0, sigma_w


@dataclass(frozen=True)
class FilteredEnvelope(BaseEnvelope, HasLatex):
    r"""Envelope defined by spectral multiplication.

    Given an input spectrum :math:`Z_\mathrm{in}(\omega)` and a transfer function
    :math:`H(\omega)`, the output spectrum is

    .. math::

        Z_\mathrm{out}(\omega) = H(\omega)\,Z_\mathrm{in}(\omega).

    The time-domain field :math:`\zeta_\mathrm{out}(t)` is obtained numerically
    using an FFT-based inverse transform on a fixed grid, followed by complex
    interpolation.

    Notes
    -----
    - This class is intended to make basis changes "real": generic overlaps may
      call :meth:`time_eval` via numeric fallback, so :meth:`time_eval` should be
      coherent with :meth:`freq_eval`.
    - The absolute Fourier convention is not critical as long as internal
      operations (overlap, plotting) are consistent. If you later care about
      absolute scaling, unify the convention across all envelopes.

    """

    base: EnvelopeProto
    transfer: TransferFunctionProto

    n_fft: int = 2**15
    w_span_sigma: float = 12.0

    @property
    def signature(self) -> SignatureProto:
        """Stable signature for caching/comparison."""
        return ("filtered", self.base.signature, self.transfer.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> SignatureProto:
        """Approximate signature with rounded base-envelope parameters.

        Parameters
        ----------
        decimals:
            Number of decimals for rounding in the base envelope's approx signature.
        ignore_global_phase:
            If True, request that the base envelope ignore its global phase (if supported).

        Returns
        -------
        SignatureProto
            Approximate signature tuple.

        """
        base_approx = getattr(self.base, "approx_signature", None)
        if callable(base_approx):
            base_sig = base_approx(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            )
        else:
            base_sig = self.base.signature

        return (
            "filtered_approx",
            base_sig,
            self.transfer.signature,
            int(self.n_fft),
            float(self.w_span_sigma),
        )

    def center_and_scale(self) -> tuple[float, float]:
        r"""Estimate a time-domain center and scale from the filtered spectrum.

        This computes a temporary time-domain signal via an FFT-based inverse transform,
        then estimates the mean and standard deviation of :math:`|\zeta(t)|^2`.

        Returns
        -------
        center:
            Center time.
        scale:
            Characteristic scale.

        """
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
        r"""Evaluate the filtered spectrum :math:`Z_\mathrm{out}(\omega)`.

        Parameters
        ----------
        w:
            Frequency grid.

        Returns
        -------
        RCArray
            Complex samples of :math:`Z_\mathrm{out}(\omega)`.

        """
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
        return cast(RCArray, out)

    def time_eval(self, t: FloatArray) -> RCArray:
        r"""Evaluate the filtered time-domain field :math:`\zeta_\mathrm{out}(t)`.

        The method:
        1. Builds a centered frequency grid around :math:`\omega_0`.
        2. Samples :math:`Z_\mathrm{out}(\omega)` on that grid.
        3. Applies an FFT-based inverse transform to obtain a centered time grid.
        4. Re-applies the carrier :math:`e^{i\omega_0 t}` and interpolates onto ``t``.

        Parameters
        ----------
        t:
            Time grid.

        Returns
        -------
        RCArray
            Complex samples of :math:`\zeta_\mathrm{out}(t)`.

        """
        t = np.asarray(t, dtype=float)

        w0 = float(getattr(self.base, "omega0", 0.0))

        _, sigma_w = _estimate_spectral_window(
            self.base, w0_fallback=w0, sigma_w_fallback=1.0
        )
        W_width = float(self.w_span_sigma) * float(sigma_w)

        phase_step = 0.2
        W_carrier = (np.pi * abs(w0)) / max(phase_step, 1e-12)

        W = max(W_width, W_carrier, 1e-12)

        n = int(self.n_fft)
        if n < 64:
            raise ValueError(f"n_fft too small: {n}")
        if not np.isfinite(W) or W <= 0.0:
            raise ValueError(f"Invalid frequency window W={W}")

        dw = 2.0 * W / float(n)
        k = np.arange(n, dtype=float) - (n // 2)
        w_rel = k * dw
        w = w0 + w_rel

        Z = np.asarray(self.freq_eval(w), dtype=complex)

        t0 = _fit_linear_phase_delay(w_rel, Z, frac=0.2)
        if abs(t0) > 0.0:
            Z = _remove_linear_phase(w_rel, Z, t0)

        if not np.isfinite(Z).all():
            bad = np.argwhere(~np.isfinite(Z))
            i = int(bad[0, 0])
            raise ValueError(f"Non-finite freq_eval at w={w[i]}: {Z[i]!r}")

        Z0 = np.fft.ifftshift(Z)

        z_rel = np.fft.ifft(Z0) * (n * dw) / (2.0 * np.pi)
        z_rel = np.fft.fftshift(z_rel)

        dt = 2.0 * np.pi / (float(n) * dw)
        t_grid = k * dt

        z = z_rel * np.exp(1j * w0 * t_grid)

        return _interp_complex_1d(t, t_grid, z)

    def delayed(self, dt: float) -> FilteredEnvelope:
        """Return a delayed copy of this filtered envelope.

        Parameters
        ----------
        dt:
            Delay applied to the base envelope.

        Returns
        -------
        FilteredEnvelope
            Delayed filtered envelope.

        """
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
        """Return a phased copy of this filtered envelope.

        Parameters
        ----------
        dphi:
            Phase increment applied to the base envelope.

        Returns
        -------
        FilteredEnvelope
            Phased filtered envelope.

        """
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
        """LaTeX (mathtext) representation of the defining spectral relation."""
        return r"Z_{\mathrm{out}}(\omega)=H(\omega)\,Z_{\mathrm{in}}(\omega)"

    def overlap_with_generic(self, other: EnvelopeProto) -> complex:
        r"""Compute overlap using a frequency-domain quadrature.

        This approximates

        .. math::

            \langle \zeta_1, \zeta_2 \rangle
            \;\approx\;
            \frac{1}{2\pi}\int \overline{Z_1(\omega)}\,Z_2(\omega)\,d\omega,

        using a finite window and trapezoidal quadrature.

        Parameters
        ----------
        other:
            Envelope to overlap with.

        Returns
        -------
        complex
            Approximate overlap.

        """
        w0_self, sigma_w_self = _estimate_spectral_window(
            self.base,
            w0_fallback=0.0,
            sigma_w_fallback=1.0,
        )
        w0_other, sigma_w_other = _estimate_spectral_window(
            other,
            w0_fallback=w0_self,
            sigma_w_fallback=sigma_w_self,
        )

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
