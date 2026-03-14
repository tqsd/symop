r"""
Filtering a Gaussian envelope with multiple transfer functions
==============================================================

This example demonstrates how several common spectral transfer functions
transform the same input Gaussian envelope.

Mathematics
-----------

A spectral filter is modeled as multiplication in the frequency domain:

.. math::

   Z_{\mathrm{out}}(\omega) = H(\omega)\,Z_{\mathrm{in}}(\omega),

with the corresponding time-domain field obtained by an inverse transform:

.. math::

   \zeta_{\mathrm{out}}(t)
   =
   \frac{1}{2\pi}\int_{-\infty}^{\infty} Z_{\mathrm{out}}(\omega)\,e^{-i\omega t}\,d\omega.

We compare input/output in time and frequency domains for a collection of
transfer functions:

- Gaussian low-pass / high-pass / band-pass
- Rectangular band-pass
- Super-Gaussian band-pass
- Constant phase :math:`e^{i\phi_0}`
- Time delay :math:`e^{-i\omega\tau}`
- Quadratic dispersion :math:`\exp\left(-i\frac{\beta_2}{2}(\omega-\omega_\mathrm{ref})^2\right)`
- Compositions: product and cascade

All transfer functions are callable on a frequency grid and provide a stable
``.signature`` for caching/comparison.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from symop.modes.envelopes.filtered import FilteredEnvelope
from symop.modes.envelopes.gaussian import GaussianEnvelope

from symop.modes.transfer.cascade import Cascade
from symop.modes.transfer.constant_phase import ConstantPhase
from symop.modes.transfer.gaussian_bandpass import GaussianBandpass
from symop.modes.transfer.gaussian_highpass import GaussianHighpass
from symop.modes.transfer.gaussian_lowpass import GaussianLowpass
from symop.modes.transfer.product import Product
from symop.modes.transfer.quadratic_dispersion import QuadraticDispersion
from symop.modes.transfer.rect_bandpass import RectBandpass
from symop.modes.transfer.supergaussian_bandpass import SuperGaussianBandpass
from symop.modes.transfer.time_delay import TimeDelay


def _normalize_time(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amp = np.abs(y)
    m = float(amp.max()) if amp.size else 0.0
    if m > 0.0:
        return y / m, amp / m
    return y, amp


def _normalize_pos(x: np.ndarray) -> np.ndarray:
    m = float(x.max()) if x.size else 0.0
    if m > 0.0:
        return x / m
    return x


# %%
# Setup
# -----
# Define an input Gaussian envelope :math:`\zeta_{\mathrm{in}}(t)` and a list of
# spectral transfer functions :math:`H(\omega)` to apply.

env_in = GaussianEnvelope(omega0=10.0, sigma=0.35, tau=0.0, phi0=0.0)

# Choose a frequency span and FFT parameters for the FilteredEnvelope backend.
# The output envelopes are constructed as:
#
# :math:`Z_{\mathrm{out}}(\omega) = H(\omega) Z_{\mathrm{in}}(\omega)`.
#
n_fft = 2**14
w_span_sigma = 12.0

w0 = float(env_in.omega0)

# Define base transfer functions (each one is a callable H(w)).
H_low = GaussianLowpass(w0=w0, sigma_w=2.0)
H_high = GaussianHighpass(w0=w0, sigma_w=2.0)
H_band = GaussianBandpass(w0=w0, sigma_w=1.0)
H_rect = RectBandpass(w0=w0, width=2.0)
H_sg = SuperGaussianBandpass(w0=w0, sigma_w=1.0, order=4)

# Phase-only elements (magnitude 1).
H_phi = ConstantPhase(phi0=0.7)
H_delay = TimeDelay(tau=0.35)
H_disp = QuadraticDispersion(beta2=0.08, w_ref=w0)

# Compositions.
# Product: :math:`H(\omega)=H_1(\omega)H_2(\omega)`.
H_prod = Product(a=H_band, b=H_delay)

# Cascade: :math:`H(\omega)=H_n(\omega)\cdots H_2(\omega)H_1(\omega)`.
H_casc = Cascade(parts=(H_low, H_delay, H_disp, H_phi))

transfers: list[tuple[str, object]] = [
    ("Gaussian low-pass", H_low),
    ("Gaussian high-pass", H_high),
    ("Gaussian band-pass", H_band),
    ("Rectangular band-pass", H_rect),
    ("Super-Gaussian band-pass", H_sg),
    ("Constant phase", H_phi),
    ("Time delay", H_delay),
    ("Quadratic dispersion", H_disp),
    ("Product: band-pass * delay", H_prod),
    ("Cascade: low-pass -> delay -> dispersion -> phase", H_casc),
]

# Construct all filtered envelopes.
env_outs: list[tuple[str, FilteredEnvelope]] = []
for name, H in transfers:
    env_outs.append(
        (
            name,
            FilteredEnvelope(
                base=env_in,
                transfer=H,  # type: ignore[arg-type]
                n_fft=n_fft,
                w_span_sigma=w_span_sigma,
            ),
        )
    )

# %%
# Basic diagnostics
# -----------------
# For each filter, compute the overlap :math:`\langle \mathrm{in} \mid \mathrm{out} \rangle`
# as a compact similarity metric.

print("=== overlaps <in|out> ===")
for name, env_out in env_outs:
    ov = env_in.overlap(env_out)
    print(f"{name:40s}  {ov!r}   |ov|={abs(ov):.6g}")

# %%
# Time-domain comparison (amplitude)
# ----------------------------------
# Plot the normalized time-domain amplitudes :math:`\lvert \zeta(t) \rvert` for input and each output.

c_in, s_in = env_in.center_and_scale()
c = float(c_in)
S = float(s_in)

t = np.linspace(c - 8.0 * S, c + 8.0 * S, 2500)

yin = env_in.time_eval(t)
yin_n, amp_in = _normalize_time(yin)

fig_t = plt.figure(figsize=(10, 7))
ax = fig_t.add_subplot(1, 1, 1)

ax.plot(t, amp_in, label=r"$|\zeta_{\mathrm{in}}(t)|$", linewidth=2.0)

for name, env_out in env_outs:
    yout = env_out.time_eval(t)
    _, amp_out = _normalize_time(yout)
    ax.plot(t, amp_out, label=name, linestyle="--")

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"amplitude (normalized)")
ax.set_title(
    r"Time-domain amplitude: $|\zeta_{\mathrm{out}}(t)|$ for various $H(\omega)$"
)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, ncol=2)
fig_t.tight_layout()

# %%
# Frequency-domain comparison (spectral intensity + filter magnitude)
# -------------------------------------------------------------------
# For each case, compare spectral intensities:
#
# - :math:`|Z_{\mathrm{in}}(\omega)|^2`
# - :math:`|Z_{\mathrm{out}}(\omega)|^2`
# - :math:`|H(\omega)|^2`
#
# We normalize each curve to its own maximum to compare shapes.

w = np.linspace(w0 - 10.0, w0 + 10.0, 2500)
Zin = env_in.freq_eval(w)
spec_in = _normalize_pos(np.abs(Zin) ** 2)

n_cases = len(env_outs)
fig_w = plt.figure(figsize=(12, 2.8 * n_cases))

for i, (name, env_out) in enumerate(env_outs, start=1):
    ax = fig_w.add_subplot(n_cases, 1, i)

    Zout = env_out.freq_eval(w)
    spec_out = _normalize_pos(np.abs(Zout) ** 2)

    # Grab transfer from the original list by matching name.
    H = None
    for n2, H2 in transfers:
        if n2 == name:
            H = H2
            break
    assert H is not None

    Hw = H(w)  # type: ignore[call-arg]
    spec_H = _normalize_pos(np.abs(Hw) ** 2)

    ax.plot(w, spec_in, label=r"$|Z_{\mathrm{in}}(\omega)|^2$", linewidth=2.0)
    ax.plot(
        w, spec_out, label=r"$|Z_{\mathrm{out}}(\omega)|^2$", linestyle="--"
    )
    ax.plot(w, spec_H, label=r"$|H(\omega)|^2$", linestyle=":")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"norm.")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=3)

fig_w.tight_layout()

if __name__ == "__main__":
    plt.show()
