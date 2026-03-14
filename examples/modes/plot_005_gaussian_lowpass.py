r"""
Filtering a Gaussian envelope in the frequency domain
=====================================================

This example demonstrates how a spectral filter transforms an envelope.

Mathematics
-----------

A filter is modeled as multiplication in the frequency domain:

.. math::

   Z_{\mathrm{out}}(\omega) = H(\omega)\,Z_{\mathrm{in}}(\omega),

with the corresponding time-domain field obtained by an inverse transform:

.. math::

   \zeta_{\mathrm{out}}(t)
   =
   \frac{1}{2\pi}\int_{-\infty}^{\infty} Z_{\mathrm{out}}(\omega)\,e^{-i\omega t}\,d\omega.

We compare the input and output envelopes in both time and frequency domains.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from symop.modes.envelopes.filtered import FilteredEnvelope
from symop.modes.envelopes.gaussian import GaussianEnvelope

# Adjust this import to your actual transfer-function implementation.
# The transfer must be callable: H(w) -> complex array, and have .signature.
from symop.modes.transfer.gaussian_lowpass import GaussianLowpass

# %%
# Setup
# -----
# Define an input Gaussian envelope and a Gaussian low-pass filter.

env_in = GaussianEnvelope(omega0=10.0, sigma=0.35, tau=0.0, phi0=0.0)

# Example parameters (adapt to your class signature):
# center: filter center frequency
# sigma_w: spectral width (rad/s)
H = GaussianLowpass(w0=10.0, sigma_w=2.0)

env_out = FilteredEnvelope(
    base=env_in, transfer=H, n_fft=2**14, w_span_sigma=12.0
)

# %%
# Basic diagnostics
# -----------------
# Overlap of the filtered envelope with the original.

ov = env_in.overlap(env_out)
print("overlap <in|out>:", ov)
print("magnitude |<in|out>|:", abs(ov))

# %%
# Time-domain comparison
# ----------------------
# Plot amplitude and real/imag for both envelopes on a common time grid.

c_in, s_in = env_in.center_and_scale()
c_out, s_out = env_out.center_and_scale()
c = 0.5 * (c_in + c_out)
S = max(s_in, s_out)
t = np.linspace(c - 8.0 * S, c + 8.0 * S, 2000)

yin = env_in.time_eval(t)
yout = env_out.time_eval(t)

amp_in = np.abs(yin)
amp_out = np.abs(yout)

if amp_in.max() > 0:
    yin = yin / amp_in.max()
    amp_in = amp_in / amp_in.max()
if amp_out.max() > 0:
    yout = yout / amp_out.max()
    amp_out = amp_out / amp_out.max()

fig1 = plt.figure(figsize=(8, 6))

ax = fig1.add_subplot(2, 1, 1)
ax.plot(t, amp_in, label=r"$|\zeta_{\mathrm{in}}(t)|$")
ax.plot(t, amp_out, label=r"$|\zeta_{\mathrm{out}}(t)|$", linestyle="--")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"amplitude (normalized)")
ax.set_title(r"Time-domain envelopes (before vs after filtering)")
ax.grid(alpha=0.3)
ax.legend()

ax = fig1.add_subplot(2, 1, 2)
ax.plot(t, np.real(yin), label=r"$\Re\,\zeta_{\mathrm{in}}(t)$")
ax.plot(
    t, np.imag(yin), label=r"$\Im\,\zeta_{\mathrm{in}}(t)$", linestyle="--"
)
ax.plot(t, np.real(yout), label=r"$\Re\,\zeta_{\mathrm{out}}(t)$")
ax.plot(
    t, np.imag(yout), label=r"$\Im\,\zeta_{\mathrm{out}}(t)$", linestyle="--"
)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"field (normalized)")
ax.grid(alpha=0.3)
ax.legend(ncol=2)

fig1.tight_layout()

# %%
# Frequency-domain comparison
# ---------------------------
# Plot spectral intensity profiles :math:`\lvert Z(\omega)\rvert|^2` for input/output, and the filter |H(omega)|.

w = np.linspace(env_in.omega0 - 10.0, env_in.omega0 + 10.0, 2000)

Zin = env_in.freq_eval(w)
Zout = env_out.freq_eval(w)
Hw = H(w)

spec_in = np.abs(Zin) ** 2
spec_out = np.abs(Zout) ** 2
spec_H = np.abs(Hw) ** 2

# Normalize intensities for visual comparison.
if spec_in.max() > 0:
    spec_in = spec_in / spec_in.max()
if spec_out.max() > 0:
    spec_out = spec_out / spec_out.max()
if spec_H.max() > 0:
    spec_H = spec_H / spec_H.max()

fig2 = plt.figure(figsize=(8, 6))

ax = fig2.add_subplot(2, 1, 1)
ax.plot(w, spec_in, label=r"$|Z_{\mathrm{in}}(\omega)|^2$")
ax.plot(w, spec_out, label=r"$|Z_{\mathrm{out}}(\omega)|^2$", linestyle="--")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"intensity (normalized)")
ax.set_title(r"Spectral intensity (before vs after filtering)")
ax.grid(alpha=0.3)
ax.legend()

ax = fig2.add_subplot(2, 1, 2)
ax.plot(w, spec_H, label=r"$|H(\omega)|^2$")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"transmission (normalized)")
ax.set_title(r"Filter magnitude response")
ax.grid(alpha=0.3)
ax.legend()

fig2.tight_layout()

if __name__ == "__main__":
    plt.show()
