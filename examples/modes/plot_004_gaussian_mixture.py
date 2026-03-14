r"""
Gaussian mixture: envelope shape, overlap vs delay, and spectrum
================================================================

We construct a small mixture of normalized Gaussian envelopes and study:

1) The time-domain envelope magnitude compared to a reference component.
2) How the overlap magnitude decays with temporal delay.
3) The spectral intensity profile of the mixture.

Mathematics
-----------

A Gaussian mixture envelope is

.. math::

   \zeta(t) = \sum_{k=1}^{K} c_k g_k(t),

with normalized components :math:`\langle g_k, g_k \rangle = 1`.
The mixture is normalized so that

.. math::

   \langle \zeta, \zeta \rangle = 1.

The overlap is defined as

.. math::

   \langle f_1, f_2 \rangle
   =
   \int_{-\infty}^{\infty} f_1(t)^* f_2(t)\,dt.

We also inspect the spectral intensity derived from the Fourier transform

.. math::

   Z(\omega)
   =
   \int_{-\infty}^{\infty} \zeta(t)\,e^{+i\omega t}\,dt,

and its intensity

.. math::

   I(\omega) = |Z(\omega)|^2.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope

# %%
# Setup
# -----
# Build a small mixture. The constructor enforces normalization based on
# closed-form overlaps, so the provided weights will be rescaled.

g0 = GaussianEnvelope(omega0=10.0, sigma=0.35, tau=0.00, phi0=0.0)
g1 = GaussianEnvelope(omega0=10.0, sigma=0.45, tau=0.35, phi0=0.3)
g2 = GaussianEnvelope(omega0=10.0, sigma=0.40, tau=-0.25, phi0=-0.2)

components = (g0, g1, g2)
weights = np.array([1.0 + 0.0j, 0.6 - 0.2j, 0.4 + 0.1j], dtype=complex)

mix = GaussianMixtureEnvelope(components=components, weights=weights)

print("mixture norm2:", float(mix.norm2()))
print("component 0 norm2:", float(g0.norm2()))

# %%
# Plot time-domain envelope
# -------------------------
# Use the BaseEnvelope plotting utility. We overlay the mixture and a reference
# Gaussian component. Normalization in plotting is optional; here we show raw
# amplitudes (the mixture is normalized in the inner-product sense, not max|.|).

fig_env, axs_env = g0.plot_many(
    [g0, mix],
    title="Time-domain envelopes: reference Gaussian vs Gaussian mixture",
    labels=["g0", "mix"],
    show_real_imag=False,
    show_phase=False,
    show_formula=True,
    normalize_envelope=False,
)

# %%
# Overlap magnitude vs delay
# --------------------------
# Evaluate |<mix, mix(delayed)>| as a function of delay.

delays = np.linspace(-2.5, 2.5, 201)
mags = np.empty_like(delays, dtype=float)

for i, dt in enumerate(delays):
    mags[i] = abs(mix.overlap(mix.delayed(float(dt))))

print("max |overlap|:", float(mags.max()))
print("min |overlap|:", float(mags.min()))

fig_ov = plt.figure(figsize=(8, 4))
ax_ov = fig_ov.add_subplot(1, 1, 1)
ax_ov.plot(delays, mags)
ax_ov.set_xlabel(r"$\Delta t$")
ax_ov.set_ylabel(r"$\left|\langle \zeta(t), \zeta(t-\Delta t)\rangle\right|$")
ax_ov.set_title(
    "Overlap magnitude vs temporal delay (mixture vs delayed mixture)"
)
ax_ov.grid(alpha=0.3)
fig_ov.tight_layout()

# %%
# Spectrum
# --------
# Compute the spectral intensity |Z(w)|^2 for the mixture.
# Use a window around the common carrier.

w0 = float(g0.omega0)
w = np.linspace(w0 - 12.0, w0 + 12.0, 2500)
Z = mix.freq_eval(w)
spec = np.abs(Z) ** 2

fig_sp = plt.figure(figsize=(8, 4))
ax_sp = fig_sp.add_subplot(1, 1, 1)
ax_sp.plot(w, spec)
ax_sp.set_xlabel(r"$\omega$")
ax_sp.set_ylabel(r"$\left|Z(\omega)\right|^2$")
ax_sp.set_title(r"Spectral intensity profile of the Gaussian mixture")
ax_sp.grid(alpha=0.3)
fig_sp.tight_layout()

if __name__ == "__main__":
    plt.show()
