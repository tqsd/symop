r"""
Overlap magnitude vs delay and spectral profile
===============================================

We study how the magnitude of the overlap between two identical Gaussian
envelopes decays with time delay, and we also inspect the corresponding
spectral intensity profile.

Mathematics
-----------

The overlap is defined as

.. math::

   \langle f_1, f_2 \rangle
   =
   \int f_1(t)^* f_2(t)\,dt.

We also plot the spectral intensity derived from the Fourier transform

.. math::

   Z(\omega)
   =
   \int_{-\infty}^{\infty} \zeta(t)\,e^{+i\omega t}\,dt,

and its intensity

.. math::

   I(\omega) = \lvert Z(\omega)\rvert^2.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from symop.modes.envelopes.gaussian import GaussianEnvelope

# %%
# Setup
# -----
# Create a base Gaussian envelope and evaluate overlaps against delayed copies.

base = GaussianEnvelope(omega0=10.0, sigma=0.4, tau=0.0, phi0=0.0)

delays = np.linspace(-2.5, 2.5, 151)
mags = np.empty_like(delays, dtype=float)

for i, dt in enumerate(delays):
    mags[i] = abs(base.overlap(base.delayed(float(dt))))

print("max |overlap|:", float(mags.max()))
print("min |overlap|:", float(mags.min()))

# %%
# Spectrum
# --------
# Compute the spectral intensity :math:`\lvert Z(\omega)\rvert^2`.

w = np.linspace(base.omega0 - 10.0, base.omega0 + 10.0, 2000)
Z = base.freq_eval(w)
spec = np.abs(Z) ** 2

# %%
# Plot overlap
# ------------
# First figure: overlap magnitude vs delay.

fig1 = plt.figure(figsize=(8, 4))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(delays, mags)
ax1.set_xlabel(r"$\Delta t$")
ax1.set_ylabel(r"$\left|\langle \zeta(t), \zeta(t-\Delta t) \rangle\right|$")
ax1.set_title(r"Overlap magnitude vs temporal delay")
ax1.grid(alpha=0.3)
fig1.tight_layout()

# %%
# Plot spectrum
# -------------
# Second figure: spectral intensity profile.

fig2 = plt.figure(figsize=(8, 4))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(w, spec)
ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel(r"$\left|Z(\omega)\right|^2$")
ax2.set_title(r"Spectral intensity profile of $\zeta(t)$")
ax2.grid(alpha=0.3)
fig2.tight_layout()

if __name__ == "__main__":
    plt.show()
