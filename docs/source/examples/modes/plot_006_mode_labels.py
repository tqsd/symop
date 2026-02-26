r"""
Mode labels: envelope, polarization, and path
=============================================

This example demonstrates how a full mode label consists of:

- Path label
- Polarization label
- Envelope

The overlap factorizes as:

.. math::

   \langle m_1, m_2 \rangle =
   \langle p_1, p_2 \rangle
   \langle \pi_1, \pi_2 \rangle
   \langle \zeta_1, \zeta_2 \rangle.

We show:

1. Envelope overlap under time shift.
2. Polarization overlap under rotation.
3. Path orthogonality.
4. Combined ModeLabel overlap.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel
from symop.modes.envelopes.gaussian import GaussianEnvelope


# %%
# Reference components
# --------------------

path_A = PathLabel("A")
path_B = PathLabel("B")

pol_H = PolarizationLabel.H()

env_ref = GaussianEnvelope(omega0=10.0, sigma=0.4, tau=0.0, phi0=0.0)

mode_ref = ModeLabel(path=path_A, pol=pol_H, envelope=env_ref)


# %%
# 1) Envelope overlap under time shift
# -------------------------------------

taus = np.linspace(-3.0, 3.0, 400)
env_overlap = []

for tau in taus:
    env_shift = GaussianEnvelope(
        omega0=10.0, sigma=0.4, tau=float(tau), phi0=0.0
    )
    env_overlap.append(abs(env_ref.overlap(env_shift)) ** 2)

env_overlap = np.array(env_overlap)


# %%
# 2) Polarization overlap under rotation
# ---------------------------------------

thetas = np.linspace(0.0, 2.0 * np.pi, 400)
pol_overlap = []

for th in thetas:
    pol_rot = PolarizationLabel.linear(float(th))
    pol_overlap.append(abs(pol_H.overlap(pol_rot)) ** 2)

pol_overlap = np.array(pol_overlap)


# %%
# 3) Mode overlap including path
# -------------------------------

mode_same_path = []
mode_diff_path = []

for th in thetas:
    pol_rot = PolarizationLabel.linear(float(th))

    mode_same = ModeLabel(
        path=path_A,
        pol=pol_rot,
        envelope=env_ref,
    )

    mode_diff = ModeLabel(
        path=path_B,  # different path
        pol=pol_rot,
        envelope=env_ref,
    )

    mode_same_path.append(abs(mode_ref.overlap(mode_same)) ** 2)
    mode_diff_path.append(abs(mode_ref.overlap(mode_diff)) ** 2)

mode_same_path = np.array(mode_same_path)
mode_diff_path = np.array(mode_diff_path)


# %%
# Plotting
# --------

fig = plt.figure(figsize=(10, 9))

# Envelope overlap
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(taus, env_overlap)
ax1.set_title(r"Envelope overlap $|\langle \zeta(0) | \zeta(\tau)\rangle|^2$")
ax1.set_xlabel(r"$\tau$")
ax1.set_ylabel("overlap")
ax1.grid(alpha=0.3)

# Polarization overlap
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(thetas, pol_overlap)
ax2.set_title(r"Polarization overlap $|\langle H | \pi(\theta)\rangle|^2$")
ax2.set_xlabel(r"$\theta$")
ax2.set_ylabel("overlap")
ax2.grid(alpha=0.3)

# Mode overlap (path effect)
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(thetas, mode_same_path, label="same path")
ax3.plot(thetas, mode_diff_path, label="different path", linestyle="--")
ax3.set_title("Mode overlap including path label")
ax3.set_xlabel(r"$\theta$")
ax3.set_ylabel("overlap")
ax3.grid(alpha=0.3)
ax3.legend()

fig.tight_layout()

if __name__ == "__main__":
    plt.show()
