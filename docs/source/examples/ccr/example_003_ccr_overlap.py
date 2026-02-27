r"""
Overlap-driven CCR commutators (LadderOp)
=========================================

This example illustrates the generalized CCR implemented at the ladder-operator level:

.. math::

    [a_i, a_j^\dagger] = <mode_i, mode_j>.

Since your mode overlap factorizes (path, polarization, envelope), you get
a clean demonstration that commutators "turn off" when modes are orthogonal.

We show:

1. Same mode: commutator is ~1.
2. Different path: commutator is 0.
3. Same path/pol but delayed envelope: commutator magnitude follows overlap.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from symop.core.operators import ModeOp
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel


def make_mode(path: str, *, tau: float = 0.0) -> ModeOp:
    env = GaussianEnvelope(omega0=10.0, sigma=0.4, tau=float(tau), phi0=0.0)
    lab = ModeLabel(
        path=PathLabel(path), pol=PolarizationLabel.H(), envelope=env
    )
    return ModeOp(label=lab)


# %%
# 1) Same mode: [a, adag] = 1
# ---------------------------
m = make_mode("A", tau=0.0)
c_same = m.ann.commutator(m.create)
assert abs(c_same - (1.0 + 0.0j)) <= 1e-12

# %%
# 2) Different path: commutator is 0
# ----------------------------------
mA = make_mode("A", tau=0.0)
mB = make_mode("B", tau=0.0)
c_path = mA.ann.commutator(mB.create)
assert abs(c_path - 0.0j) <= 1e-14

# %%
# 3) Envelope delay controls commutator magnitude
# ----------------------------------------------
taus = np.linspace(-3.0, 3.0, 400)
vals = []

ref = make_mode("A", tau=0.0)
for t in taus:
    shifted = make_mode("A", tau=float(t))
    vals.append(abs(ref.ann.commutator(shifted.create)) ** 2)

vals = np.array(vals)

fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(1, 1, 1)
ax.plot(taus, vals)
ax.set_title(r"Overlap-driven commutator: $|[a(t=0), a^\dagger(t=\tau)]|^2$")
ax.set_xlabel(r"$\tau$")
ax.set_ylabel("value")
ax.grid(alpha=0.3)
fig.tight_layout()

if __name__ == "__main__":
    plt.show()
