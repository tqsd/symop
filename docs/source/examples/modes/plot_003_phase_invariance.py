r"""
Global phase: magnitude invariance and phase rotation
=====================================================

A global phase on one envelope rotates the complex overlap but does not change
its magnitude:

.. math::

   \langle f, e^{i\phi} g \rangle = e^{i\phi}\langle f, g \rangle,

so

.. math::

   \left|\langle f, e^{i\phi} g \rangle\right|
   =
   \left|\langle f, g \rangle\right|.

In this example we choose two partially overlapping Gaussian envelopes and sweep
a global phase :math:`\phi` applied to :math:`g`.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from symop.modes.envelopes.gaussian import GaussianEnvelope


def build() -> object:
    # Choose two envelopes with nontrivial (not ~1) overlap.
    f = GaussianEnvelope(omega0=12.0, sigma=0.35, tau=0.0, phi0=0.0)
    g = f.delayed(0.6)

    ov0 = f.overlap(g)
    print("base overlap <f|g>:", ov0)
    print("base magnitude |<f|g>|:", abs(ov0))
    print("base phase arg(<f|g>):", float(np.angle(ov0)))

    phis = np.linspace(0.0, 2.0 * np.pi, 181)
    mags = np.empty_like(phis, dtype=float)
    args = np.empty_like(phis, dtype=float)

    for i, phi in enumerate(phis):
        ov = f.overlap(g.phased(float(phi)))
        mags[i] = abs(ov)
        args[i] = float(np.angle(ov))

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(phis, mags)
    ax1.set_xlabel(r"$\phi$")
    ax1.set_ylabel(r"$\left|\langle f, e^{i\phi} g\rangle\right|$")
    ax1.set_title(r"Magnitude is invariant under a global phase")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(phis, np.unwrap(args))
    ax2.set_xlabel(r"$\phi$")
    ax2.set_ylabel(
        r"$\mathrm{unwrap}\,\arg\!\left(\langle f, e^{i\phi} g\rangle\right)$"
    )
    ax2.set_title(r"Overlap phase shifts linearly with $\phi$")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def main() -> None:
    fig = build()
    plt.show()


if __name__ == "__main__":
    main()
