"""
Overlap magnitude vs delay
=========================

The overlap of two time envelopes is

.. math::

   \\langle f_1, f_2 \\rangle = \\int f_1(t)^* f_2(t)\\,dt.

For identical Gaussians, delaying one of them reduces the overlap magnitude.
This is the key mechanism behind treating sufficiently separated pulses as
approximately orthogonal modes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from symop.modes.envelopes.gaussian import GaussianEnvelope


def build() -> object:
    base = GaussianEnvelope(omega0=10.0, sigma=0.4, tau=0.0, phi0=0.0)

    delays = np.linspace(-2.5, 2.5, 151)
    mags = np.empty_like(delays, dtype=float)

    for i, dt in enumerate(delays):
        shifted = base.delayed(float(dt))
        mags[i] = abs(base.overlap(shifted))

    print("max |overlap|:", float(mags.max()))
    print("min |overlap|:", float(mags.min()))

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(delays, mags)
    ax.set_xlabel("delay dt")
    ax.set_ylabel("|<e(t), e(t-dt)>|")
    ax.set_title("Gaussian overlap magnitude vs delay")
    fig.tight_layout()
    return fig


def main() -> None:
    fig = build()

    out = Path(__file__).with_suffix(".png")
    try:
        fig.savefig(out, dpi=150)
        print("saved figure to:", out)
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt

        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
