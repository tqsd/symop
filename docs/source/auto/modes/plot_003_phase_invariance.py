r"""
Global phase and overlap invariance
===================================

A global phase factor does not change the magnitude of an overlap:

.. math::

   \\langle f, e^{i\\phi} g \\rangle = e^{i\\phi} \\langle f, g \\rangle

so

.. math::

   \\left|\\langle f, e^{i\\phi} g \\rangle\\right|
   = \\left|\\langle f, g \\rangle\\right|.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from symop.modes.envelopes.gaussian import GaussianEnvelope


def build() -> object:
    e1 = GaussianEnvelope(omega0=12.0, sigma=0.35, tau=0.2, phi0=0.0)
    e2 = e1.delayed(0.3)

    base = e1.overlap(e2)
    print("base overlap:", base)
    print("base magnitude:", abs(base))

    phis = np.linspace(0.0, 2.0 * np.pi, 121)
    mags = np.empty_like(phis, dtype=float)
    args = np.empty_like(phis, dtype=float)

    for i, phi in enumerate(phis):
        ov = e1.overlap(e2.phased(float(phi)))
        mags[i] = abs(ov)
        args[i] = float(np.angle(ov))

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(phis, mags)
    ax1.set_xlabel("phi")
    ax1.set_ylabel("|overlap|")
    ax1.set_title("Magnitude is invariant under global phase")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(phis, np.unwrap(args))
    ax2.set_xlabel("phi")
    ax2.set_ylabel("arg(overlap) [rad]")
    ax2.set_title("Phase shifts linearly with phi")

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
