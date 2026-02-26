r"""
Gaussian envelopes: delay, phase, and overlap
=============================================

This example introduces :class:`symop.modes.envelopes.gaussian.GaussianEnvelope`.

We build one Gaussian time-domain field :math:`\\zeta(t)` and derive a second one
by delaying and phasing it. We then:

- compute the overlap (inner product) numerically
- plot both envelopes on the same time grid
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from symop.modes.envelopes.gaussian import GaussianEnvelope


def build() -> object:
    e1 = GaussianEnvelope(omega0=10.0, sigma=0.3, tau=0.0, phi0=0.0)
    e2 = e1.delayed(0.4).phased(0.8)

    ov = e1.overlap(e2)
    print("overlap:", ov)
    print("abs(overlap):", abs(ov))

    fig, _ = GaussianEnvelope.plot_many(
        [e1, e2],
        show_phase=True,
        normalize_envelope=True,
        title="Two Gaussian envelopes (delay + phase)",
    )
    return fig


def main() -> None:
    fig = build()
    plt.show()


if __name__ == "__main__":
    main()
