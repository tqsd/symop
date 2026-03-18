r"""Text rendering for Gaussian envelope types.

Provides text-dispatch implementations for Gaussian-based temporal
envelopes, including single Gaussian envelopes and Gaussian mixtures.

The output is intended for debugging and inspection, exposing key
parameters such as central frequency, width, delay, phase, and mixture
structure.
"""

from __future__ import annotations

from typing import Any

from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope
from symop.viz._dispatch import text


@text.register
def _(env: GaussianEnvelope, /, **kwargs: Any) -> str:
    r"""Render a Gaussian envelope as a text representation.

    Parameters
    ----------
    env:
        Gaussian envelope instance.
    **kwargs:
        Additional keyword arguments (ignored).

    Returns
    -------
    str
        String representation including central frequency ``omega0``,
        temporal width ``sigma``, delay ``tau``, and phase offset ``phi0``.

    Notes
    -----
    All parameters are converted to ``float`` for stable and concise output.

    """
    return (
        "GaussianEnvelope("
        f"omega0={float(env.omega0)!r}, "
        f"sigma={float(env.sigma)!r}, "
        f"tau={float(env.tau)!r}, "
        f"phi0={float(env.phi0)!r}"
        ")"
    )


@text.register
def _(env: GaussianMixtureEnvelope, /, **kwargs: Any) -> str:
    r"""Render a Gaussian mixture envelope as a text summary.

    Parameters
    ----------
    env:
        Gaussian mixture envelope instance.
    **kwargs:
        Additional keyword arguments (ignored).

    Returns
    -------
    str
        Summary string including the number of components ``K`` and the
        shape of the mixture weights array.

    Notes
    -----
    This representation is intentionally compact and does not expand
    individual mixture components.

    """
    k = len(env.components)
    return f"GaussianMixtureEnvelope(K={k}, weights_shape={tuple(env.weights.shape)!r})"
