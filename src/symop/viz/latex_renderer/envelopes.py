"""LaTeX rendering for temporal envelope objects.

This module provides LaTeX dispatcher implementations for Gaussian-based
temporal envelopes. Each envelope may optionally define a custom LaTeX
representation via a ``latex`` attribute. If not provided, a default
symbolic placeholder is used.

The renderers are intentionally minimal and avoid embedding detailed
functional forms, delegating such responsibility to higher-level or
user-defined representations when needed.
"""

from __future__ import annotations

from typing import Any

from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope
from symop.viz._dispatch import latex


@latex.register
def _(env: GaussianEnvelope, /, **kwargs: Any) -> str:
    r"""Render a :class:`GaussianEnvelope` as a LaTeX string.

    Parameters
    ----------
    env:
        Gaussian envelope instance to render.
    **kwargs:
        Additional keyword arguments (currently unused, accepted for
        compatibility with the dispatcher interface).

    Returns
    -------
    str
        LaTeX representation of the envelope.

    Notes
    -----
    - If the envelope defines a non-empty ``latex`` attribute, it is used
      directly.
    - Otherwise, a default placeholder ``\\zeta(t)`` is returned.

    """
    s = getattr(env, "latex", None)
    if isinstance(s, str) and s:
        return s
    return r"\zeta(t)"


@latex.register
def _(env: GaussianMixtureEnvelope, /, **kwargs: Any) -> str:
    r"""Render a :class:`GaussianMixtureEnvelope` as a LaTeX string.

    Parameters
    ----------
    env:
        Gaussian mixture envelope instance to render.
    **kwargs:
        Additional keyword arguments (currently unused, accepted for
        compatibility with the dispatcher interface).

    Returns
    -------
    str
        LaTeX representation of the envelope.

    Notes
    -----
    - If the envelope defines a non-empty ``latex`` attribute, it is used
      directly.
    - Otherwise, a default placeholder ``\\zeta(t)`` is returned.

    """
    s = getattr(env, "latex", None)
    if isinstance(s, str) and s:
        return s
    return r"\zeta(t)"
