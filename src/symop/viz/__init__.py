r"""Visualization utilities for symbolic objects.

This package provides a unified interface for rendering and displaying
objects in different formats, including:

- Text (terminal-friendly representations)
- LaTeX (Jupyter/IPython rendering)
- Plots (matplotlib-based visualization)

Subpackages
-----------
text_renderer
    Plain-text renderers for symbolic objects.
latex_renderer
    LaTeX renderers for mathematical display.
plots
    Plotting backends for visualizing envelopes and related structures.

Modules
-------
_dispatch
    Core singledispatch-based rendering interface (``text``, ``latex``, ``plot``).
_optional
    Helpers for optional dependencies (e.g., matplotlib, IPython).
handlers
    Aggregates and registers all renderers via import side effects.

Notes
-----
- Rendering is type-driven via the dispatch system in ``_dispatch``.
- Submodules are typically imported for their side effects (handler registration).
- The package is designed for extensibility: new renderers can be added
  via ``*.register`` without modifying core logic.

"""

from __future__ import annotations

import matplotlib as mpl

from symop.viz import handlers  # noqa: F401
from symop.viz._dispatch import display, latex, plot, text

# Configure math rendering globally for the viz subsystem
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"


# Public API

__all__ = [
    "display",
    "plot",
    "latex",
    "text",
]
