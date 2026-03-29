# ruff: noqa: F401
r"""Visualization backend registration.

Aggregates and registers plot, LaTeX, and text renderers for symbolic
objects across the symop framework.

Importing this module ensures that all visualization handlers are
registered via side effects, enabling the dispatch system to resolve
appropriate implementations for:

- Envelopes and labels
- Operators and monomials
- Terms and polynomials
- State representations

Notes
-----
- Modules are imported for their side effects only (handler registration).
- Rendering is type-driven via the dispatch system in ``symop.viz._dispatch``.
- No public API is exposed; this module serves as a central registration hub.

"""

from __future__ import annotations

import symop.viz.plots.measurements as _plot_measurements
from symop.viz.latex_renderer import (
    density_poly_states as _latex_density_poly_states,
)
from symop.viz.latex_renderer import density_polys as _latex_density_polys
from symop.viz.latex_renderer import envelopes as _latex_envelopes
from symop.viz.latex_renderer import ket_poly_states as _latex_ket_poly_states
from symop.viz.latex_renderer import ket_polys as _latex_ket_polys
from symop.viz.latex_renderer import labels as _latex_labels
from symop.viz.latex_renderer import (
    measurement_results as _latex_measurement_results,
)
from symop.viz.latex_renderer import monomials as _latex_monomials
from symop.viz.latex_renderer import op_polys as _latex_op_polys
from symop.viz.latex_renderer import operators as _latex_operators
from symop.viz.latex_renderer import terms as _latex_terms
from symop.viz.plots import density_poly_states as _plot_density_poly_states
from symop.viz.plots import density_polys as _plot_density_polys
from symop.viz.plots import envelopes as _plot_envelopes
from symop.viz.plots import ket_poly_states as _plot_ket_poly_states
from symop.viz.plots import ket_polys as _plot_ket_polys
from symop.viz.plots import labels as _plot_labels
from symop.viz.plots import monomials as _plot_monomials
from symop.viz.plots import op_polys as _plot_op_polys
from symop.viz.plots import operators as _plot_operators
from symop.viz.plots import terms as _plot_terms
from symop.viz.text_renderer import (
    density_poly_states as _text_density_poly_states,
)
from symop.viz.text_renderer import density_polys as _text_density_polys
from symop.viz.text_renderer import envelopes as _text_envelopes
from symop.viz.text_renderer import ket_poly_states as _text_ket_poly_states
from symop.viz.text_renderer import ket_polys as _text_ket_polys
from symop.viz.text_renderer import labels as _text_labels
from symop.viz.text_renderer import (
    measurement_results as _text_measurement_results,
)
from symop.viz.text_renderer import monomials as _text_monomials
from symop.viz.text_renderer import op_polys as _text_op_polys
from symop.viz.text_renderer import operators as _text_operators
from symop.viz.text_renderer import terms as _text_terms

__all__: list[str] = []
