"""LaTeX rendering for mode-label metadata.

This module provides LaTeX dispatcher implementations for path labels,
polarization labels, and full mode labels. These renderers are intended
to provide compact, human-readable symbolic representations for metadata
attached to modes.

Complex amplitudes appearing in polarization Jones vectors are formatted
using a small local helper.
"""

from __future__ import annotations

from typing import Any

from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import Path
from symop.modes.labels.polarization import Polarization
from symop.viz._dispatch import latex
from symop.viz.latex_renderer._latex_utils import fmt_complex_latex


@latex.register
def _(p: Path, /, **kwargs: Any) -> str:
    r"""Render a :class:`Path` as a LaTeX string.

    Parameters
    ----------
    p:
        Path label to render.
    **kwargs:
        Additional keyword arguments accepted for dispatcher compatibility.
        They are currently unused.

    Returns
    -------
    str
        LaTeX representation of the path.

    Notes
    -----
    The path name is rendered in a simple functional form:

    .. math::

        \mathrm{path}(\cdot)

    Escaping or sanitization of path names is not currently performed.

    """
    return r"\mathrm{path}(" + str(p.name) + ")"


@latex.register
def _(pol: Polarization, /, **kwargs: Any) -> str:
    """Render a :class:`Polarization` as a Jones vector in LaTeX form.

    Parameters
    ----------
    pol:
        Polarization label to render.
    **kwargs:
        Additional keyword arguments accepted for dispatcher compatibility.
        They are currently unused.

    Returns
    -------
    str
        LaTeX representation of the polarization as a column vector.

    Notes
    -----
    The Jones coefficients are formatted using
    :func:`_fmt_complex_latex` and displayed as a two-component column
    vector.

    """
    a, b = pol.jones
    a_s = fmt_complex_latex(a)
    b_s = fmt_complex_latex(b)
    return r"\begin{pmatrix}" + a_s + r"\\" + b_s + r"\end{pmatrix}"


@latex.register
def _(m: ModeLabel, /, **kwargs: Any) -> str:
    """Render a :class:`ModeLabel` as a composite LaTeX string.

    Parameters
    ----------
    m:
        Mode label to render.
    **kwargs:
        Additional keyword arguments accepted for dispatcher compatibility.
        They are currently unused by this function directly.

    Returns
    -------
    str
        LaTeX representation of the mode label.

    Notes
    -----
    The mode label is rendered compositionally from its constituent
    parts:

    - path
    - polarization
    - envelope

    Each component is rendered via the global ``latex`` dispatcher.
    The resulting expression is intentionally compact to avoid overly
    large labels in rendered output.

    """
    return (
        r"\langle \mathrm{mode} \rangle:"
        + r"\ \ "
        + latex(m.path)
        + r"\ \otimes\ "
        + latex(m.polarization)
        + r"\ \otimes\ "
        + latex(m.envelope)
    )
