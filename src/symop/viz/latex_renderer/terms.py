"""LaTeX rendering for polynomial term objects.

This module provides LaTeX dispatcher implementations for
:class:`KetTerm` and :class:`DensityTerm`. These term-level renderers
form an intermediate layer between low-level monomial rendering and
higher-level polynomial rendering.

Each term is rendered by formatting its coefficient together with its
monomial or operator-kernel body.
"""

from __future__ import annotations

from typing import Any

from symop.core.terms import DensityTerm, KetTerm
from symop.viz._dispatch import latex
from symop.viz.latex_renderer._latex_utils import (
    apply_coeff,
    latex_config_from_kwargs,
)


def _latex_monomial_or_identity(obj: Any, **kwargs: Any) -> str:
    r"""Render an object as LaTeX, defaulting to the identity operator.

    Parameters
    ----------
    obj:
        Object to render via the global ``latex`` dispatcher.
    **kwargs:
        Additional keyword arguments forwarded to the dispatcher.

    Returns
    -------
    str
        LaTeX representation of ``obj`` if non-empty, otherwise
        ``\\mathbb{I}``.

    Notes
    -----
    This helper is used to ensure that empty monomial-like renderings
    are interpreted as the identity operator.

    """
    s = latex(obj, **kwargs)
    return s if s else "\\mathbb{I}"


@latex.register(KetTerm)
def _latex_ket_term(obj: KetTerm, /, **kwargs: Any) -> str:
    r"""Render a :class:`KetTerm` as a LaTeX string.

    Parameters
    ----------
    obj:
        Ket term to render.
    **kwargs:
        Additional keyword arguments forwarded to the LaTeX rendering
        pipeline. These may include formatting options such as
        ``decimals`` and ``eps``.

    Returns
    -------
    str
        LaTeX representation of the ket term.

    Notes
    -----
    The ket term is rendered as its coefficient multiplied by its
    monomial body. If the monomial renders to an empty string, it is
    interpreted as the identity operator ``\\mathbb{I}``.

    """
    cfg = latex_config_from_kwargs(kwargs)

    c = complex(obj.coeff)
    m = _latex_monomial_or_identity(obj.monomial, **kwargs)

    return apply_coeff(c, m, decimals=cfg.decimals, empty_body="\\mathbb{I}")


@latex.register(DensityTerm)
def _latex_density_term(obj: DensityTerm, /, **kwargs: Any) -> str:
    r"""Render a :class:`DensityTerm` as a LaTeX string.

    Parameters
    ----------
    obj:
        Density term to render.
    **kwargs:
        Additional keyword arguments forwarded to the LaTeX rendering
        pipeline. These may include formatting options such as
        ``decimals`` and ``eps``.

    Returns
    -------
    str
        LaTeX representation of the density term.

    Notes
    -----
    The density term is rendered in kernel-like form as

    .. math::

        L \, (\cdot) \, R

    where ``L`` and ``R`` are the rendered left and right monomials.
    Empty monomial renderings are interpreted as the identity operator
    ``\\mathbb{I}``.

    """
    cfg = latex_config_from_kwargs(kwargs)

    c = complex(obj.coeff)
    L = _latex_monomial_or_identity(obj.left, **kwargs)
    R = _latex_monomial_or_identity(obj.right, **kwargs)

    core = rf"{L}\,(\cdot)\,{R}"
    return apply_coeff(c, core, decimals=cfg.decimals, empty_body="\\mathbb{I}")
