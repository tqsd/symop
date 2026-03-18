r"""Text rendering for monomials.

Provides a text-dispatch implementation for :class:`Monomial`, formatting
products of creation and annihilation operators in a readable,
terminal-friendly form.

The output is intended for debugging and inspection of symbolic CCR-based
operator expressions.
"""

from __future__ import annotations

from typing import Any

from symop.core.monomial import Monomial
from symop.viz._dispatch import text


def _join_ops_ascii(parts: list[str]) -> str:
    r"""Join operator parts into a single ASCII string.

    Parameters
    ----------
    parts:
        List of operator string representations.

    Returns
    -------
    str
        Space-separated operator string, or ``"I"`` if the list is empty.

    Notes
    -----
    The identity operator is represented by ``"I"`` when no operators
    are present.

    """
    if not parts:
        return "I"
    return " ".join(parts)


@text.register(Monomial)
def _text_monomial(obj: Monomial, /, **kwargs: Any) -> str:
    r"""Render a monomial as an ordered operator product.

    Parameters
    ----------
    obj:
        Monomial to render.
    **kwargs:
        Additional keyword arguments forwarded to the ``text`` dispatcher.

    Returns
    -------
    str
        String representation of the monomial as a sequence of operators,
        or ``"I"`` if the monomial is the identity.

    Notes
    -----
    - Creation operators are rendered first, followed by annihilation operators.
    - Individual operators are rendered via the ``text`` dispatcher.
    - The identity monomial is represented by ``"I"``.

    """
    if obj.is_identity:
        return "I"

    parts: list[str] = []
    for op in obj.creators:
        parts.append(text(op, **kwargs))
    for op in obj.annihilators:
        parts.append(text(op, **kwargs))
    return _join_ops_ascii(parts)
