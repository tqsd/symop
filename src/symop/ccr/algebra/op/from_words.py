"""Construct operator terms from ladder-operator words.

This module provides a small helper that materializes operator words into
tuples of ladder operators and pairs them with coefficients. The resulting
term objects are created via a caller-provided factory, which keeps this
function independent of concrete term implementations and avoids import
cycles.

In the operator-algebra view, a single term corresponds to a "word" in ladder
operators with a scalar prefactor:

.. math::

    T = c * o_1 o_2 ... o_L,

where each :math:`o_k` is a ladder operator instance.
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.ccr._typing import OpTermFactory, OpTermT
from symop.core.protocols.ops.operators import LadderOp


def from_words(
    words: Iterable[Iterable[LadderOp]],
    coeffs: Iterable[complex] | None = None,
    *,
    term_factory: OpTermFactory[OpTermT],
) -> tuple[OpTermT, ...]:
    """Build operator terms from raw operator words and coefficients.

    Each input word is materialized into a tuple of ladder operators and paired
    with a coefficient. The output term objects are constructed using the
    provided ``term_factory``.

    Parameters
    ----------
    words:
        Iterable of operator words, each a sequence of ladder operators.
    coeffs:
        Optional coefficients. If omitted, each word gets coefficient ``1.0``.
        If provided, the number of coefficients must match the number of words.
    term_factory:
        Callable constructing a term from ``(ops, coeff)``.

    Returns
    -------
    tuple[OpTermT, ...]
        Tuple of constructed operator terms.

    Raises
    ------
    ValueError
        If ``coeffs`` is provided and its length does not match the number of
        materialized words.

    Notes
    -----
    This function intentionally does not provide a default factory to avoid
    coupling to concrete implementations and to prevent circular imports.
    Callers such as ``OpPoly.from_words`` should supply their concrete term
    class.

    """
    ws = [tuple(w) for w in words]

    cs: list[complex]

    if coeffs is None:
        cs = [1.0] * len(ws)
    else:
        cs = list(coeffs)
        if len(cs) != len(ws):
            raise ValueError(
                "coeffs must match the number of words: "
                f"len(coeffs)={len(cs)} != len(words)={len(ws)}"
            )

    return tuple(term_factory(w, c) for w, c in zip(ws, cs, strict=True))
