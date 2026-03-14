r"""Multiplication of operator-term tuples.

This module implements the Cartesian product multiplication of two
collections of operator terms.

Given two finite collections

.. math::

    A = \{ T_i \}, \quad
    B = \{ S_j \},

their product is defined term-wise as

.. math::

    T_i S_j
    =
    (c_i d_j) \, (W_i || V_j),

where ``W_i || V_j`` denotes concatenation of operator words.
"""

from __future__ import annotations

from symop.ccr._typing import OpTermFactory, OpTermT
from symop.core.protocols.terms.op_term import OpTerm


def multiply(
    a: tuple[OpTerm, ...],
    b: tuple[OpTerm, ...],
    *,
    term_factory: OpTermFactory[OpTermT],
) -> tuple[OpTermT, ...]:
    """Cartesian product multiplication of operator terms.

    For each pair ``(t_i, t_j)`` in ``a x b``, a new term is formed
    whose operator word is the concatenation

    .. math::

        W_{ij} = W_i || V_j,

    and whose coefficient is

    .. math::

        c_{ij} = c_i d_j.

    Parameters
    ----------
    a:
        Left operator terms.
    b:
        Right operator terms.
    term_factory:
        Callable constructing a term from ``(ops, coeff)``.

    Returns
    -------
    tuple[OpTermT, ...]
        Tuple containing ``len(a) * len(b)`` terms in row-major order.

    Notes
    -----
    This function performs purely algebraic concatenation. No
    simplification or combination of like terms is performed.
    Callers should apply :func:`combine_like_terms` if normalization
    is required.

    """
    return tuple(
        term_factory(ti.ops + tj.ops, ti.coeff * tj.coeff) for ti in a for tj in b
    )
