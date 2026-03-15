"""Symbolic rewrite utilities for CCR polynomial representations.

This module provides generic substitution and expansion routines for
polynomial objects used in the canonical commutation relation (CCR)
algebra implementation. The functions allow ladder operators appearing
inside polynomial expressions to be replaced by linear combinations of
other operators.

The rewrite process follows three conceptual steps:

1. **Local substitution**
   Each ladder operator is replaced using a user-provided substitution
   function mapping

   ``a -> sum_i c_i * b_i``.

2. **Word expansion**
   Substitutions are applied to every operator in a word, producing a
   cartesian product of all substitution choices.

3. **Normal ordering**
   The resulting operator words are normal ordered using the CCR
   normal-ordering routine, producing monomials compatible with the
   symbolic polynomial representation.

The utilities support rewriting of the following CCR polynomial types:

- ``KetPoly``      : symbolic ket polynomials
- ``DensityPoly``  : symbolic density polynomials
- ``OpPoly``       : operator polynomials

The substitution logic is shared across these representations, ensuring
consistent symbolic manipulation across states, operators, and channels.

Notes
-----
These routines operate purely at the symbolic algebra level. They do not
perform numerical evaluation or truncation of Hilbert spaces.

Small coefficient terms can be pruned using the ``eps`` parameter during
intermediate expansion.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

from symop.ccr.algebra.ket import KetPoly
from symop.ccr.protocols import (
    DensityPoly as DensityPolyProtocol,
)
from symop.ccr.protocols import (
    KetPoly as KetPolyProtocol,
)
from symop.ccr.protocols import (
    OpPoly as OpPolyProtocol,
)
from symop.core.protocols.ops import (
    LadderOp as LadderOpProtocol,
)
from symop.core.protocols.ops import (
    Monomial as MonomialProtocol,
)

SubstFn = Callable[[LadderOpProtocol], Sequence[tuple[complex, LadderOpProtocol]]]


def expand_word_substitution(
    word: Sequence[LadderOpProtocol],
    subst_fn: SubstFn,
    *,
    eps: float = 0.0,
) -> list[tuple[complex, tuple[LadderOpProtocol, ...]]]:
    """Expand a word of ladder operators using a substitution rule.

    Each operator in the input word is replaced using ``subst_fn``.
    The substitutions are combined via a cartesian product, generating
    all possible operator words resulting from the replacements.

    Parameters
    ----------
    word :
        Sequence of ladder operators forming the operator word.
    subst_fn :
        Substitution function mapping a ladder operator to a sequence of
        ``(coefficient, operator)`` pairs representing a linear
        combination replacement.
    eps :
        Optional pruning threshold. Intermediate products with absolute
        coefficient smaller than ``eps`` are discarded.

    Returns
    -------
    list of tuple
        List of ``(coefficient, new_word)`` pairs, where ``new_word`` is a
        tuple of substituted ladder operators.

    Notes
    -----
    This routine does **not** perform normal ordering. It only performs
    substitution and expansion of operator words.

    """
    lists: list[Sequence[tuple[complex, LadderOpProtocol]]] = [
        subst_fn(op) for op in word
    ]
    out: list[tuple[complex, tuple[LadderOpProtocol, ...]]] = []
    for choices in product(*lists):
        c = 1.0 + 0.0j
        ops: list[LadderOpProtocol] = []
        for coeff, new_op in choices:
            c *= complex(coeff)
            if eps and abs(c) <= eps:
                # mild pruning during buildup
                ops = []
                break
            ops.append(new_op)
        if not ops and word:
            continue
        if eps and abs(c) <= eps:
            continue
        out.append((c, tuple(ops)))
    return out


def _normalize_word_to_monomials(
    word: Sequence[LadderOpProtocol],
    *,
    eps: float,
) -> list[tuple[complex, MonomialProtocol]]:
    """Convert an operator word into normal-ordered monomials.

    Parameters
    ----------
    word :
        Sequence of ladder operators representing an operator word.
    eps :
        Numerical pruning threshold passed to the normal-ordering routine.

    Returns
    -------
    list of tuple
        List of ``(coefficient, monomial)`` pairs obtained after
        normal ordering the operator word.

    Notes
    -----
    Normal ordering is performed using the CCR ordering routine
    ``ket_from_word``.

    """
    from symop.ccr.algebra.ket.from_word import ket_from_word

    terms = ket_from_word(
        ops=tuple(word),
        eps=eps,
    )
    out: list[tuple[complex, MonomialProtocol]] = []
    for kt in terms:
        out.append((complex(kt.coeff), kt.monomial))
    return out


def rewrite_ketpoly(
    poly: KetPolyProtocol,
    subst_fn: SubstFn,
    *,
    eps: float = 1e-12,
) -> KetPoly:
    r"""Apply a ladder-operator substitution to a ``KetPoly``.

    Each ladder operator appearing in the polynomial's monomials is
    replaced using ``subst_fn``. The resulting expressions are expanded
    and normal ordered.

    Parameters
    ----------
    poly :
        Input ket polynomial.
    subst_fn :
        Substitution function mapping ladder operators to linear
        combinations of operators.
    eps :
        Numerical pruning threshold used during expansion and
        normal ordering.

    Returns
    -------
    KetPoly
        New polynomial with substitutions applied and terms combined.

    Notes
    -----
    The output is always returned as the concrete ``KetPoly`` algebra
    implementation, even if the input object satisfies only the
    ``KetPolyProtocol``.

    """
    from symop.ccr.algebra.ket.from_word import ket_from_word
    from symop.core.terms import KetTerm

    out_terms: list[KetTerm] = []
    for t in poly.terms:
        word = (*t.monomial.creators, *t.monomial.annihilators)
        for c2, w2 in expand_word_substitution(word, subst_fn, eps=eps):
            expanded = ket_from_word(
                ops=w2,
                eps=eps,
            )
            for tt in expanded:
                out_terms.append(tt.scaled(t.coeff * c2))

    return KetPoly(tuple(out_terms)).combine_like_terms()


def rewrite_densitypoly(
    rho: DensityPolyProtocol,
    left_subst_fn: SubstFn,
    right_subst_fn: SubstFn | None = None,
    *,
    eps: float = 1e-12,
) -> DensityPolyProtocol:
    r"""Rewrite a density polynomial using ladder-operator substitutions.

    Substitutions are applied independently to the left and right
    monomials of each density term.

    Parameters
    ----------
    rho :
        Input density polynomial.
    left_subst_fn :
        Substitution function applied to ladder operators appearing in
        left monomials.
    right_subst_fn :
        Optional substitution function for right monomials. If ``None``,
        ``left_subst_fn`` is used for both sides.
    eps :
        Numerical pruning threshold used during intermediate expansions.

    Returns
    -------
    DensityPolyProtocol
        Rewritten density polynomial with substitutions applied and
        like terms combined.

    Notes
    -----
    The rewrite process proceeds as:

    1. Expand substitutions for left and right words.
    2. Normal order each resulting word.
    3. Form the cartesian product of left/right monomial combinations.
    4. Construct new density terms from the resulting monomials.

    """
    from symop.ccr.algebra.density import DensityPoly
    from symop.core.terms import DensityTerm

    if right_subst_fn is None:
        right_subst_fn = left_subst_fn

    out_terms: list[DensityTerm] = []

    for t in rho.terms:
        left_word = (*t.left.creators, *t.left.annihilators)
        right_word = (*t.right.creators, *t.right.annihilators)

        left_expanded = expand_word_substitution(left_word, left_subst_fn, eps=eps)
        right_expanded = expand_word_substitution(right_word, right_subst_fn, eps=eps)

        left_monos: list[tuple[complex, MonomialProtocol]] = []
        right_monos: list[tuple[complex, MonomialProtocol]] = []

        for cL, wL in left_expanded:
            for kcoeff, mono in _normalize_word_to_monomials(wL, eps=eps):
                left_monos.append((cL * kcoeff, mono))

        for cR, wR in right_expanded:
            for kcoeff, mono in _normalize_word_to_monomials(wR, eps=eps):
                right_monos.append((cR * kcoeff, mono))

        for (cL, mL), (cR, mR) in product(left_monos, right_monos):
            out_terms.append(DensityTerm(t.coeff * cL * cR, mL, mR))

    return DensityPoly(tuple(out_terms)).combine_like_terms()


def rewrite_oppoly(
    op: OpPolyProtocol,
    subst_fn: SubstFn,
    *,
    eps: float = 1e-12,
) -> OpPolyProtocol:
    """Rewrite an operator polynomial using ladder-operator substitution.

    Parameters
    ----------
    op :
        Operator polynomial whose terms contain words of ladder operators.
    subst_fn :
        Substitution function mapping ladder operators to linear
        combinations of operators.
    eps :
        Numerical pruning threshold used during expansion.

    Returns
    -------
    OpPolyProtocol
        Operator polynomial with substitutions applied and like terms
        combined.

    Notes
    -----
    Unlike state polynomials, operator polynomials do not require
    normal ordering during rewriting because the operator word
    structure is preserved.

    """
    from symop.ccr.algebra.op import OpPoly
    from symop.core.terms import OpTerm

    out_terms: list[OpTerm] = []
    for t in op.terms:
        if abs(t.coeff) <= eps:
            continue
        for c2, new_word in expand_word_substitution(t.ops, subst_fn, eps=eps):
            coeff = t.coeff * c2
            if abs(coeff) <= eps:
                continue
            out_terms.append(OpTerm(coeff=coeff, ops=new_word))
    return OpPoly(tuple(out_terms)).combine_like_terms()
