r"""Symbolic normal ordering of ladder-operator words for ket expansions.

This module provides routines that expand ordered products of ladder
operators (operator *words*) into linear combinations of normally ordered
monomials acting on kets.

Given a word

.. math::

    W = \hat o_1 \hat o_2 \cdots \hat o_L,

with :math:`\hat o_k` bosonic ladder operators (creation or annihilation),
the goal is to compute a symbolic normal-ordered expansion

.. math::

    W = \sum_k c_k\, M_k,

where each monomial :math:`M_k` has all creation operators to the left of
all annihilation operators.

The rewrite uses the canonical commutation relations (CCR)

.. math::

    [\hat a_i, \hat a_j^\dagger]
    = \langle m_i \mid m_j \rangle,

allowing for non-orthogonal modes via the overlap factor
:math:`\langle m_i \mid m_j \rangle`.

Algorithmic strategy
--------------------

The expansion is performed left-to-right. When inserting a creation operator
into a partially normal-ordered monomial, the algorithm:

1. Appends the creation operator to the creator list.
2. Contracts it once with each existing annihilator, removing that
   annihilator and multiplying the coefficient by the CCR scalar.

This "single-sided contraction" convention avoids double counting and yields
the exact normal-ordered form.

Scope
-----

This module performs purely symbolic algebra. It does not:

- Evaluate states numerically.
- Apply the result to a vacuum state.
- Perform basis changes or approximations.

Higher-level operations (e.g. applying to vacuum or combining like terms
across polynomials) are handled in other modules of the CCR layer.

Design notes
------------

- Monomials preserve operator order within creators and annihilators.
- Structural identity is defined via stable, hashable signatures.
- No pretty-printing or presentation logic is included here.

This file sits near the core of the CCR ket algebra and should remain
dependency-light and algorithmically transparent.
"""

from __future__ import annotations

from collections.abc import Iterable

from symop.core.monomial import Monomial
from symop.core.protocols import LadderOpProto
from symop.core.protocols.signature import SignatureProto
from symop.core.terms import KetTerm


def ket_from_word(
    *,
    ops: Iterable[LadderOpProto],
    eps: float = 1e-12,
) -> tuple[KetTerm, ...]:
    r"""Expand a ladder-operator word into normally ordered ket terms.

    This function takes an ordered product (a *word*) of ladder operators

    .. math::

        W = \hat o_1 \hat o_2 \cdots \hat o_L,

    and rewrites it symbolically into a finite linear combination of *normally
    ordered* monomials (all creators to the left of all annihilators):

    .. math::

        W = \sum_k c_k\, M_k.

    The rewrite uses the canonical commutation relations (CCR) for bosonic
    ladder operators, allowing non-orthogonal modes via a mode overlap factor:

    .. math::

        [\hat a_i, \hat a_j^\dagger] = \langle m_i \mid m_j \rangle.

    Here the scalar commutator value is provided by the ladder-operator
    implementation (typically via ``a.commutator(adag)``).

    Algorithm
    ---------
    The implementation processes operators left-to-right, maintaining a linear
    combination of normally ordered monomials.

    - If the next operator is an annihilator, it is appended to the monomial's
      annihilator list (normal order is preserved).
    - If the next operator is a creator, it is appended to the creator list.
      Additionally, it is contracted once with each existing annihilator in the
      monomial, producing extra terms where that annihilator is removed and the
      coefficient is multiplied by the CCR scalar.

    This "single-sided contraction" convention avoids double counting and
    produces the exact normal-ordered expansion of the input word.

    Parameters
    ----------
    ops:
        Ladder operators forming the word to be expanded, processed
        left-to-right.
    eps:
        Coefficient threshold. Terms with :math:`|c| \le \texttt{eps}` are
        discarded.

    Returns
    -------
    tuple[KetTerm, ...]
        Ket terms :math:`(c_k, M_k)` representing the normal-ordered expansion,
        sorted by monomial signature.

    Notes
    -----
    This routine performs symbolic normal ordering only. It does not evaluate
    the action on a vacuum state. If you need :math:`W\lvert 0\rangle`, apply a
    separate step that discards any term containing annihilation operators.

    """
    coeffs: dict[SignatureProto, complex] = {}
    reps: dict[SignatureProto, Monomial] = {}

    m0 = Monomial.identity()
    k0 = m0.signature
    coeffs[k0] = 1.0 + 0.0j
    reps[k0] = m0

    for op in ops:
        new_coeffs: dict[SignatureProto, complex] = {}
        new_reps: dict[SignatureProto, Monomial] = {}

        for k, c in coeffs.items():
            m = reps[k]

            if op.is_annihilation:
                m_pass = Monomial(m.creators, m.annihilators + (op,))
                kp = m_pass.signature
                new_coeffs[kp] = new_coeffs.get(kp, 0.0j) + c
                new_reps.setdefault(kp, m_pass)
                continue

            # op is creation
            m_pass = Monomial(m.creators + (op,), m.annihilators)
            kp = m_pass.signature
            new_coeffs[kp] = new_coeffs.get(kp, 0.0j) + c
            new_reps.setdefault(kp, m_pass)

            for idx, a in enumerate(m.annihilators):
                w = a.commutator(op)
                if w != 0.0:
                    m_contract = Monomial(
                        m.creators,
                        m.annihilators[:idx] + m.annihilators[idx + 1 :],
                    )
                    kc = m_contract.signature
                    new_coeffs[kc] = new_coeffs.get(kc, 0.0j) + c * w
                    new_reps.setdefault(kc, m_contract)

        coeffs, reps = new_coeffs, new_reps

    terms = [KetTerm(c, reps[k]) for k, c in coeffs.items() if abs(c) > eps]
    terms.sort(key=lambda t: t.monomial.signature)
    return tuple(terms)
