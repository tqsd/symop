from __future__ import annotations
from typing import Dict, Iterable, Tuple

from symop_proto.core.protocols import (
    KetTermProto,
    LadderOpProto,
    MonomialProto,
)


def ket_from_word(
    *, ops: Iterable[LadderOpProto], eps: float = 1e-12
) -> Tuple[KetTermProto, ...]:
    """Construct ket terms from a sequence ("word") of ladder operators.

    This function expands a product of ladder operators (a "word") into
    a linear combination of normally ordered :class:`KetTerm` instances.
    Commutation relations between creation and annihilation operators are
    applied recursively at each step to generate all valid monomials and
    their coefficients.

    A *word* is an ordered iterable of ladder operators, e.g.
    ``[adag[1], a[2], adag[3]]``, representing an operators product
    :math:`\\hat a_1^\\dagger \\hat a_2 \\hat a_3^\\dagger`.
    The resulting ket terms correspond to the symbolic expansion of this
    operator word acting on the vacuum.

    Args:
        ops: Operator objects representing the operator word to expand.
            Operators are processed from left to right in the order they
            appear.

    Returns:
        Tuple of resulting ket terms, each containing the combined complex
        coefficent and the corresponding monomial in normal order. Terms
        with negligible coefficients (|c| < 1e-12) are discarded and the
        output is sorted by monomial signature.

    Notes:
        - Each time an operator pair is encountered, their commutator
          contributes an additional "contracted" term.
        - The algorithm performs a symbolic normal ordering without explicit
          matrix representations.
        - Coefficeints are accumulated numerically, symbolic coefficients can
          also be supported if provided by the commutator definitions

    """

    from symop_proto.core.monomial import Monomial
    from symop_proto.core.terms import KetTerm

    coeffs: Dict[tuple, complex] = {}
    reps: Dict[tuple, Monomial] = {}
    k0 = Monomial((), ()).signature
    coeffs[k0] = 1.0 + 0.0j
    reps[k0] = Monomial((), ())
    for op in ops:
        new_coeffs: Dict[tuple, complex] = {}
        new_reps: Dict[tuple, Monomial] = {}
        for k, c in coeffs.items():
            m = reps[k]
            if op.is_annihilation:
                m_pass = Monomial(m.creators, m.annihilators + (op,))
                kp = m_pass.signature
                new_coeffs[kp] = new_coeffs.get(kp, 0j) + c
                new_reps.setdefault(kp, m_pass)
                for idx, Ck in enumerate(m.creators):
                    w = op.commutator(Ck)
                    if w != 0.0:
                        m_contract = Monomial(
                            m.creators[:idx] + m.creators[idx + 1:],
                            m.annihilators,
                        )
                        kc = m_contract.signature
                        new_coeffs[kc] = new_coeffs.get(kc, 0j) + c * w
                        new_reps.setdefault(kc, m_contract)
            else:
                m_pass = Monomial(m.creators + (op,), m.annihilators)
                kp = m_pass.signature
                new_coeffs[kp] = new_coeffs.get(kp, 0j) + c
                new_reps.setdefault(kp, m_pass)
                for idx, Ak in enumerate(m.annihilators):
                    w = Ak.commutator(op)
                    if w != 0.0:
                        m_contract = Monomial(
                            m.creators,
                            m.annihilators[:idx] + m.annihilators[idx + 1:],
                        )
                        kc = m_contract.signature
                        new_coeffs[kc] = new_coeffs.get(kc, 0j) + c * w
                        new_reps.setdefault(kc, m_contract)
        coeffs, reps = new_coeffs, new_reps
    terms = [KetTerm(c, reps[k]) for k, c in coeffs.items() if abs(c) > eps]
    terms.sort(key=lambda t: t.monomial.signature)
    return tuple(terms)
