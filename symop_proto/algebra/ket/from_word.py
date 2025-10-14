from __future__ import annotations
from typing import Callable, Dict, Iterable, Optional, Tuple

from symop_proto.core.protocols import (
    KetTermProto,
    LadderOpProto,
    MonomialProto,
)


TermFactory = Callable[[complex, MonomialProto], KetTermProto]


def ket_from_word(
    *,
    ops: Iterable[LadderOpProto],
    apply_to_vacuum: bool = False,
    eps: float = 1e-12,
    term_factory: Optional[TermFactory] = None,
) -> Tuple[KetTermProto, ...]:
    r"""Construct ket terms from a sequence (*word*) of ladder operators.

    Expands an operator *word* into a linear combination of normally ordered
    :class:`KetTerm` instances. The expansion is done symbolically using the
    cannonical commutation relations (including non-orthogonal modes).

    Args:
        ops: Operator objects representing the operator word to expand.
            Operators are processed from left to right in the order they
            appear.
        apply_to_vacuum: If True the annihilation ordering is skipped and
            the annihilation operators are dropped.
        eps: Trheshold for neglecting the coefficients.
        term_factory: The term factory to build the resulting :class:`KetPoly`


    Returns:
        Tuple of resulting ket terms, each containing the combined complex
        coefficent and the corresponding monomial in normal order. Terms
        with negligible coefficients (|c| < 1e-12) are discarded and the
        output is sorted by monomial signature.

    Mathematics:
    ------------
        The algorithm performs **symbolic normal ordering** by recursively
        applying the commutator:

        .. math::

            [\hat a_i, \hat a_j^\dagger] = \langle m_i | m_j \rangle,

        where :math:`m_i` refers to the mode degrees of freedom (path,
        polarization, envelope).

        For an unordere product (a *word*)

        .. math::

            W = \hat a_{i_1}^{s_1}\,\hat a_{i_2}^{s_2}\,\cdots\,
            \hat a_{i_n}^{s_n},\quad
            s_k \in \{1,\dagger\}.


        The function produces a normally ordere]expansion

        .. math::

            W = \sum_k c_k
            \hat a_{j_1(k)}^\dagger \cdots
            \hat a_{j_m(k)}^\dagger
            \hat a_{l_1(k)} \cdots \hat a_{l_p(k)},

        here each coefficient :math:`c_k` results from the commutation rules
        generated during the reordering.

        If ``apply_to_vacuum=True``, the result models
        :math:`W\lvert 0\rangle`, and all terms still containing annihilator
        vanish:

        .. math::

            W |0\rangle = \sum_k c_k
            \hat a{j_1(k)}^\dagger \cdots \hat a_{j_m(k)}^\dagger
            |0\rangle.

        This corresponds to a superposition of Fock basis vectors with
        with amblitudes :math:`c_k`.

    Notes:
        * The left-to-right scheme contracts **once**, when a creation
          operator crosses existing annihilation operator: symmetric
          contraction on both sides would double-count.
        * Non-orthogonality is handled via the inner product
          :math:`\langle m_i \mid m_j \rangle` in the commutator definitions.

    Examples:
    ---------
    **Operator normal ordering (single orthogonal mode):**
    For :math:`[\hat a, \hat a^\dagger]=1`,

    .. math::

        \hat a \, \hat a^\dagger \;=\; a^\dagger a \;+\; 1, \qquad
        \hat a \, \hat a^\dagger a^\dagger \; = \; \hat a^\dagger
        \hat a^\dagger \hat a \; +\; 2\,\hat a^\dagger.

    .. jupyter-execute::

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.algebra.ket.from_word import ket_from_word
        from symop_proto.algebra.polynomial import KetPoly
        from IPython.display import Math, display

        env = GaussianEnvelope(omega0=1.0, sigma=0.3, tau=0.0, phi0=0.0)
        label = ModeLabel(PathLabel("A"), PolarizationLabel.H())
        m = ModeOp(env=env, label=label)

        terms = ket_from_word(
            ops=[m.ann, m.create, m.create],
            apply_to_vacuum=False)

        ket = KetPoly(terms)

        display(Math(ket.latex))

        terms = ket_from_word(
            ops=[m.ann, m.create, m.create],
            apply_to_vacuum=True
            )
        ket = KetPoly(terms)

        display(Math(ket.latex))
    """

    from symop_proto.core.monomial import Monomial

    if term_factory is None:
        from symop_proto.core.terms import KetTerm as _KetTerm

        term_factory = _KetTerm

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
    terms = [
        term_factory(c, reps[k]) for k, c in coeffs.items() if abs(c) > eps
    ]
    if apply_to_vacuum:
        terms = [t for t in terms if not t.monomial.annihilators]

    terms.sort(key=lambda t: t.monomial.signature)
    return tuple(terms)
