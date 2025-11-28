from __future__ import annotations
from itertools import product
from typing import Callable, List, Sequence, Tuple, Optional

from symop_proto.algebra.protocols import (
    DensityPolyProto,
    KetPolyProto,
    OpPolyProto,
    OpTermProto,
)
from symop_proto.core.protocols import (
    DensityTermProto,
    LadderOpProto,
    KetTermProto,
    MonomialProto,
)


KetTermFactory = Callable[[complex, MonomialProto], KetTermProto]

KetPolyFactory = Callable[[Tuple[KetTermProto, ...]], KetPolyProto]
DensityTermFactory = Callable[
    [complex, MonomialProto, MonomialProto], DensityTermProto
]

DensityPolyFactory = Callable[[Tuple[DensityTermProto, ...]], DensityPolyProto]

OpTermFactory = Callable[[complex, Tuple[LadderOpProto, ...]], OpTermProto]
OpPolyFactory = Callable[[Tuple[OpTermProto, ...]], OpPolyProto]


def expand_word_substitution(
    word: Sequence[LadderOpProto],
    subst_fn: Callable[
        [LadderOpProto], Sequence[Tuple[complex, LadderOpProto]]
    ],
):
    """
    Expand a single operator word by substituting each operator via `subst_fn`
    and taking the cartesian product of all choices.

    Returns a list of (coeff, new_word).
    """
    lists: List[Sequence[Tuple[complex, LadderOpProto]]] = [
        subst_fn(op) for op in word
    ]
    out: List[Tuple[complex, Tuple[LadderOpProto, ...]]] = []
    for choices in product(*lists):
        c = 1.0 + 0.0j
        ops: List[LadderOpProto] = []
        for coeff, new_op in choices:
            c *= coeff
            ops.append(new_op)
        out.append((c, tuple(ops)))
    return out


def rewrite_ketpoly(
    poly: KetPolyProto,
    subst_fn: Callable[
        [LadderOpProto], Sequence[Tuple[complex, LadderOpProto]]
    ],
    *,
    term_factory: Optional[KetTermFactory] = None,
    poly_factory: Optional[KetPolyFactory] = None,
    apply_to_vacuum: bool = False,
    eps: float = 1e-12,
) -> KetPolyProto:
    r"""
    Rewrite every operator in a :class:`KetPoly` by substituting each ladder
    operator with a linear combination given by ``subst_fn`` and re-expanding
    symbolically (normal ordering preserved via :func:`ket_from_word`).

    Parameters
    ----------
    poly
        Input ket polynomial.
    subst_fn
        Function mapping a single :class:`LadderOp` to a sequence of
        ``(coeff, LadderOp)`` pairs that represent the substituted linear
        combination.
    term_factory
        Factory for building individual :class:`KetTerm` objects from a
        ``(coeff, monomial)`` pair. Defaults to
        ``symop_proto.core.terms.KetTerm``.
    poly_factory
        Factory that wraps a tuple of :class:`KetTerm` into a concrete
        :class:`KetPoly` implementation. Defaults to
        ``symop_proto.algebra.polynomial.KetPoly``.
    apply_to_vacuum
        If ``True``, treat each expanded word as acting on
        :math:`\lvert 0 \rangle` (i.e. drop any annihilators that survive).
        Useful for creators-only states.
    eps
        Magnitude threshold for discarding tiny coefficients when building
        terms via :func:`ket_from_word`.

    Returns
    -------
    KetPolyProto
        A new ket polynomial built with ``poly_factory`` and like terms
        combined.

    Notes
    -----
    * This routine is agnostic to the actual physics; it only substitutes
      ladder operators and re-normal-orders symbolically.
    * For passive linear optics (unitaries over modes), ``subst_fn`` will
      map creators to linear combinations of creators and annihilators to
      combinations of annihilators. For active Gaussian devices (Bogoliubov
      maps), your ``subst_fn`` may also return annihilators for a creator,
      which is still handled by the normal-ordering step.

    Examples
    --------
    50:50 beamsplitter on a single photon:

    .. jupyter-execute::

        import numpy as np
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp, OperatorKind
        from symop_proto.algebra.polynomial import KetPoly
        from symop_proto.rewrites.functions.substitution import rewrite_ketpoly

        # two H-polarized temporal modes (same envelope for simplicity)
        env = GaussianEnvelope(omega0=0.0, sigma=1.0, tau=0.0, phi0=0.0)
        A = ModeOp(env=env,
                   label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeOp(env=env,
                   label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))

        # 50:50 BS unitary (acts on column (a_A^\dagger, a_B^\dagger))
        c = s = 1/np.sqrt(2)
        U = np.array([[c, s], [-s, c]], dtype=np.complex128)

        # substitution map for ladder ops
        def subst(op):
            # find which column this operator belongs to
            modes = (A, B)
            try:
                k = modes.index(op.mode)
            except ValueError:
                return [(1.0+0j, op)]
            col = U[:, k]
            if op.kind is OperatorKind.CREATE:
                return [(complex(col[0]), A.create),
                        (complex(col[1]), B.create)]
            else:
                # annihilator uses conjugated column
                return [(complex(np.conj(col[0])), A.ann),
                        (complex(np.conj(col[1])), B.ann)]

        # input state |1_A> = a_A^\dagger |0>
        psi_in = KetPoly.from_word(ops=(A.create,))
        psi_out = rewrite_ketpoly(psi_in, subst, apply_to_vacuum=True)

        display(psi_out)
    """
    # lazy defaults
    if term_factory is None:
        from symop_proto.core.terms import KetTerm as _KetTerm

        term_factory = _KetTerm
    if poly_factory is None:
        from symop_proto.algebra.polynomial import KetPoly as _KetPoly

        def poly_factory(terms):
            return _KetPoly(terms).combine_like_terms()

    from symop_proto.algebra.ket.from_word import (
        ket_from_word as _ket_from_word,
    )

    out_terms: List[KetTermProto] = []
    for t in poly.terms:
        word = (*t.monomial.creators, *t.monomial.annihilators)
        for c2, w2 in expand_word_substitution(word, subst_fn):
            # Expand symbolically & optionally drop annihilators (|0⟩)
            expanded = _ket_from_word(
                ops=w2,
                apply_to_vacuum=apply_to_vacuum,
                eps=eps,
                term_factory=term_factory,
            )
            if not expanded:
                continue
            for tt in expanded:
                out_terms.append(tt.scaled(t.coeff * c2))

    return poly_factory(tuple(out_terms))


def _normalize_word_to_monomials(
    word: Sequence[LadderOpProto],
    *,
    ket_term_factory: KetTermFactory,
    eps: float,
) -> List[Tuple[complex, MonomialProto]]:
    """
    Use ket_from_word to normal order a raw word and extract monomials.
    Returns a list of (coeff, monomial).
    """
    from symop_proto.algebra.ket.from_word import (
        ket_from_word as _ket_from_word,
    )

    terms = _ket_from_word(
        ops=word,
        apply_to_vacuum=False,  # do NOT drop annihilators on density sides
        eps=eps,
        term_factory=ket_term_factory,
    )
    out: List[Tuple[complex, MonomialProto]] = []
    for kt in terms:
        out.append((kt.coeff, kt.monomial))
    return out


def rewrite_densitypoly(
    rho: DensityPolyProto,
    left_subst_fn: Callable[
        [LadderOpProto], Sequence[Tuple[complex, LadderOpProto]]
    ],
    right_subst_fn: Optional[
        Callable[[LadderOpProto], Sequence[Tuple[complex, LadderOpProto]]]
    ] = None,
    *,
    ket_term_factory: Optional[KetTermFactory] = None,
    density_term_factory: Optional[DensityTermFactory] = None,
    density_poly_factory: Optional[DensityPolyFactory] = None,
    eps: float = 1e-12,
) -> DensityPolyProto:
    r"""
    Rewrite a symbolic density operator :math:`\rho` by substituting every
    ladder operator on the **left** and **right** monomials and re-expanding
    symbolically (with proper normal ordering).

    Parameters
    ----------
    rho
        Input density polynomial.
    left_subst_fn
        Substitution function for operators on the **left** (ket) side.
        Maps one ladder operator to a sequence of ``(coeff, LadderOp)`` pairs.
    right_subst_fn
        Optional substitution function for the **right** (bra) side.
        If ``None``, ``left_subst_fn`` is used for both sides (typical for
        passive unitaries implementing :math:`\rho \mapsto U\rho U^\dagger`
        at the ladder level).
    ket_term_factory
        Factory for building :class:`KetTerm` during normalization of words.
        Defaults to ``symop_proto.core.terms.KetTerm``.
    density_term_factory
        Factory for building :class:`DensityTerm` from
        ``(coeff, left_monomial, right_monomial)``.
        Defaults to ``symop_proto.core.terms.DensityTerm``.
    density_poly_factory
        Factory that wraps a tuple of :class:`DensityTerm` into a concrete
        :class:`DensityPoly` implementation. Defaults to
        ``symop_proto.algebra.polynomial.DensityPoly`` with
        ``combine_like_terms()`` applied.
    eps
        Magnitude threshold for discarding tiny coefficients.

    Returns
    -------
    DensityPolyProto
        A rewritten density operator with like terms combined.

    Notes
    -----
    * This function is **representation-free**: it only substitutes ladder
      operators and uses your symbolic normal ordering to rebuild monomials.
    * For passive linear optics (beamsplitters, phase shifters, waveplates),
      set the same substitution on both sides. For non-unitary channels or
      asymmetric rewrites, pass distinct ``left_subst_fn`` and ``right_subst_fn``.

    Examples
    --------
    50:50 beamsplitter acting on the pure state :math:`|1_A\rangle\langle 1_A|`:

    .. jupyter-execute::

        import numpy as np
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.core.operators import ModeOp, OperatorKind
        from symop_proto.algebra.polynomial import KetPoly, DensityPoly
        from symop_proto.rewrites.functions.substitution import rewrite_densitypoly

        env = GaussianEnvelope(omega0=0, sigma=1.0, tau=0, phi0=0)
        A = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))

        # 50:50 BS: column convention on (A, B)
        c = s = 1/np.sqrt(2)
        U = np.array([[c, s], [-s, c]], dtype=np.complex128)
        modes = (A, B)

        def subst(op):
            # find column for this op
            try:
                k = modes.index(op.mode)
            except ValueError:
                return [(1+0j, op)]
            col = U[:, k]
            if op.kind is OperatorKind.CREATE:
                return [(complex(col[0]), A.create),
                        (complex(col[1]), B.create)]
            else:
                return [(complex(np.conj(col[0])), A.ann),
                        (complex(np.conj(col[1])), B.ann)]

        # |1_A><1_A|
        psi = KetPoly.from_word(ops=(A.create,))
        rho = DensityPoly.pure(psi)

        rho_bs = rewrite_densitypoly(rho, subst)  # same map on both sides
        display(rho_bs)
    """
    # defaults
    if ket_term_factory is None:
        from symop_proto.core.terms import KetTerm as _KetTerm

        ket_term_factory = _KetTerm
    if density_term_factory is None:
        from symop_proto.core.terms import DensityTerm as _DensityTerm

        density_term_factory = _DensityTerm
    if density_poly_factory is None:
        from symop_proto.algebra.polynomial import DensityPoly as _DensityPoly

        def density_poly_factory(terms):
            return _DensityPoly(terms).combine_like_terms()

    if right_subst_fn is None:
        right_subst_fn = left_subst_fn

    out_terms: List[DensityTermProto] = []

    for t in rho.terms:
        # Build raw words (creators followed by annihilators) for each side
        left_word = (*t.left.creators, *t.left.annihilators)
        right_word = (*t.right.creators, *t.right.annihilators)

        # Substitute each operator in the words
        left_expanded = expand_word_substitution(left_word, left_subst_fn)
        right_expanded = expand_word_substitution(right_word, right_subst_fn)

        # Normalize to monomials via ket_from_word (do NOT drop annihilators)
        left_monos: List[Tuple[complex, MonomialProto]] = []
        right_monos: List[Tuple[complex, MonomialProto]] = []

        for cL, wL in left_expanded:
            for kcoeff, mono in _normalize_word_to_monomials(
                wL, ket_term_factory=ket_term_factory, eps=eps
            ):
                left_monos.append((cL * kcoeff, mono))

        for cR, wR in right_expanded:
            for kcoeff, mono in _normalize_word_to_monomials(
                wR, ket_term_factory=ket_term_factory, eps=eps
            ):
                right_monos.append((cR * kcoeff, mono))

        # Cartesian product of left/right monomials
        for (cL, mL), (cR, mR) in product(left_monos, right_monos):
            out_terms.append(density_term_factory(t.coeff * cL * cR, mL, mR))

    return density_poly_factory(tuple(out_terms))


def rewrite_oppoly(
    op: OpPolyProto,
    subst_fn: Callable[
        [LadderOpProto], Sequence[Tuple[complex, LadderOpProto]]
    ],
    *,
    op_term_factory: Optional[OpTermFactory] = None,
    op_poly_factory: Optional[OpPolyFactory] = None,
) -> OpPolyProto:
    """
    Substitute every ladder operator in each OpTerm's word via `subst_fn`,
    expand by cartesian products, then combine like terms with your factory.
    """
    if op_term_factory is None:
        from symop_proto.algebra.operator_polynomial import OpTerm as _OpTerm

        def op_term_factory(c, w):
            return _OpTerm(ops=tuple(w), coeff=c)

    if op_poly_factory is None:
        from symop_proto.algebra.operator_polynomial import OpPoly as _OpPoly

        def op_poly_factory(terms):
            return _OpPoly(terms).combine_like_terms()

    out: List[OpTermProto] = []
    for t in op.terms:
        for c2, new_word in expand_word_substitution(t.ops, subst_fn):
            out.append(op_term_factory(t.coeff * c2, new_word))
    return op_poly_factory(tuple(out))
