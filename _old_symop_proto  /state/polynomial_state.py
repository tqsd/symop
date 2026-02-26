from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass, field, replace
from itertools import count

from symop_proto.algebra.operator_polynomial import OpPoly
from symop_proto.algebra.polynomial import DensityPoly, KetPoly
from symop_proto.algebra.pretty.density import density_repr
from symop_proto.algebra.pretty.ket import ket_latex, ket_repr
from symop_proto.algebra.protocols import (
    DensityPolyProto,
    KetPolyProto,
    OpPolyProto,
)
from symop_proto.core.operators import LadderOp, ModeOp, OperatorKind
from symop_proto.core.protocols import LadderOpProto, ModeOpProto
from symop_proto.core.terms import KetTerm
from symop_proto.state.pretty.polynomial_state import (
    density_state_latex_from_terms,
)
from symop_proto.state.protocols import (
    DensityPolyStateProto,
    KetPolyStateProto,
)

_state_counter = count(1)


@dataclass(frozen=True)
class KetPolyState(KetPolyStateProto):
    r"""Physical ket state consisting of a **creators-only** polynomial
    acting on vacuum :math:`\lvert 0\rangle`.

    This wrapper guarantees that the underlying ket polynomial contains only
    the creation operators (and optiona identity terms), making it a valid
    physical state vector up to normalization.

    Notes
    -----
    - Use :meth:`normalized` or :py:attr:`norm2` to manage the normalization
    - Convert to a density operator via :meth:`to_density`.

    Parameters
    ----------
    - ket: A creators-only :class:`~symop_proto.algebra.polynomial.KetPoly`
    - label: Optional label used in pretty/LaTeX rendering
    - index: Optional (automatically assigned) index used in rendering

    Examples
    --------

    .. jupyter-execute::

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.algebra.polynomial import KetPoly
        from symop_proto.state.polynomial_state import KetPolyState
        from IPython.display import Math, display

        env = GaussianEnvelope(omega0=1.0, sigma=0.3, tau=0.0, phi0=0.0)
        label = ModeLabel(PathLabel("A"), PolarizationLabel.H())
        m = ModeOp(env=env, label=label)

        psi = KetPolyState.from_creators([m.create, m.create])
        display(psi)

    """

    ket: KetPolyProto
    label: str | None = None
    index: int | None = field(default_factory=lambda: next(_state_counter))

    def __post_init__(self):
        """Validate the creators-only invariant.

        Raises
        ------
        ValueError: If ``ket`` contains annihilators

        """
        if not self.ket.is_creator_only:
            raise ValueError("State must be creators-only (plus identity terms).")

    @staticmethod
    def vacuum() -> KetPolyState:
        r"""Construct the vacuum state :math:`\lvert 0\rangle`.

        Returns
        -------
        ``KetPolyState``: A state whose underlying polynomial is the identity term only.

        Examples
        --------

        .. jupyter-execute::

            from symop_proto.state.polynomial_state import KetPolyState
            from IPython.display import Math, display
            psi0 = KetPolyState.vacuum()
            psi0.norm2
            psi0.is_normalized()
            display(psi0)

        """
        return KetPolyState(KetPoly((KetTerm.identity(),)))

    @staticmethod
    def from_creators(
        creators: Iterable[LadderOpProto], coeff: complex = 1.0
    ) -> KetPolyState:
        r"""Build a state from a word of **creation** operators.

        Parameters
        ----------
        - creators: Iterable of creators
        - coeff: Global complex cofficient

        Returns
        -------
        ``KetPolyState``: The resulting physical state.


        .. jupyter-execute::

            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.labels.path_label import PathLabel
            from symop_proto.labels.polarization_label import PolarizationLabel
            from symop_proto.labels.mode_label import ModeLabel
            from symop_proto.core.operators import ModeOp
            from symop_proto.algebra.polynomial import KetPoly
            from symop_proto.state.polynomial_state import KetPolyState
            from IPython.display import Math, display

            env1 = GaussianEnvelope(omega0=1.0, sigma=0.3, tau=0.0, phi0=0.0)
            label1 = ModeLabel(PathLabel("A"), PolarizationLabel.H())
            m1 = ModeOp(env=env1, label=label1)
            env2 = GaussianEnvelope(omega0=1.0, sigma=0.5, tau=0.0, phi0=0.0)
            label2 = ModeLabel(PathLabel("B"), PolarizationLabel.H())
            m2 = ModeOp(env=env2, label=label2)

            psi = KetPolyState.from_creators([m2.create, m1.create])
            display(psi)

        """
        creators = tuple(creators)
        if any(getattr(op, "kind", None) != OperatorKind.CREATE for op in creators):
            raise ValueError("from_creators expects only creation operators.")
        return KetPolyState(
            KetPoly.from_ops(
                creators=creators, annihilators=(), coeff=coeff
            ).combine_like_terms()
        )

    def with_label(self, label: str | None) -> KetPolyState:
        return replace(self, label=label)

    def with_index(self, index: int | None) -> KetPolyState:
        return replace(self, index=index)

    def _with_ket(self, new_ket: KetPolyProto) -> KetPolyState:
        return replace(self, ket=new_ket)

    @staticmethod
    def from_ketpoly(ket: KetPolyProto) -> KetPolyState:
        return KetPolyState(ket.combine_like_terms())

    def is_normalized(self, *, eps: float = 1e-14) -> bool:
        return self.ket.is_normalized(eps=eps)

    def normalized(self, *, eps: float = 1e-14) -> KetPolyState:
        return self._with_ket(self.ket.normalize(eps=eps))

    def to_density(self) -> DensityPolyState:
        return DensityPolyState.pure(self.ket)

    def expect(
        self, op: OpPolyProto, *, normalize: bool = True, eps: float = 1e-14
    ) -> complex:
        return self.to_density().expect(op, normalize=normalize, eps=eps)

    @property
    def norm2(self) -> float:
        return self.ket.norm2()

    @property
    def unique_modes(self) -> tuple[ModeOpProto, ...]:
        return self.ket.unique_modes

    def __repr__(self) -> str:
        return ket_repr(self.ket.terms, is_state=True)

    def _repr_latex_(self) -> str:
        lbl = self.index
        return rf"$\lvert \psi_{lbl}\rangle = {ket_latex(self.ket.terms, is_state=True, show_identity=False)}\lvert0\rangle$"

    def __matmul__(self, other):
        # state @ something - not a well-defined right action for kets
        return NotImplemented

    def __rmatmul__(self, other) -> KetPolyState:
        """Supports:
        OpPoly   @ KetPolyState  -> KetPolyState
        KetPoly  @ KetPolyState  -> KetPolyState
        LadderOp @ KetPolyState  -> KetPolyState
        Iterable[LadderOp] @ KetPolyState -> KetPolyState
        ModeOp   @ KetPolyState  -> interpreted as ModeOp.create @ state (optional)
        """
        # 1) OpPoly @ state
        if isinstance(other, OpPoly):
            pairs = ((t.coeff, t.ops) for t in other.terms)
            kp = self.ket.apply_words(pairs).combine_like_terms()
            return self._with_ket(kp)

        # 2) KetPoly @ state
        if isinstance(other, KetPolyProto):
            kp = other.multiply(self.ket).combine_like_terms()
            return self._with_ket(kp)

        # 3a) LadderOp @ state  (single op -> length-1 word)
        if isinstance(other, LadderOp):
            kp = KetPoly.from_word(ops=(other,)).multiply(self.ket).combine_like_terms()
            return self._with_ket(kp)

        # 3b) ModeOp @ state (optional sugar: treat as its creation operator)
        if isinstance(other, ModeOp):
            op = other.create  # or .annihilate depending on your intent
            kp = KetPoly.from_word(ops=(op,)).multiply(self.ket).combine_like_terms()
            return self._with_ket(kp)

        # 3c) Iterable[LadderOp] @ state (a whole word)
        if isinstance(other, IterableABC):
            ops = tuple(other)
            if all(isinstance(op, LadderOp) for op in ops):
                kp = KetPoly.from_word(ops=ops).multiply(self.ket).combine_like_terms()
                return self._with_ket(kp)

        return NotImplemented


@dataclass(frozen=True)
class DensityPolyState(DensityPolyStateProto):
    """Physical density operator rho with nice ergonomics."""

    rho: DensityPolyProto
    label: str | None = None
    index: int | None = field(default_factory=lambda: next(_state_counter))

    def with_label(self, label: str | None) -> DensityPolyState:
        return replace(self, label=label)

    def with_index(self, index: int | None) -> DensityPolyState:
        return replace(self, index=index)

    @staticmethod
    def pure(psi: KetPolyProto | KetPolyStateProto) -> DensityPolyState:
        if isinstance(psi, KetPolyStateProto):
            ket = psi.ket
        else:
            ket = psi
        return DensityPolyState(DensityPoly.pure(ket))

    @staticmethod
    def from_densitypoly(
        rho: DensityPolyProto,
        *,
        normalize_trace: bool = False,
        eps: float = 1e-14,
    ) -> DensityPolyState:
        return DensityPolyState(
            rho.normalize_trace(eps=eps) if normalize_trace else rho
        )

    @staticmethod
    def mix(
        states: Sequence[DensityPolyStateProto],
        weights: Sequence[float],
        *,
        normalize_weights: bool = True,
    ) -> DensityPolyState:
        if len(states) != len(weights):
            raise ValueError("states and weights must have same length")
        wsum = float(sum(weights))
        if wsum <= 0:
            raise ValueError("weights must sum to a positive value")
        scale = 1.0 / wsum if normalize_weights else 1.0
        combined = DensityPoly(())  # start empty
        for s, w in zip(states, weights, strict=False):
            if w == 0:
                continue
            combined = DensityPoly(
                (
                    *combined.terms,
                    *s.rho.scaled(w * scale).terms,
                )
            )
        combined = combined.combine_like_terms()
        return DensityPolyState(combined.normalize_trace())

    def expect(
        self, op: OpPolyProto, *, normalize: bool = True, eps: float = 1e-14
    ) -> complex:
        out = 0.0 + 0.0j
        for t in op.terms:
            if t.coeff == 0:
                continue
            out += t.coeff * self.rho.apply_right(t.ops).trace()

        if not normalize:
            return out

        tr = self.trace()
        if abs(tr) <= eps:
            raise ValueError
        return out / tr

    def trace(self) -> complex:
        return self.rho.trace()

    def partial_trace(self, trace_over_modes: set) -> DensityPolyState:
        reduced = self.rho.partial_trace(trace_over_modes)
        return DensityPolyState(reduced.normalize_trace())

    def purity(self) -> float:
        return self.rho.purity()

    def is_normalized(self, *, eps: float = 1e-14) -> bool:
        return abs(1 - self.rho.trace()) < 1

    def normalized(self, *, eps: float = 1e-14) -> DensityPolyState:
        return DensityPolyState(self.rho.normalize_trace(eps=eps))

    def is_trace_normalized(self, eps: float = 1e-12) -> bool:
        return abs(self.trace() - 1.0) <= eps

    def is_pure(self, eps: float = 1e-12) -> bool:
        return abs(self.purity() - 1.0) <= eps

    # --- pretty-printing ---
    def __repr__(self) -> str:
        return density_repr(self.rho.terms)

    def _repr_latex_(self) -> str:
        return density_state_latex_from_terms(
            self.rho.terms, label=self.label, index=self.index
        )
