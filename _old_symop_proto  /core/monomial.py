from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple

from symop_proto.core.protocols import (
    LadderOpProto,
    ModeOpProto,
    MonomialProto,
    SignatureProto,
)


@dataclass(frozen=True)
class Monomial(MonomialProto):
    """A normally ordered product of ladder operators

    A :class:`Monomial` represents a single term in the operator
    algebra, consisting of an ordered tuple of creation and
    annihilation operators.
    The default empty monomial (without operators) corresponds to
    the identity operator.

    Attributes:
        creators: Tuple of creation operators applied in order from
            left to right.
        annihilators: Tuple of annihilation operators applied in order
            from left to right.

    Properties:
        mode_ops: Unique ordered tuple of all mode operators appearing
            in the monomial.
        signature: Exact tuple identifier of the monomial, used for sorting
            and comparing.
        approx_signature: Approximate identifier for nearly equivalent
            monomials, used in numerical merging.
        is_creator_only: Returns ``True`` if the monomial contains only
            creation operators.

    Methods:
        adjoint(): Returns the Hermitian adjoint of the monomial by
            swapping creators and annihilators and taking their individual
            adjoints.

    Notes:
        - The monomial is assumed to be in *normal order*, meaning all
          creation operators precese the annihilation operators.
        - An empty monomial (``creators=(), annihilators=()``) represents
          the identity operator :math:`\\mathbb{I}`.

    """

    creators: Tuple[LadderOpProto, ...] = ()
    annihilators: Tuple[LadderOpProto, ...] = ()

    def __post_init__(self):
        if not isinstance(self.creators, tuple):
            object.__setattr__(self, "creators", tuple(self.creators))
        if not isinstance(self.annihilators, tuple):
            object.__setattr__(self, "annihilators", tuple(self.annihilators))

    @property
    def mode_ops(self) -> Tuple[ModeOpProto, ...]:
        seen: Set[Tuple] = set()
        out: List[ModeOpProto] = []
        for op in (*self.creators, *self.annihilators):
            sig = op.mode.signature
            if sig not in seen:
                seen.add(sig)
                out.append(op.mode)
        return tuple(out)

    def adjoint(self) -> Monomial:
        dag_creators = tuple(op.dagger() for op in self.annihilators)
        dag_annihilators = tuple(op.dagger() for op in self.creators)
        return Monomial(creators=dag_creators, annihilators=dag_annihilators)

    @property
    def signature(self) -> SignatureProto:
        c = tuple(sorted(op.signature for op in self.creators))
        a = tuple(sorted(op.signature for op in self.annihilators))
        return ("cre", c, "ann", a)

    def approx_signature(self, **env_kw) -> SignatureProto:
        c = tuple(sorted(op.approx_signature(**env_kw) for op in self.creators))
        a = tuple(sorted(op.approx_signature(**env_kw) for op in self.annihilators))
        return ("cre", c, "ann", a)

    @property
    def is_creator_only(self) -> bool:
        return len(self.creators) >= 0 and len(self.annihilators) == 0

    @property
    def is_annihilator_only(self) -> bool:
        return len(self.annihilators) >= 0 and len(self.creators) == 0

    @property
    def is_identity(self) -> bool:
        return len(self.creators) == 0 and len(self.annihilators) == 0

    @property
    def has_creators(self) -> bool:
        return len(self.creators) > 0

    @property
    def has_annihilators(self) -> bool:
        return len(self.annihilators) > 0
