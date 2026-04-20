"""Monomials of ladder operators.

This module defines the Monomial data structure, representing ordered products
of creation and annihilation operators together with structural queries and
signatures used for hashing and merging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from symop.core.protocols.ops import (
    LadderOp as LadderOpProtocol,
)
from symop.core.protocols.ops import (
    ModeOp as ModeOpProtocol,
)
from symop.core.protocols.ops import (
    Monomial as MonomialProtocol,
)
from symop.core.types.signature import Signature


@dataclass(frozen=True)
class Monomial:
    r"""A normally ordered product of ladder operators.

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

    Notes
    -----
    - The monomial is assumed to be in *normal order*, meaning all
      creation operators precede the annihilation operators.
    - An empty monomial (``creators=(), annihilators=()``) represents
      the identity operator :math:`\\mathbb{I}`.

    This class implements
    :class:`~symop.core.protocols.ops.monomial.Monomial` protocol.

    """

    creators: tuple[LadderOpProtocol, ...] = field(default_factory=tuple)
    annihilators: tuple[LadderOpProtocol, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize fields to tuples for immutability and predictable behavior."""
        if not isinstance(self.creators, tuple):
            object.__setattr__(self, "creators", tuple(self.creators))
        if not isinstance(self.annihilators, tuple):
            object.__setattr__(self, "annihilators", tuple(self.annihilators))

    @staticmethod
    def identity() -> Monomial:
        """Return an identity ``Monomial``."""
        return Monomial((), ())

    @property
    def mode_ops(self) -> tuple[ModeOpProtocol, ...]:
        """Return all unique ``ModeOps`` from creators and annihilators."""
        seen: set[Signature] = set()
        out: list[ModeOpProtocol] = []
        for op in (*self.creators, *self.annihilators):
            sig = op.mode.signature
            if sig not in seen:
                seen.add(sig)
                out.append(op.mode)
        return tuple(out)

    def adjoint(self) -> Monomial:
        """Return an adjoint of this ``Monomial``."""
        dag_creators = tuple(op.dagger() for op in self.annihilators)
        dag_annihilators = tuple(op.dagger() for op in self.creators)
        return Monomial(creators=dag_creators, annihilators=dag_annihilators)

    @property
    def signature(self) -> Signature:
        """Return a signature."""
        c = tuple(op.signature for op in self.creators)
        a = tuple(op.signature for op in self.annihilators)
        return ("monomial", "cre", c, "ann", a)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Signature:
        """Return an approximate signature."""
        c = tuple(
            op.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            )
            for op in self.creators
        )
        a = tuple(
            op.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            )
            for op in self.annihilators
        )
        return ("monomial_approx", "cre", c, "ann", a)

    @property
    def is_creator_only(self) -> bool:
        """Returns ``True`` if this ``Monomial`` consists of creators only."""
        return len(self.creators) > 0 and len(self.annihilators) == 0

    @property
    def is_annihilator_only(self) -> bool:
        """Return ``True`` if this ``Monomial`` consists of annihilators only."""
        return len(self.annihilators) > 0 and len(self.creators) == 0

    @property
    def is_identity(self) -> bool:
        """Return ``True`` if this ``Monomial`` represents an identity."""
        return len(self.creators) == 0 and len(self.annihilators) == 0

    @property
    def has_creators(self) -> bool:
        """Return ``True`` if this ``Monomial`` has any creators."""
        return len(self.creators) > 0

    @property
    def has_annihilators(self) -> bool:
        """Return ``True`` if this ``Monomial has any annihilators."""
        return len(self.annihilators) > 0


if TYPE_CHECKING:
    _monomial_check: MonomialProtocol = Monomial()
