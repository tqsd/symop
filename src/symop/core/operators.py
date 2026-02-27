"""Operator primitives for mode and ladder-operator bookkeeping.

This module defines logical modes and their associated ladder operators,
including signatures and commutation behavior based on overlaps.

A logical mode is represented by :class:`ModeOp`, which is defined by a
composite mode label (typically including path, polarization, and envelope).
The generalized canonical commutation relations are implemented by
:class:`LadderOp` using overlaps induced by the mode label.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from itertools import count

from symop.core.protocols import (
    EnvelopeLike,
    LabelProto,
    LadderOpProto,
    ModeLabelLike,
    ModeOpProto,
    OperatorKindProto,
    SignatureProto,
)

_mode_display_counter = count(1)


class OperatorKind(StrEnum):
    """Kinds of ladder operators.

    The string values are used in signatures and for compact display:

    - ``"a"`` for annihilation operators
    - ``"adag"`` for creation operators
    """

    ANN = "a"
    CREATE = "adag"


@dataclass(frozen=True)
class ModeOp(ModeOpProto):
    r"""A logical mode, defined by a composite mode label.

    Conceptually, a ``ModeOp`` represents a (possibly composite) bosonic mode
    characterized by a label ``label`` that fully specifies how the mode overlaps
    with other modes (e.g., via path, polarization, and envelope components).

    This object is also the factory for its ladder operators ``ann`` and
    ``create``:

    .. math::

        a_{\mathrm{mode}}, \quad a_{\mathrm{mode}}^\dagger.

    The commutation relations are implemented on :class:`LadderOp` using the
    overlap induced by ``label``.

    Notes
    -----
    ``ModeOp`` is immutable (``frozen=True``). The ``with_*`` helpers return
    updated copies via :func:`dataclasses.replace`.

    The ``display_index`` is intended purely for UI/debugging and does not
    contribute to signatures.

    """

    label: ModeLabelLike

    user_label: str | None = None
    display_index: int | None = field(
        default_factory=lambda: next(_mode_display_counter)
    )

    ann: LadderOpProto = field(init=False, repr=False, compare=False)
    create: LadderOpProto = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Create cached ladder-operator views (ann/create) bound to this mode."""
        object.__setattr__(self, "ann", LadderOp(kind=OperatorKind.ANN, mode=self))
        object.__setattr__(
            self, "create", LadderOp(kind=OperatorKind.CREATE, mode=self)
        )

    def with_user_label(self, tag: str) -> ModeOp:
        """Return an updated ``ModeOp`` with new ``user_label``."""
        return replace(self, user_label=tag)

    def with_index(self, idx: int) -> ModeOp:
        """Return an updated ``ModeOp`` with new ``index``."""
        return replace(self, display_index=idx)

    def with_envelope(self, envelope: EnvelopeLike) -> ModeOp:
        """Return an updated ``ModeOp`` with new ``Envelope``."""
        return replace(self, label=self.label.with_envelope(envelope))

    def with_label(self, label: ModeLabelLike) -> ModeOp:
        """Return an updated ``ModeOp`` with new ``label``."""
        return replace(self, label=label)

    def with_pol(self, pol: LabelProto) -> ModeOp:
        """Return an updated ``ModeOp`` with new ``Polarization``."""
        return replace(self, label=self.label.with_pol(pol))

    def with_path(self, path: LabelProto) -> ModeOp:
        """Return an updated ``ModeOp`` with new ``Path``."""
        return replace(self, label=self.label.with_path(path))

    @property
    def signature(self) -> SignatureProto:
        """Return a signature."""
        return ("mode", self.label.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> SignatureProto:
        """Return an approximate signature."""
        return (
            "mode_approx",
            self.label.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
        )


@dataclass(frozen=True)
class LadderOp(LadderOpProto):
    r"""A bosonic ladder operator bound to a specific logical mode.

    A ``LadderOp`` is either an annihilation operator :math:`a` or a creation
    operator :math:`a^\dagger` acting on the mode specified by ``mode``.

    The (generalized) canonical commutation relations used here are determined
    by the overlap of the logical modes. For two logical modes :math:`i` and
    :math:`j`, define:

    .. math::

        G_{ij} = \langle \mathrm{mode}_i, \mathrm{mode}_j \rangle
            = \langle \mathrm{label}_i, \mathrm{label}_j \rangle.

    Then the commutators are:

    .. math::

        [a_i, a_j^\dagger] = G_{ij}, \qquad
        [a_i^\dagger, a_j] = -G_{ij}, \qquad
        [a_i, a_j] = [a_i^\dagger, a_j^\dagger] = 0.

    If the mode overlap is (numerically) zero, the operators commute.
    """

    kind: OperatorKindProto
    mode: ModeOpProto

    @property
    def is_annihilation(self) -> bool:
        r"""Return ``True`` if this operator is an annihilation operator :math:`a`."""
        return self.kind.value == OperatorKind.ANN.value

    @property
    def is_creation(self) -> bool:
        r"""Return ``True`` if this operator is a creation operator :math:`a^\dagger`."""
        return self.kind.value == OperatorKind.CREATE.value

    def dagger(self) -> LadderOpProto:
        r"""Return the Hermitian adjoint of this ladder operator.

        .. math::

            (a_i)^\dagger = a_i^\dagger, \qquad
            (a_i^\dagger)^\dagger = a_i.
        """
        return self.mode.ann if self.is_creation else self.mode.create

    def commutator(self, other: LadderOpProto) -> complex:
        r"""Compute the commutator with another ladder operator.

        The result implements the generalized CCR based on logical-mode overlap:

        .. math::

            [a_i, a_j^\dagger] = \langle \mathrm{mode}_i, \mathrm{mode}_j \rangle
                            = \langle \mathrm{label}_i, \mathrm{label}_j \rangle.

        The other cases follow by antisymmetry and vanishing same-kind commutators:

        .. math::

            [a_i^\dagger, a_j] = -[a_j, a_i^\dagger], \qquad
            [a_i, a_j] = [a_i^\dagger, a_j^\dagger] = 0.

        Numerical note
        --------------
        If the overlap magnitude is below ``1e-15``, this returns ``0j`` to avoid
        noise from nearly-orthogonal modes.
        """
        G = self.mode.label.overlap(other.mode.label)
        if abs(G) < 1e-15:
            return 0.0 + 0.0j
        if self.is_annihilation and other.is_creation:
            return G
        if self.is_creation and other.is_annihilation:
            return -G
        return 0.0 + 0.0j

    @property
    def signature(self) -> SignatureProto:
        """Return a signature."""
        return ("lop", self.kind.value, self.mode.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> SignatureProto:
        """Return an approximate signature."""
        return (
            "lop",
            self.kind.value,
            self.mode.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
        )
