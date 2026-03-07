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
from itertools import count
from typing import Self

from symop.core.protocols.modes.labels import (
    Envelope as EnvelopeProtocol,
)
from symop.core.protocols.modes.labels import (
    ModeLabel as ModeLabelProtocol,
)
from symop.core.protocols.modes.labels import (
    Path as PathProtocol,
)
from symop.core.protocols.modes.labels import (
    Polarization as PolarizationProtocol,
)
from symop.core.protocols.ops import (
    LadderOp as LadderOpProtocol,
)
from symop.core.types import OperatorKind
from symop.core.types.signature import Signature

_mode_display_counter = count(1)


@dataclass(frozen=True)
class ModeOp:
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

    This implements
    :class:`~symop.core.protocols.ops.operators.ModeOp` protocol.

    """

    label: ModeLabelProtocol

    user_label: str | None = None
    display_index: int | None = field(
        default_factory=lambda: next(_mode_display_counter)
    )

    _ann: LadderOp = field(init=False, repr=False, compare=False)
    _cre: LadderOp = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Create cached ladder-operator views (ann/create) bound to this mode."""
        object.__setattr__(self, "_ann", LadderOp(kind=OperatorKind.ANN, mode=self))
        object.__setattr__(self, "_cre", LadderOp(kind=OperatorKind.CRE, mode=self))

    @property
    def ann(self) -> LadderOp:
        """Public reference to an annihilation `LadderOp`."""
        return self._ann

    @property
    def cre(self) -> LadderOp:
        """Public reference to an creation `LadderOp`."""
        return self._cre

    @property
    def annihilate(self) -> LadderOp:
        """Alias for `self.ann`."""
        return self.ann

    @property
    def create(self) -> LadderOp:
        """Alias for `self.cre`."""
        return self.cre

    def with_user_label(self, tag: str) -> Self:
        """Return an updated ``ModeOp`` with new ``user_label``."""
        return replace(self, user_label=tag)

    def with_index(self, idx: int) -> Self:
        """Return an updated ``ModeOp`` with new ``index``."""
        return replace(self, display_index=idx)

    def with_envelope(self, envelope: EnvelopeProtocol) -> Self:
        """Return an updated ``ModeOp`` with new ``Envelope``."""
        return replace(self, label=self.label.with_envelope(envelope))

    def with_label(self, label: ModeLabelProtocol) -> Self:
        """Return an updated ``ModeOp`` with new ``label``."""
        return replace(self, label=label)

    def with_polarization(self, polarization: PolarizationProtocol) -> Self:
        """Return an updated ``ModeOp`` with new ``Polarization``."""
        return replace(self, label=self.label.with_polarization(polarization))

    def with_path(self, path: PathProtocol) -> Self:
        """Return an updated ``ModeOp`` with new ``Path``."""
        return replace(self, label=self.label.with_path(path))

    @property
    def signature(self) -> Signature:
        """Return a signature."""
        return ("mode", self.label.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Signature:
        """Return an approximate signature."""
        return (
            "mode_approx",
            self.label.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
        )


@dataclass(frozen=True)
class LadderOp:
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

    Notes
    -----
    ``LadderOp`` is immutable (``frozen=True``). The ``with_*`` helpers return
    updated copies via :func:`dataclasses.replace`.

    This implements
    :class:`~symop.core.protocols.ops.operators.LadderOp` protocol.

    """

    kind: OperatorKind
    mode: ModeOp

    @property
    def is_annihilation(self) -> bool:
        r"""Return ``True`` if this operator is an annihilation operator :math:`a`."""
        return self.kind == OperatorKind.ANN

    @property
    def is_creation(self) -> bool:
        r"""Return ``True`` if this operator is a creation operator :math:`a^\dagger`."""
        return self.kind == OperatorKind.CRE

    def dagger(self) -> LadderOp:
        r"""Return the Hermitian adjoint of this ladder operator.

        .. math::

            (a_i)^\dagger = a_i^\dagger, \qquad
            (a_i^\dagger)^\dagger = a_i.
        """
        return self.mode.ann if self.is_creation else self.mode.cre

    def commutator(self, other: LadderOpProtocol) -> complex:
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
    def signature(self) -> Signature:
        """Return a signature."""
        return ("lop", self.kind.value, self.mode.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Signature:
        """Return an approximate signature."""
        return (
            "lop",
            self.kind.value,
            self.mode.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
        )
