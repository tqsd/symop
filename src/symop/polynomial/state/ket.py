"""Polynomial ket-state wrapper.

This module provides a lightweight state container around
:class:`~symop.ccr.algebra.ket.poly.KetPoly`, exposing semantic
information about modes, paths, and labels derived from the
polynomial representation.

The wrapper serves several roles:

- attaches lightweight metadata (label, index) to a ket polynomial
- exposes convenient views of the modes contained in the state
- supports relabeling operations used by device kernels
- provides normalization helpers for ket states

Important:
---------
Mode labels are not stored independently. They are derived directly
from the concrete modes present in the ket polynomial, ensuring that
semantic information always reflects the actual algebraic state.

The wrapper also enforces the ket-state invariant that the underlying
polynomial is creators only (plus the identity term).

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import count
from typing import Literal

from symop.ccr.algebra.ket.poly import KetPoly
from symop.core.protocols.devices.label_edit import (
    DeleteModeLabel,
    LabelEdit,
    SetModeLabel,
)
from symop.core.protocols.modes import (
    ModeLabel as ModeLabelProtocol,
)
from symop.core.protocols.modes import (
    Path as PathProtocol,
)
from symop.core.protocols.ops import (
    LadderOp as LadderOpProtocol,
)
from symop.core.protocols.ops import (
    ModeOp as ModeOpProtocol,
)
from symop.core.types.operator_kind import OperatorKind
from symop.core.types.signature import Signature
from symop.polynomial.protocols.density import (
    DensityPolyState as DensityPolyStateProtocol,
)
from symop.polynomial.protocols.ket import KetPolyState as KetPolyStateProtocol
from symop.polynomial.rewrites.relabel_modes import ket_relabel_modes

_state_counter = count(1)


@dataclass(frozen=True)
class KetPolyState(KetPolyStateProtocol):
    r"""Polynomial ket state wrapper.

    Invariant
    ---------
    The underlying ket polynomial must be creators only (plus identity).

    Notes
    -----
    Semantic mode information is derived from the concrete modes present in
    ``ket``. In particular, mode labels are not stored independently;
    they are derived from the unique modes appearing in the polynomial.

    """

    ket: KetPoly
    label: str | None = None
    index: int | None = field(default_factory=lambda: next(_state_counter))

    def __post_init__(self) -> None:
        r"""Validate the ket-state invariant.

        Raises
        ------
        ValueError
            If the underlying ket polynomial contains annihilation
            operators and therefore does not represent a valid ket-state
            wrapper input.

        """
        if not self.ket.is_creator_only:
            raise ValueError("State must be creators-only (plus identity terms)")

    @property
    def rep_kind(self) -> Literal["poly"]:
        r"""Return the representation kind of this state.

        Returns
        -------
        Literal["poly"]
            The polynomial representation kind.

        """
        return "poly"

    @property
    def state_kind(self) -> Literal["ket"]:
        r"""Return the state kind of this object.

        Returns
        -------
        Literal["ket"]
            The ket-state kind.

        """
        return "ket"

    @cached_property
    def _modes_cached(self) -> tuple[ModeOpProtocol, ...]:
        r"""Cache the unique modes appearing in the ket polynomial.

        Returns
        -------
        tuple[ModeOpProtocol, ...]
            Unique modes extracted from ``rho`` in the order provided by
            ``KetPoly.unique_modes``.

        """
        return self.ket.unique_modes

    @property
    def modes(self) -> tuple[ModeOpProtocol, ...]:
        r"""Return the unique modes present in the ket polynomial.

        Returns
        -------
        tuple[ModeOpProtocol, ...]
            Unique modes appearing in ``ket``.

        """
        return self._modes_cached

    @cached_property
    def _mode_by_signature_cached(self) -> dict[Signature, ModeOpProtocol]:
        r"""Cache a lookup from mode signature to mode object.

        Returns
        -------
        dict[Signature, ModeOpProtocol]
            Mapping from mode signature to the corresponding concrete mode.

        """
        return {mode.signature: mode for mode in self.modes}

    @property
    def mode_by_signature(self) -> dict[Signature, ModeOpProtocol]:
        r"""Return a lookup from mode signature to mode object.

        Returns
        -------
        dict[Signature, ModeOpProtocol]
            Mapping from mode signature to the corresponding mode.

        """
        return self._mode_by_signature_cached

    @cached_property
    def _mode_labels_cached(self) -> dict[Signature, ModeLabelProtocol]:
        r"""Cache a lookup from mode signature to semantic mode label.

        Returns
        -------
        dict[Signature, ModeLabelProtocol]
            Mapping from mode signature to the corresponding mode label.

        """
        return {mode.signature: mode.label for mode in self.modes}

    @property
    def mode_labels(self) -> dict[Signature, ModeLabelProtocol]:
        r"""Return a lookup from mode signature to semantic label.

        Returns
        -------
        dict[Signature, ModeLabelProtocol]
            Mapping from mode signature to the corresponding mode label.

        """
        return self._mode_labels_cached

    @cached_property
    def _modes_by_path_cached(
        self,
    ) -> dict[PathProtocol, tuple[ModeOpProtocol, ...]]:
        r"""Cache grouping of modes by path.

        Returns
        -------
        dict[PathProtocol, tuple[ModeOpProtocol, ...]]
            Mapping from path to the tuple of modes whose labels lie on that
            path.

        """
        buckets: dict[PathProtocol, list[ModeOpProtocol]] = {}
        for mode in self.modes:
            buckets.setdefault(mode.label.path, []).append(mode)
        return {path: tuple(modes) for path, modes in buckets.items()}

    @property
    def modes_by_path(self) -> dict[PathProtocol, tuple[ModeOpProtocol, ...]]:
        r"""Return modes grouped by path.

        Returns
        -------
        dict[PathProtocol, tuple[ModeOpProtocol, ...]]
            Mapping from path to the tuple of modes on that path.

        """
        return self._modes_by_path_cached

    def modes_on_path(self, path: PathProtocol) -> tuple[ModeOpProtocol, ...]:
        r"""Return all modes that live on a given path.

        Parameters
        ----------
        path:
            Path identifier.

        Returns
        -------
        tuple[ModeOpProtocol, ...]
            Modes whose label path matches ``path``.

        Notes
        -----
        If no modes exist on the requested path, an empty tuple is returned.

        """
        return self.modes_by_path.get(path, ())

    def label_for_mode(self, mode_sig: Signature) -> ModeLabelProtocol:
        r"""Return the semantic label for a given mode signature.

        Parameters
        ----------
        mode_sig:
            Signature of the mode whose label should be returned.

        Returns
        -------
        ModeLabelProtocol
            Semantic label associated with the mode.

        Raises
        ------
        KeyError
            If the mode signature is not present in the state.

        """
        try:
            return self.mode_labels[mode_sig]
        except KeyError as exc:
            raise KeyError(f"No mode label for mode signature {mode_sig!r}.") from exc

    def labels_on_path(self, path: PathProtocol) -> dict[Signature, ModeLabelProtocol]:
        r"""Return labels for all modes on a given path.

        Parameters
        ----------
        path:
            Path identifier.

        Returns
        -------
        dict[Signature, ModeLabelProtocol]
            Mapping from mode signature to semantic label for modes on the
            given path.

        """
        return {
            mode.signature: self.mode_labels[mode.signature]
            for mode in self.modes_on_path(path)
        }

    def relabel_modes(
        self,
        mode_map: dict[Signature, ModeOpProtocol],
    ) -> KetPolyState:
        r"""Return a new ket state with selected ``ModeOp``s relabeled.

        Parameters
        ----------
        mode_map:
            Mapping from old labels to replacement label.

        Returns
        -------
        KetPolyState
            New state with all matching mode paths updated.

        Notes
        -----
        Only modes whose current path appears in ``mode_map`` are modified.
        Other modes are left unchanged.

        """
        ket2 = KetPoly(ket_relabel_modes(self.ket.terms, mode_map=mode_map))
        return replace(self, ket=ket2)

    def relabel_paths(
        self,
        path_map: dict[PathProtocol, PathProtocol],
    ) -> KetPolyState:
        r"""Return a new ket state with selected paths relabeled.

        Parameters
        ----------
        path_map:
            Mapping from old paths to replacement paths.

        Returns
        -------
        KetPolyState
            New state with all matching mode paths updated.

        Notes
        -----
        Only modes whose current path appears in ``path_map`` are modified.
        Other modes are left unchanged.

        """
        mode_map: dict[Signature, ModeOpProtocol] = {}

        for mode in self.modes:
            old_path = mode.label.path
            new_path = path_map.get(old_path)
            if new_path is None:
                continue
            mode_map[mode.signature] = mode.with_path(new_path)

        return self if not mode_map else self.relabel_modes(mode_map)

    def relabel_labels(
        self,
        label_map: dict[Signature, ModeLabelProtocol],
    ) -> KetPolyState:
        r"""Return a new ket state with selected labels replaced.

        Parameters
        ----------
        label_map:
            Mapping from mode signature to replacement label.

        Returns
        -------
        KetPolyState
            New state with updated labels for the selected modes.

        Raises
        ------
        KeyError
            If a supplied mode signature is not present in the state.

        """
        mode_map: dict[Signature, ModeOpProtocol] = {}

        for mode_sig, new_label in label_map.items():
            old_mode = self.mode_by_signature.get(mode_sig)
            if old_mode is None:
                raise KeyError(f"Cannot relabel missing mode signature {mode_sig!r}.")
            mode_map[mode_sig] = old_mode.with_label(new_label)

        return self if not mode_map else self.relabel_modes(mode_map)

    def apply_label_edits(
        self,
        edits: tuple[LabelEdit, ...],
    ) -> KetPolyState:
        r"""Apply semantic label edits to the state.

        Parameters
        ----------
        edits:
            Sequence of label edits to apply.

        Returns
        -------
        KetPolyState
            New state with the requested label changes applied.

        Raises
        ------
        KeyError
            If an edit refers to a missing mode signature.
        NotImplementedError
            If a delete-label edit is requested.
        TypeError
            If an unsupported edit type is encountered.

        Notes
        -----
        Since labels are derived from concrete modes in the ket
        polynomial, edits are implemented by rebuilding the affected
        modes and relabeling them algebraically.

        """
        label_map: dict[Signature, ModeLabelProtocol] = {}

        for edit in edits:
            if isinstance(edit, SetModeLabel):
                old_mode = self.mode_by_signature.get(edit.mode_sig)
                if old_mode is None:
                    raise KeyError(
                        f"Cannot update label for missing mode {edit.mode_sig!r}."
                    )
                label_map[edit.mode_sig] = edit.label

            elif isinstance(edit, DeleteModeLabel):
                raise NotImplementedError(
                    "DeleteModeLabel is not supported when labels are derived "
                    "from the ket. Remove or replace the mode algebraically instead."
                )

            else:
                raise TypeError(f"Unsupported label edit type: {type(edit).__name__}")

        return self if not label_map else self.relabel_labels(label_map)

    def filtered_on_path(
        self,
        *,
        in_path: PathProtocol,
        out_path: PathProtocol | None,
        update_label: Callable[[ModeLabelProtocol], ModeLabelProtocol],
    ) -> KetPolyState:
        r"""Apply a label update to all modes on an input path.

        Parameters
        ----------
        in_path:
            Path whose modes should be updated.
        out_path:
            Optional replacement path for the affected modes. If None,
            the original paths are retained.
        update_label:
            Callable that transforms each mode label before optional
            path replacement.

        Returns
        -------
        KetPolyState
            New state with updated labels for all modes on ``in_path``.

        Notes
        -----
        This is a convenience helper for path-bound devices whose action
        is naturally expressed as a relabeling of all modes on a selected
        path.

        """
        label_map: dict[Signature, ModeLabelProtocol] = {}

        for mode in self.modes_on_path(in_path):
            old_label = mode.label
            new_label = update_label(old_label)
            if out_path is not None:
                new_label = new_label.with_path(out_path)
            label_map[mode.signature] = new_label

        return self if not label_map else self.relabel_labels(label_map)

    @staticmethod
    def vacuum() -> KetPolyState:
        r"""Construct the vacuum state.

        Returns
        -------
        KetPolyState
            Ket state represented by the identity-only ket polynomial.

        """
        return KetPolyState(KetPoly.identity())

    @staticmethod
    def from_creators(
        creators: Iterable[LadderOpProtocol],
        *,
        coeff: complex = 1.0,
    ) -> KetPolyState:
        r"""Build a ket state from a word of creation operators.

        Parameters
        ----------
        creators:
            Iterable of ladder operators, all of which must be creation
            operators.
        coeff:
            Scalar coefficient multiplying the resulting ket term.

        Returns
        -------
        KetPolyState
            Ket state constructed from the supplied creation-operator
            word.

        Raises
        ------
        ValueError
            If any supplied operator is not a creation operator.

        """
        ops = tuple(creators)
        if any(op.kind != OperatorKind.CRE for op in ops):
            raise ValueError("from_creators expects only creation operators.")

        ket = KetPoly.from_ops(
            creators=ops,
            annihilators=(),
            coeff=coeff,
        ).combine_like_terms()
        return KetPolyState(ket)

    @staticmethod
    def from_ketpoly(ket: KetPoly) -> KetPolyState:
        r"""Construct a state wrapper from a ket polynomial.

        Parameters
        ----------
        ket:
            Ket polynomial to wrap.

        Returns
        -------
        KetPolyState
            New wrapper around the combined ket polynomial.

        """
        return KetPolyState(ket.combine_like_terms())

    def with_label(self, label: str | None) -> KetPolyState:
        r"""Return a copy of the state with a new human-readable label.

        Parameters
        ----------
        label:
            Optional label attached to the state wrapper.

        Returns
        -------
        KetPolyState
            New state instance with updated label.

        Notes
        -----
        This label is metadata only and does not affect the underlying
        polynomial representation.

        """
        return replace(self, label=label)

    def with_index(self, index: int | None) -> KetPolyState:
        r"""Return a copy of the state with a new index.

        Parameters
        ----------
        index:
            Optional state identifier.

        Returns
        -------
        KetPolyState
            New state instance with updated index.

        Notes
        -----
        The index is typically used to distinguish states in simulation
        pipelines or logging but has no semantic meaning for the state
        representation itself.

        """
        return replace(self, index=index)

    def normalized(self, *, eps: float = 1e-14) -> KetPolyState:
        r"""Return a normalized copy of the ket state.

        Parameters
        ----------
        eps:
            Threshold used internally by ket normalization.

        Returns
        -------
        KetPolyState
            New state with normalized ket polynomial.

        """
        return replace(self, ket=self.ket.normalize(eps=eps))

    def is_normalized(self, *, eps: float = 1e-14) -> bool:
        r"""Check whether the ket state is normalized.

        Parameters
        ----------
        eps:
            Numerical tolerance used by the underlying normalization
            check.

        Returns
        -------
        bool
            True if the ket polynomial is normalized within ``eps``.

        """
        return self.ket.is_normalized(eps=eps)

    @property
    def norm2(self) -> float:
        r"""Return the squared norm of the ket state.

        Returns
        -------
        float
            Squared norm of the underlying ket polynomial.

        """
        return self.ket.norm2()

    def to_density(self) -> DensityPolyStateProtocol:
        r"""Convert the ket state to the corresponding pure density state.

        Returns
        -------
        DensityPolyState
            Pure density-state wrapper constructed from this ket state.

        """
        from symop.polynomial.state.density import DensityPolyState

        return DensityPolyState.pure(self)
