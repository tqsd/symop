"""Polynomial density-state wrapper.

This module provides a lightweight state container around
:class:`~symop.ccr.algebra.density.poly.DensityPoly`, exposing
semantic information about modes, paths, and labels derived from
the polynomial representation.

The wrapper serves several roles:

- attaches lightweight metadata (label, index) to a density polynomial
- exposes convenient views of the modes contained in the state
- supports relabeling operations used by device kernels
- provides normalization and trace utilities

Important:
---------
Mode labels are **not stored independently**. They are derived directly
from the concrete modes present in the density polynomial, ensuring
that semantic information always reflects the actual algebraic state.

This avoids inconsistencies between symbolic operators and state
metadata when transformations rewrite the polynomial.

"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import count
from typing import TYPE_CHECKING, Literal

from symop.ccr.algebra.density.poly import DensityPoly
from symop.core.protocols.base.signature import Signature
from symop.core.protocols.devices.label_edit import (
    DeleteModeLabel,
    LabelEdit,
    SetModeLabel,
)
from symop.core.protocols.modes.labels import (
    ModeLabel as ModeLabelProtocol,
)
from symop.core.protocols.modes.labels import (
    Path as PathProtocol,
)
from symop.core.protocols.ops.operators import ModeOp as ModeOpProtocol
from symop.devices.measurement.target import MeasurementTarget
from symop.polynomial.protocols.density import (
    DensityPolyState as DensityPolyStateProtocol,
)
from symop.polynomial.protocols.ket import KetPolyState as KetPolyStateProtocol
from symop.polynomial.rewrites.relabel_modes import density_relabel_modes

_state_counter = count(1)


@dataclass(frozen=True)
class DensityPolyState(DensityPolyStateProtocol):
    r"""Polynomial density state wrapper.

    Notes
    -----
    Semantic mode information is derived from the concrete modes present in
    ``rho``. Labels are not stored independently; they are derived from the
    unique modes appearing in the density polynomial.

    """

    rho: DensityPoly
    label: str | None = None
    index: int | None = field(default_factory=lambda: next(_state_counter))

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
    def state_kind(self) -> Literal["density"]:
        r"""Return the state kind of this object.

        Returns
        -------
        Literal["density"]
            The density-state kind.

        """
        return "density"

    @cached_property
    def _modes_cached(self) -> tuple[ModeOpProtocol, ...]:
        r"""Cache the unique modes appearing in the density polynomial.

        Returns
        -------
        tuple[ModeOpProtocol, ...]
            Unique modes extracted from ``rho`` in the order provided by
            ``DensityPoly.unique_modes``.

        """
        return self.rho.unique_modes

    @property
    def modes(self) -> tuple[ModeOpProtocol, ...]:
        r"""Return the unique modes present in the density polynomial.

        Returns
        -------
        tuple[ModeOpProtocol, ...]
            Unique modes appearing in ``rho``.

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
    def modes_by_path(
        self,
    ) -> dict[PathProtocol, tuple[ModeOpProtocol, ...]]:
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

    def labels_on_path(
        self,
        path: PathProtocol,
    ) -> dict[Signature, ModeLabelProtocol]:
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
        mode_map: Mapping[Signature, ModeOpProtocol],
    ) -> DensityPolyState:
        r"""Return a new density state with selected ``ModeOp``s relabeled.

        Parameters
        ----------
        mode_map:
            Mapping from old labels to replacement label.

        Returns
        -------
        DensityPolyState
            New state with all matching mode paths updated.

        Notes
        -----
        Only modes whose current path appears in ``mode_map`` are modified.
        Other modes are left unchanged.

        """
        rho2 = DensityPoly(density_relabel_modes(self.rho.terms, mode_map=mode_map))
        return replace(self, rho=rho2)

    def relabel_paths(
        self,
        path_map: Mapping[PathProtocol, PathProtocol],
    ) -> DensityPolyState:
        r"""Return a new density state with selected paths relabeled.

        Parameters
        ----------
        path_map:
            Mapping from old paths to replacement paths.

        Returns
        -------
        DensityPolyState
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
        label_map: Mapping[Signature, ModeLabelProtocol],
    ) -> DensityPolyState:
        r"""Return a new density state with selected labels replaced.

        Parameters
        ----------
        label_map:
            Mapping from mode signature to replacement label.

        Returns
        -------
        DensityPolyState
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
        edits: Sequence[LabelEdit],
    ) -> DensityPolyState:
        r"""Apply semantic label edits to the state.

        Parameters
        ----------
        edits:
            Sequence of label edits to apply.

        Returns
        -------
        DensityPolyState
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
        Since labels are derived from concrete modes in the density
        polynomial, edits are implemented by rebuilding the affected modes
        and relabeling them algebraically.

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
                    "from the density polynomial. Remove or replace the mode "
                    "algebraically instead."
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
    ) -> DensityPolyState:
        r"""Apply a label update to all modes on an input path.

        Parameters
        ----------
        in_path:
            Path whose modes should be updated.
        out_path:
            Optional replacement path for the affected modes. If None, the
            original paths are retained.
        update_label:
            Callable that transforms each mode label before optional path
            replacement.

        Returns
        -------
        DensityPolyState
            New state with updated labels for all modes on ``in_path``.

        Notes
        -----
        This is a convenience helper for path-bound devices whose action is
        naturally expressed as a relabeling of all modes on a selected path.

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
    def vacuum() -> DensityPolyState:
        r"""Construct the vacuum density state.

        Returns
        -------
        DensityPolyState
            Density state wrapping the identity / vacuum polynomial used by
            the density representation.

        """
        return DensityPolyState(DensityPoly.identity())

    @staticmethod
    def pure(ket: KetPolyStateProtocol) -> DensityPolyState:
        r"""Construct a pure density state from a ket state.

        Parameters
        ----------
        ket:
            Ket state representing the pure state.

        Returns
        -------
        DensityPolyState
            Density-state wrapper containing the corresponding pure density
            polynomial.

        """
        return DensityPolyState(DensityPoly.pure(ket.ket).combine_like_terms())

    @staticmethod
    def from_densitypoly(rho: DensityPoly) -> DensityPolyState:
        r"""Construct a state wrapper from a density polynomial.

        Parameters
        ----------
        rho:
            Density polynomial to wrap.

        Returns
        -------
        DensityPolyState
            New wrapper around the combined density polynomial.

        """
        return DensityPolyState(rho.combine_like_terms())

    def with_label(self, label: str | None) -> DensityPolyState:
        r"""Return a copy of the state with a new human-readable label.

        Parameters
        ----------
        label:
            Optional label attached to the state wrapper.

        Returns
        -------
        DensityPolyState
            New state instance with updated label.

        Notes
        -----
        This label is metadata only and does not affect the underlying
        polynomial representation.

        """
        return replace(self, label=label)

    def with_index(self, index: int | None) -> DensityPolyState:
        r"""Return a copy of the state with a new index.

        Parameters
        ----------
        index:
            Optional state identifier.

        Returns
        -------
        DensityPolyState
            New state instance with updated index.

        Notes
        -----
        The index is typically used to distinguish states in simulation
        pipelines or logging but has no semantic meaning for the state
        representation itself.

        """
        return replace(self, index=index)

    def trace(self) -> complex:
        r"""Return the trace of the density polynomial.

        Returns
        -------
        complex
            Trace of the density operator represented by ``rho``.

        Notes
        -----
        For a properly normalized quantum state the trace should equal 1.

        """
        return self.rho.trace()

    def normalize_trace(self, *, eps: float = 1e-14) -> DensityPolyState:
        r"""Return a trace-normalized copy of the state.

        Parameters
        ----------
        eps:
            Threshold below which the trace is treated as zero.

        Returns
        -------
        DensityPolyState
            New state with density polynomial scaled to unit trace.

        Raises
        ------
        ValueError
            If the trace is numerically too close to zero.

        """
        tr = self.trace()
        if abs(tr) <= eps:
            raise ValueError("Cannot normalize trace: near-zero trace.")
        return replace(self, rho=self.rho.scaled(1.0 / tr))

    def is_trace_normalized(self, *, eps: float = 1e-14) -> bool:
        r"""Check whether the density operator is trace normalized.

        Parameters
        ----------
        eps:
            Numerical tolerance used when comparing to 1.

        Returns
        -------
        bool
            True if ``|Tr(rho) - 1| <= eps``.

        """
        return abs(self.trace() - 1.0) <= eps

    def multiply(self, other: DensityPolyState) -> DensityPolyState:
        r"""Return the symbolic product of two density states.

        Parameters
        ----------
        other:
            Right-hand density state.

        Returns
        -------
        DensityPolyState
            State wrapping the product density polynomial.

        """
        return DensityPolyState.from_densitypoly(self.rho.multiply(other.rho))

    def join(self, other: DensityPolyState) -> DensityPolyState:
        r"""Return the algebraic join of two density states.

        Parameters
        ----------
        other:
            Density state to combine with this state.

        Returns
        -------
        DensityPolyState
            Product state obtained by multiplying the two underlying density
            polynomials.

        Notes
        -----
        This is a convenience alias for :meth:`multiply`. It is useful in
        contexts where combining density-state branches or subsystems is more
        naturally described as joining them.

        """
        return self.multiply(other)

    def trace_out_modes(
        self,
        trace_over_modes: Sequence[ModeOpProtocol],
    ) -> DensityPolyState:
        r"""Return the reduced density state after tracing out selected modes.

        Parameters
        ----------
        trace_over_modes:
            Sequence of concrete modes to be traced out of the density
            polynomial.

        Returns
        -------
        DensityPolyState
            Reduced density-state wrapper obtained after partial trace over
            the selected modes.

        Notes
        -----
        Modes not listed in ``trace_over_modes`` are retained. The resulting
        density polynomial is combined before the new wrapper is returned.

        """
        rho2 = self.rho.partial_trace(trace_over_modes).combine_like_terms()
        return replace(self, rho=rho2)

    def trace_out_signatures(
        self,
        mode_sigs: Sequence[Signature],
    ) -> DensityPolyState:
        r"""Return the reduced density state after tracing out selected modes.

        Parameters
        ----------
        mode_sigs:
            Sequence of mode signatures identifying the modes to trace out.

        Returns
        -------
        DensityPolyState
            Reduced density-state wrapper obtained after partial trace over
            the selected modes.

        Notes
        -----
        Only signatures present in the state are used. Missing signatures are
        ignored.

        """
        modes = [
            self.mode_by_signature[sig]
            for sig in mode_sigs
            if sig in self.mode_by_signature
        ]
        return self.trace_out_modes(modes)

    def resolve_target_modes(
        self, target: MeasurementTarget
    ) -> tuple[ModeOpProtocol, ...]:
        r"""Resolve a semantic measurement target into concrete modes.

        Parameters
        ----------
        target:
            Semantic measurement target specifying paths and/or explicit
            mode signatures to be selected.

        Returns
        -------
        tuple[ModeOpProtocol, ...]
            Concrete modes selected by the target. Modes referenced both
            by path and by explicit signature are returned only once.

        Notes
        -----
        Resolution proceeds by first collecting all modes on the target
        paths and then adding any explicitly requested mode signatures.
        Missing explicit signatures are ignored.

        """
        selected: dict[Signature, ModeOpProtocol] = {}

        for path in target.paths:
            for resolved_mode in self.modes_on_path(path):
                selected[resolved_mode.signature] = resolved_mode

        for mode_sig in target.mode_sigs:
            maybe_mode = self.mode_by_signature.get(mode_sig)
            if maybe_mode is not None:
                selected[maybe_mode.signature] = maybe_mode

        return tuple(selected.values())


if TYPE_CHECKING:
    _density_check: DensityPolyStateProtocol = DensityPolyState.vacuum()
