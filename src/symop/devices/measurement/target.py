r"""Semantic measurement target definitions.

This module defines concrete data structures used to describe the
subsystems on which a measurement is performed.

A measurement target is representation-independent and refers only to
stable semantic identifiers such as logical detector ports, paths, and
explicit modes signatures. It does not depend on a particular state
representation.

The model is structured in two layers:

``MeasurementSelection``
    Describes the subsystem selected for a single logical detector port.

``MeasurementTarget``
    Groups one or more selection into a full measurement target.

This grouped structure enables support for:

- single-port measurements,
- multi-port measurements,
- coincidence detection,
- per-port or joint readout strategies.

Notes
-----
The target only specifies *where* the measurement acts. It does not
specify *how* the measurement is performed. That is the role of the
measurement specification and resolution layers.

"""

from __future__ import annotations

from dataclasses import dataclass, field

from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.core.types.signature import Signature


@dataclass(frozen=True)
class MeasurementSelection:
    r"""Selection attached to one logical detector port.

    A measurement selection defines which subsystem is associated with a
    single logical detector port. The subsystem may be specified usingg
    semantic path identifiers, explicit mode signatures, or both.

    Parameters
    ----------
    port_name:
        Logical name of the detector port.
    paths:
        Tuple of semantic path identifiers included in the selection.
    mode_sigs:
        Tuple of explicit mode signatures included in the selection.

    Notes
    -----
    Duplicate entries in ``paths`` and ``mode_sigs`` are removed during
    initialization while preserving order.

    """

    port_name: str
    paths: tuple[PathProtocol, ...] = field(default_factory=tuple)
    mode_sigs: tuple[Signature, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Normalize selection by removing duplicate entries.

        Notes
        -----
        Duplicate paths and mode signatures are removed while preserving
        their original order.

        """
        object.__setattr__(self, "paths", tuple(dict.fromkeys(self.paths)))
        object.__setattr__(
            self,
            "mode_sigs",
            tuple(dict.fromkeys(self.mode_sigs)),
        )

    @property
    def is_empty(self) -> bool:
        """Return True if the selection contains no subsystems.

        Returns
        -------
        bool
            True if both ``paths`` and ``mode_sigs`` are empty.

        """
        return not self.paths and not self.mode_sigs

    @property
    def has_paths(self) -> bool:
        """Return True if the selection includes path identifiers.

        Returns
        -------
        bool
            True if at least one path is present.

        """
        return bool(self.paths)

    @property
    def has_mode_sigs(self) -> bool:
        """Return True if the selection includes explicit modes signatures.

        Returns
        -------
        bool
            True if at least one mode signature is present.

        """
        return bool(self.mode_sigs)


@dataclass(frozen=True)
class MeasurementTarget:
    """Concrete semantic measurement target.

    A measurement target groups one or more
    :class:`MeasurementSelection` objects into a complete specification
    of the subsystem on which a measurement is defined.

    Each selection corresponds to one logical port, allowing
    multi-port and coincidence measurements to be represented.

    Parameters
    ----------
    selection:
        Tuple of measurement selections, one per logical detector port.

    Notes
    -----
    - Each ``port_name`` must be unique.
    - The target may be empty if all selections are empty.

    """

    selections: tuple[MeasurementSelection, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate uniqueness of port names.

        Raises
        ------
        ValueError
            If duplicate port names are present in ``selection``.

        """
        seen: set[str] = set()
        normalized: list[MeasurementSelection] = []

        for selection in self.selections:
            if selection.port_name in seen:
                raise ValueError(
                    "MeasurementTarget.selections must not contain "
                    f"duplicate port names: {selection.port_name!r}"
                )
            seen.add(selection.port_name)
            normalized.append(selection)

        object.__setattr__(self, "selections", tuple(normalized))

    @property
    def is_empty(self) -> bool:
        """Return True if the target is empty.

        Returns
        -------
        bool
            True if all selections are empty.

        """
        return all(selection.is_empty for selection in self.selections)

    @property
    def paths(self) -> tuple[PathProtocol, ...]:
        """Return all unique paths across all selections.

        Returns
        -------
        tuple[PathProtocol,...]
            Flattened tuple of unique paths appearing in any selection,
            preserving first occurence order.

        """
        out: list[PathProtocol] = []
        seen: set[PathProtocol] = set()
        for selection in self.selections:
            for path in selection.paths:
                if path not in seen:
                    seen.add(path)
                    out.append(path)
        return tuple(out)

    @property
    def mode_sigs(self) -> tuple[Signature, ...]:
        """Return all unique mode signatures across all selections.

        Returns
        -------
        tuple[Signature, ...]
            Flattened tuple of unique mode signatures appearing in any
            selection, preserving first occurence order.

        """
        out: list[Signature] = []
        seen: set[Signature] = set()
        for selection in self.selections:
            for sig in selection.mode_sigs:
                if sig not in seen:
                    seen.add(sig)
                    out.append(sig)
        return tuple(out)

    @property
    def has_paths(self) -> bool:
        """Return True if any selection includes path identifiers.

        Returns
        -------
        bool
            True if at least one path is present across all selections.

        """
        return bool(self.paths)

    @property
    def has_mode_sigs(self) -> bool:
        """Return True if any selection includes explicit mode signatures.

        Returns
        -------
        bool
            True if at least one mode signature is present across all
            selections.

        """
        return bool(self.mode_sigs)
