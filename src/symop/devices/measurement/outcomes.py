r"""Concrete measurement outcome objects.

This module defines immutable outcome objects used to represent
classical measurement results at the semantic layer.

Each outcome provides a stable hashable key suitable for dictionary use
and a human-readable label suitable for display or debugging.

Included outcome families cover:

- exact number outcomes,
- threshold detector outcomes,
- parity outcomes,
- and joint outcomes combining multiple port-local outcomes.

Notes
-----
These classes represent classical outcome values, not measurement
actions or specifications. They are used by measurement specs, resolved
measurements, and measurement results.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass


@dataclass(frozen=True)
class MeasurementOutcome(ABC):
    r"""Abstract base class for concrete measurement outcomes.

    Subclasses provide a stable key and a display label for a specific
    outcome family.
    """

    @property
    @abstractmethod
    def key(self) -> Hashable:
        r"""Return a stable hashable key for the outcome.

        Returns
        -------
        Hashable
            Canonical key identifying the outcome.

        """
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        r"""Return a human-readable label for the outcome.

        Returns
        -------
        str
            Display label for the outcome.

        """
        ...


@dataclass(frozen=True)
class NumberOutcome(MeasurementOutcome):
    r"""Exact photon-number measurement outcome.

    Parameters
    ----------
    count:
        Measured photon count.

    """

    count: int

    def __post_init__(self) -> None:
        r"""Validate the photon-number outcome.

        Raises
        ------
        ValueError
            If ``count`` is negative.

        """
        if self.count < 0:
            raise ValueError("Photon-number outcome must be non-negative")

    @property
    def key(self) -> Hashable:
        r"""Return the canonical key for the number outcome.

        Returns
        -------
        Hashable
            Tuple identifying the number outcome family and count.

        """
        return ("number", self.count)

    @property
    def label(self) -> str:
        r"""Return the display label for the number outcome.

        Returns
        -------
        str
            Decimal string representation of the count.

        """
        return str(self.count)


@dataclass(frozen=True)
class ThresholdOutcome(MeasurementOutcome):
    r"""Threshold-detector measurement outcome.

    Parameters
    ----------
    clicked:
        Whether the detector reported a click.

    """

    clicked: bool

    @property
    def key(self) -> Hashable:
        r"""Return the canonical key for the threshold outcome.

        Returns
        -------
        Hashable
            Tuple identifying the threshold outcome family and click
            status.

        """
        return ("threshold", self.clicked)

    @property
    def label(self) -> str:
        r"""Return the display label for the threshold outcome.

        Returns
        -------
        str
            ``"click"`` if the detector clicked, otherwise
            ``"no-click"``.

        """
        return "click" if self.clicked else "no-click"


@dataclass(frozen=True)
class ParityOutcome(MeasurementOutcome):
    r"""Parity measurement outcome.

    Parameters
    ----------
    parity:
        Measured parity label, either ``"even"`` or ``"odd"``.

    """

    parity: str

    def __post_init__(self) -> None:
        r"""Validate the parity outcome.

        Raises
        ------
        ValueError
            If ``parity`` is not ``"even"`` or ``"odd"``.

        """
        if self.parity not in {"even", "odd"}:
            raise ValueError("Parity outcome must be 'even' or 'odd'.")

    @property
    def key(self) -> Hashable:
        r"""Return the canonical key for the parity outcome.

        Returns
        -------
        Hashable
            Tuple identifying the parity outcome family and parity
            label.

        """
        return ("parity", self.parity)

    @property
    def label(self) -> str:
        r"""Return the display label for the parity outcome.

        Returns
        -------
        str
            The parity label.

        """
        return self.parity


@dataclass(frozen=True)
class JointOutcome(MeasurementOutcome):
    r"""Joint measurement outcome over multiple ports.

    Parameters
    ----------
    outcomes_by_port:
        Ordered collection mapping each logical port name to its local
        measurement outcome.

    """

    outcomes_by_port: tuple[tuple[str, MeasurementOutcome], ...]

    @property
    def key(self) -> Hashable:
        r"""Return the canonical key for the joint outcome.

        Returns
        -------
        Hashable
            Tuple identifying the joint outcome family together with the
            canonical keys of all port-local outcomes.

        """
        return (
            "joint",
            tuple((port, outcome.key) for port, outcome in self.outcomes_by_port),
        )

    @property
    def label(self) -> str:
        r"""Return the display label for the joint outcome.

        Returns
        -------
        str
            Comma-separated ``port=label`` representation of all local
            outcomes.

        """
        return ", ".join(
            f"{port}={outcome.label}" for port, outcome in self.outcomes_by_port
        )
