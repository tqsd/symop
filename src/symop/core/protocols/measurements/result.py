"""Protocols for semantic measurement results.

A measurement result is the classical or hybrid classical-quantum object
returned after evaluating a measurement action.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from symop.core.protocols.measurements.action import MeasurementAction
from symop.core.protocols.states.base import State as StateProtocol
from symop.devices.measurement.outcomes import MeasurementOutcome


@runtime_checkable
class MeasurementResult(Protocol):
    r"""Base protocol for evaluated measurement results."""

    @property
    def action(self) -> MeasurementAction:
        r"""Return the semantic action that produced this result.

        Returns
        -------
        MeasurementAction
            Action that was evaluated.
        """
        ...


@runtime_checkable
class ObservationResult(MeasurementResult, Protocol):
    r"""Result of an observation query returning outcome probabilities."""

    @property
    def probabilities(self) -> Mapping[MeasurementOutcome, object]:
        r"""Return the outcome probability map.

        Returns
        -------
        Mapping[MeasurementOutcome, object]
            Mapping from outcome to probability-like scalar. The scalar is
            left abstract at the protocol level so symbolic and numeric
            backends can both conform.
        """
        ...


@runtime_checkable
class DetectionResult(MeasurementResult, Protocol):
    r"""Result of a detection query.

    Notes
    -----
    Detection is phrased in terms of classical records or detector
    outcomes. The result may represent an exact distribution, a sampled
    record, or another coarse-grained detector response.
    """

    @property
    def record(self) -> object:
        r"""Return the classical detection record.

        Returns
        -------
        object
            Detector-dependent classical record.
        """
        ...

    @property
    def state(self) -> StateProtocol:
        r"""Return the post measurement state."""
        ...


@runtime_checkable
class PostselectionResult(MeasurementResult, Protocol):
    r"""Result of a postselection query."""

    @property
    def outcome(self) -> MeasurementOutcome:
        r"""Return the selected outcome.

        Returns
        -------
        MeasurementOutcome
            Outcome on which the state was conditioned.
        """
        ...

    @property
    def probability(self) -> object:
        r"""Return the probability weight of the selected branch.

        Returns
        -------
        object
            Probability-like scalar of the selected outcome.
        """
        ...

    @property
    def state(self) -> StateProtocol:
        r"""Return the postselected state.

        Returns
        -------
        StateProtocol
            Conditioned output quantum state.
        """
        ...
