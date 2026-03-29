"""Concrete measurement result objects.

This module defines immutable result objects returned by measurement
runtimes after evaluating semantic measurement actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from symop.core.protocols.states.base import State as StateProtocol
from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.protocols.action import (
    MeasurementAction as MeasurementActionProtocol,
)


@dataclass(frozen=True)
class MeasurementResult:
    r"""Base concrete measurement result.

    Parameters
    ----------
    action:
        Semantic action that produced this result.
    metadata:
        Optional backend-specific metadata.

    """

    action: MeasurementActionProtocol
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationResult(MeasurementResult):
    r"""Result of an observation query.

    Parameters
    ----------
    action:
        Semantic action that produced this result.
    probabilities:
        Mapping from measurement outcomes to probability-like scalars.
    metadata:
        Optional backend-specific metadata.

    """

    probabilities: Mapping[MeasurementOutcome, float] = field(default_factory=dict)
    expectation: float | None = None


@dataclass(frozen=True)
class DetectionResult(MeasurementResult):
    r"""Result of a detection query.

    Parameters
    ----------
    action:
        Semantic action that produced this result.
    record:
        Classical detector-dependent record.
    metadata:
        Optional backend-specific metadata.

    """

    record: object | None = None
    outcome: MeasurementOutcome | None = None
    probability: float | None = None
    state: StateProtocol | None = None


@dataclass(frozen=True)
class PostselectionResult(MeasurementResult):
    r"""Result of a postselection query.

    Parameters
    ----------
    action:
        Semantic action that produced this result.
    outcome:
        Outcome on which the state was conditioned.
    probability:
        Probability-like scalar of the selected branch.
    state:
        Conditioned output quantum state.
    metadata:
        Optional backend-specific metadata.

    """

    outcome: MeasurementOutcome | None = None
    probability: object | None = None
    state: StateProtocol | None = None


if TYPE_CHECKING:
    from symop.devices.protocols.result import (
        DetectionResult as DetectionResultProtocol,
    )
    from symop.devices.protocols.result import (
        MeasurementResult as MeasurementResultProtocol,
    )
    from symop.devices.protocols.result import (
        ObservationResult as ObservationResultProtocol,
    )
    from symop.devices.protocols.result import (
        PostselectionResult as PostselectionResultProtocol,
    )

    _measurement_result_check: MeasurementResultProtocol = MeasurementResult(
        action=None,  # type: ignore[arg-type]
    )
    _observation_result_check: ObservationResultProtocol = ObservationResult(
        action=None,  # type: ignore[arg-type]
    )
    _detection_result_check: DetectionResultProtocol = DetectionResult(
        action=None,  # type: ignore[arg-type]
    )
    _postselection_result_check: PostselectionResultProtocol = PostselectionResult(
        action=None,  # type: ignore[arg-type]
    )
