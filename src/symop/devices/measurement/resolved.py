r"""Resolved measurement objects.

This module defines small immutable containers for measurement
specifications together with resolved probability data produced during
measurement evaluation.

The classes here are useful as intermediate or backend-facing objects
that combine a semantic measurement specification with concrete outcome
probabilities. Specialized subclasses may expose derived statistics such
as expectation values.

Notes
-----
These objects represent resolved measurement content rather than
high-level semantic measurement requests. They are therefore distinct
from action and result objects used elsewhere in the measurement
pipeline.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from symop.devices.measurement.outcomes import (
    MeasurementOutcome,
    NumberOutcome,
)
from symop.devices.measurement.specs import (
    MeasurementSpec,
    ProjectiveNumberMeasurementSpec,
)

OutcomeT = TypeVar("OutcomeT", bound=MeasurementOutcome)
SpecT = TypeVar("SpecT", bound=MeasurementSpec)


@dataclass(frozen=True)
class ResolvedMeasurement(Generic[SpecT]):
    r"""Base resolved measurement container.

    Parameters
    ----------
    spec:
        Semantic measurement specification associated with the resolved
        data.

    """

    spec: SpecT


@dataclass(frozen=True)
class ResolvedOutcomeMeasurement(
    ResolvedMeasurement[SpecT],
    Generic[SpecT, OutcomeT],
):
    r"""Resolved measurement with explicit outcome probabilities.

    Parameters
    ----------
    spec:
        Semantic measurement specification associated with the resolved
        probabilities.
    probabilities:
        Mapping from measurement outcomes to their probabilities.

    """

    probabilities: Mapping[OutcomeT, float] = field(default_factory=dict)

    @property
    def outcomes(self) -> tuple[OutcomeT, ...]:
        r"""Return the resolved outcomes.

        Returns
        -------
        tuple[OutcomeT, ...]
            Outcomes present in the probability mapping, in mapping
            iteration order.

        """
        return tuple(self.probabilities.keys())


@dataclass(frozen=True)
class ResolvedProjectiveNumberMeasurement(
    ResolvedOutcomeMeasurement[
        ProjectiveNumberMeasurementSpec,
        NumberOutcome,
    ]
):
    r"""Resolved projective number measurement.

    This class represents resolved probabilities for an ideal projective
    number measurement and provides convenience access to the number
    expectation value.

    Parameters
    ----------
    spec:
        Projective number-measurement specification associated with the
        resolved probabilities.
    probabilities:
        Mapping from number outcomes to their probabilities.

    """

    spec: ProjectiveNumberMeasurementSpec

    @property
    def expectation(self) -> float:
        r"""Return the photon-number expectation value.

        Returns
        -------
        float
            Expected photon number computed from the resolved outcome
            probabilities.

        """
        return sum(outcome.count * prob for outcome, prob in self.probabilities.items())
