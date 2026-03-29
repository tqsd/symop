r"""Concrete semantic measurement actions.

This module defines immutable semantic action objects produced by
measurement devices during planning.

A measurement action is a representation-independent request that
combines:

- a semantic measurement intent,
- a measurement specification describing what is measured,
- and intent-specific metadata such as a selected postselection outcome.

These actions are later evaluated by a runtime for a particular backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.measurement.resolution import MeasurementResolution
from symop.devices.measurement.specs import MeasurementSpec
from symop.devices.measurement.target import MeasurementTarget
from symop.devices.types.measurement import (
    MeasurementIntent,
)


@dataclass(frozen=True, kw_only=True)
class MeasurementAction:
    r"""Base semantic measurement action.

    Parameters
    ----------
    measurement_spec:
        Semantic specification of the measurement to be performed.
    destructive:
        Whether the measured subsystem should be discarded after a
        selective measurement update, where applicable.

    Notes
    -----
    The action intent is defined by subclasses such as
    :class:`ObserveAction`, :class:`DetectAction`, and
    :class:`PostselectAction`.

    """

    measurement_spec: MeasurementSpec
    destructive: bool = False

    @property
    def target(self) -> MeasurementTarget:
        r"""Return the semantic measurement target.

        Returns
        -------
        MeasurementTarget
            Target selected by the measurement specification.

        """
        return self.measurement_spec.target

    @property
    def resolution(self) -> MeasurementResolution:
        r"""Return the semantic measurement resolution.

        Returns
        -------
        MeasurementResolution
            Resolution declared by the measurement specification.

        """
        return self.measurement_spec.resolution

    @property
    def outcome(self) -> MeasurementOutcome | None:
        r"""Return the selected outcome, if any.

        Returns
        -------
        MeasurementOutcome or None
            Default base implementation returns ``None``.

        """
        return None

    @property
    def intent(self) -> MeasurementIntent:
        r"""Return the semantic measurement intent.

        Returns
        -------
        str
            Semantic action intent string.

        Raises
        ------
        NotImplementedError
            Always raised by the base class.

        """
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ObserveAction(MeasurementAction):
    r"""Semantic action requesting an observation distribution.

    Notes
    -----
    Observation requests outcome statistics without selecting a single
    outcome branch.

    """

    destructive: bool = False

    @property
    def intent(self) -> MeasurementIntent:
        r"""Return the semantic measurement intent.

        Returns
        -------
        str
            Always ``"observe"``.

        """
        return "observe"


@dataclass(frozen=True, kw_only=True)
class DetectAction(MeasurementAction):
    r"""Semantic action requesting a sampled detector response.

    Notes
    -----
    Detection requests one sampled outcome and the corresponding
    selective state update.

    """

    destructive: bool = True

    @property
    def intent(self) -> MeasurementIntent:
        r"""Return the semantic measurement intent.

        Returns
        -------
        str
            Always ``"detect"``.

        """
        return "detect"


@dataclass(frozen=True, kw_only=True)
class PostselectAction(MeasurementAction):
    r"""Semantic action requesting conditioning on a chosen outcome.

    Parameters
    ----------
    measurement_spec:
        Semantic specification of the measurement to be performed.
    selected_outcome:
        Outcome on which the state should be conditioned.
    destructive:
        Whether the measured subsystem should be discarded after the
        selective update.

    """

    selected_outcome: MeasurementOutcome
    destructive: bool = True

    @property
    def intent(self) -> MeasurementIntent:
        r"""Return the semantic measurement intent.

        Returns
        -------
        str
            Always ``"postselect"``.

        """
        return "postselect"

    @property
    def outcome(self) -> MeasurementOutcome:
        r"""Return the selected outcome.

        Returns
        -------
        MeasurementOutcome
            Outcome on which postselection is conditioned.

        """
        return self.selected_outcome
