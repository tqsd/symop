"""Protocols for semantic measurement actions.

A measurement action is the planned semantic request that later runtime
layers evaluate for a particular state representation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.measurement.target import MeasurementTarget


@runtime_checkable
class MeasurementAction(Protocol):
    r"""Semantic measurement action.

    Notes
    -----
    This is a representation-independent measurement request. Concrete
    frameworks may refine it into observation, detection, and
    postselection actions.
    """

    @property
    def intent(self) -> str:
        r"""Return the semantic measurement intent.

        Returns
        -------
        str
            Stable intent identifier such as ``"observe"``,
            ``"detect"``, or ``"postselect"``.
        """
        ...

    @property
    def target(self) -> MeasurementTarget:
        r"""Return the semantic target of the measurement.

        Returns
        -------
        MeasurementTarget
            Target subsystem on which the action is defined.
        """
        ...

    @property
    def outcome(self) -> MeasurementOutcome | None:
        r"""Return the selected outcome, if any.

        Returns
        -------
        MeasurementOutcome | None
            Requested outcome for outcome-conditioned actions such as
            postselection, or ``None`` if no specific outcome is fixed.
        """
        ...
