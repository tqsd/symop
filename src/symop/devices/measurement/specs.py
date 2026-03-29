r"""Semantic measurement specification objects.

This module defines representation-independent measurement
specifications used by semantic measurement devices and runtime
dispatch.

The classes in this module describe what is measured, how outcomes are
reported, and, for more general measurements, which effects or
state-update operations are associated with each outcome.

Included specification families cover:

- generic measurement specifications,
- projective measurements,
- projective number measurements,
- POVMs,
- and measurement instruments.

Notes
-----
These specifications are semantic objects. They do not themselves
perform numerical measurement evaluation. Backend-specific kernels are
responsible for interpreting and executing them.

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.measurement.resolution import MeasurementResolution
from symop.devices.measurement.target import MeasurementTarget


@dataclass(frozen=True)
class MeasurementSpec:
    r"""Base semantic measurement specification.

    Parameters
    ----------
    target:
        Semantic subsystem on which the measurement is defined.
    resolution:
        Semantic resolution and reporting structure of the measurement.

    """

    target: MeasurementTarget
    resolution: MeasurementResolution = field(default_factory=MeasurementResolution)


@dataclass(frozen=True)
class ProjectiveMeasurementSpec(MeasurementSpec):
    r"""Base semantic specification for projective measurements.

    A projective measurement is defined by mutually orthogonal outcome
    projectors whose sum equals the identity on the measured subsystem.

    Notes
    -----
    Concrete subclasses determine the actual projective family, such as
    photon-number projectors or parity projectors.

    """


@dataclass(frozen=True)
class ProjectiveNumberMeasurementSpec(ProjectiveMeasurementSpec):
    r"""Semantic specification for ideal projective number measurement.

    This spec represents an exact photon-number measurement on the
    selected subsystem.

    Notes
    -----
    The selected subsystem is defined by ``target``, while ``resolution``
    determines whether number is reported as a total count, per-port
    count, joint multi-port count, or along additional resolved axes
    such as time bins.

    A backend may execute this spec either through a specialized fast
    path or by lowering it to a generic POVM or instrument form.

    """


@dataclass(frozen=True)
class POVMSpec(MeasurementSpec):
    r"""Semantic specification for a general POVM measurement.

    A POVM measurement is defined by a finite family of positive
    semidefinite effects whose sum equals the identity on the measured
    subsystem.

    Parameters
    ----------
    outcomes:
        Explicit outcome set of the measurement.
    effects:
        Mapping from outcome to the corresponding POVM effect object.

    Notes
    -----
    This specification is sufficient to define observation
    probabilities. It does not by itself uniquely determine the
    post-measurement state update for selective measurements such as
    detection or postselection. For that, use :class:`InstrumentSpec`.

    The concrete type of each effect is intentionally left abstract at
    this layer. A backend may interpret effects as symbolic operators,
    matrices, or lazily constructed objects.

    """

    outcomes: tuple[MeasurementOutcome, ...] = field(default_factory=tuple)
    effects: Mapping[MeasurementOutcome, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        r"""Validate POVM outcome and effect definitions.

        Raises
        ------
        ValueError
            If the outcome set is empty, contains duplicates, or if an
            effect is missing for any declared outcome.

        """
        if not self.outcomes:
            raise ValueError("POVMSpec.outcomes must not be empty.")

        if tuple(dict.fromkeys(self.outcomes)) != self.outcomes:
            raise ValueError("POVMSpec.outcomes must not contain duplicates.")

        missing = [outcome for outcome in self.outcomes if outcome not in self.effects]
        if missing:
            raise ValueError(
                "POVMSpec.effects must define an effect for every outcome. "
                f"Missing outcomes: {missing!r}"
            )


@dataclass(frozen=True)
class InstrumentSpec(POVMSpec):
    r"""Semantic specification for a general measurement instrument.

    A measurement instrument defines not only outcome probabilities but
    also outcome-conditioned state updates. Each outcome is associated
    with one or more operation objects, such as Kraus operators.

    Parameters
    ----------
    operations:
        Mapping from outcome to the corresponding outcome-conditioned
        operation objects.

    Notes
    -----
    An instrument induces a POVM by summing the adjoint-product of the
    outcome operations. This class therefore refines :class:`POVMSpec`
    by adding explicit state-update semantics.

    The concrete type of each operation is intentionally left abstract
    at this layer. A backend may interpret operations as symbolic
    operators, matrices, channels, or lazily constructed objects.

    """

    operations: Mapping[MeasurementOutcome, tuple[object, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        r"""Validate instrument outcome, effect, and operation definitions.

        Raises
        ------
        ValueError
            If required operations are missing for any declared outcome or
            if any declared outcome is assigned an empty operation tuple.

        """
        super().__post_init__()

        missing = [
            outcome for outcome in self.outcomes if outcome not in self.operations
        ]
        if missing:
            raise ValueError(
                "InstrumentSpec.operations must define operations for every outcome. "
                f"Missing outcomes: {missing!r}"
            )

        empty = [
            outcome
            for outcome, ops in self.operations.items()
            if outcome in self.outcomes and not ops
        ]
        if empty:
            raise ValueError(
                "InstrumentSpec.operations must not contain empty operation tuples. "
                f"Empty outcomes: {empty!r}"
            )
