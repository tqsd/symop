r"""Type aliases for semantic measurement resolution.

This module defines small literal-based type aliases used to describe
how a measured device resolves a selected subsystems and how its outcomes
are reported at the semantic layer.

The goal of these types is to keep measurement planning explicit and
representation-independent. A semantic measurement action should be able to
state, in a compact and structured way:

- which detector-visible axes are being resolved,
- how outcomes across one or moreports are grouped,
- and what readout family the detector reports.

These aliases do not describe the concrete backend implementation of a
measurement. In particular, they do not specify the actual POVM effects,
Kraus operators, or state-update rule. Those belond to a lower layer
such as a measurement specification or instrument description.

Overview
--------
A measurement resolution is described along three types:

``MeasurementAxis``
    Describe the physical or semantic axes in which the detector
    resolves the selected subsystem. Multiple axis can be defined
    for a single detector.

``MeasurementGrouping``
    Describes how outcomes from multiple logical detector ports are
    combined.

``MeasurementReadout``
    Describes the family of detector outcomes that should be reported.

These three types are rintentionally separated becauss they answer different
questions:

- "axis": what structure of the subsystem is visible to the detector?
- "grouping": how are multiple detector channels combined into outcome?
- "readout": what kind of classical record is produced?

Measurement Axis
------------------
``MeasurementAxis`` specifies the detector-visible decomposition of the
selected subsystems.

The supported values are:
``"path"``
    Resolve the measurement in terms of detector paths or channels.
    This is the most common default for ideal path-local detectors.
    For example, a number detector on one input path typically measures
    the total photon number associated with the path-visible subsystems.

``"mode"``
    Resolve the measurement with respect to explicitly selected modes.
    This is more fine-grained and representation-sensitive option and is
    typically most useful when the measured modes are already well defined
    at the semantic level.

``"time_bin"``
    Resolve the measurement in terms of temporal bins or gated detection
    windows. This is useful for early/late-bin detectors or any measurement
    whose outcomes depend on arrival-time structure.

``"polarization"``
    Resolve the measurement with respect to polarization subspaces or
    polarization-resolved detector channels.

Measurement grouping
--------------------
``MeasurementGrouping`` specifies how outcomes from multiple logical ports
are combined.

Supported values are:
``"total"``
    Exact-number readout, such as 0, 1, 2, ...
``"threshold"``
    Threshold or bucket detector readout (``"click"``/``"no-click"``)
``"custom"``
    A custom readout family defined elswhere, for example by an explicit
    POVM or instrument specification. This value indicates that the literal
    alone is not sufficient to determine the outcome set.

Notes
-----
These aliases are intended for semantic planning and validation. They are
small on purpose and should remain stable and easy to inspect.

For arbitrary POVMs or more general instruments, these types should be
supplemented by a richer measurement-specification objects carrying the
explicit outcome model and operator-level data required for execution.

"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias

MeasurementIntent: TypeAlias = Literal["observe", "detect", "postselect"]

MeasurementAxis: TypeAlias = Literal["path", "mode", "time_bin", "polarization"]


MeasurementGrouping: TypeAlias = Literal["total", "per_port", "joint_ports"]


MeasurementReadout: TypeAlias = Literal["number", "treshold", "parity", "custom"]


class MeasurementIntentEnum(StrEnum):
    OBSERVE = "observe"
    DETECT = "detect"
    POSTSELECT = "postselect"


class MeasurementAxisEnum(StrEnum):
    PATH = "path"
    MODE = "mode"
    TIME_BIN = "time_bin"
    POLARIZATION = "polarization"


class MeasurementGrouppingEnum(StrEnum):
    TOTAL = "total"
    PER_PORT = "per_port"
    JOINT_PORTS = "joint_ports"


class MeasurementReadoutEnum(StrEnum):
    NUMBER = "number"
    THRESHOLD = "threshold"
    PARITY = "parity"
    CUSTOM = "custom"
