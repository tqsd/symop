r"""Measurement resolution specification.

This module defines the :class:`MeasurementResolution` dataclass, which
describes how a measurement device resolves a selected subsystem and how
its outcomes are reported at the semantic layer.

The resolution is defined along three orthogonal axes:

- **axes**:
    The detector-visible decomposition of the subsystem (e.g. path,
    mode, time-bin, polarization). Multiple axis may be combined.

- **grouping**:
    How outcomes from multiple logical detector ports are combined
    (e.g. total, per-port, or joint-outcomes).

- **readout**:
    The family of classical outcomes produced by the detector
    (e.g. exact number, threshold, parity, or custom).

This specification is purely semantic and representation-independent.
It does not define how a measurement is implemented internally
(e.g. via POVMs, Kraus operators, or symbolic projections). Instead,
it provides structured information used during measurement planning
and dispatch to select or construct appropriate measurement kernels.

Notes
-----
The resolution object is typically embedded into measurement actions
and later interpreted by the runtime or kernel layer.

For arbitrary POVMs or general instrument specification that defines the
explicit outcome model and operator-level behavior.

"""

from __future__ import annotations

from dataclasses import dataclass

from symop.devices.types.measurement import (
    MeasurementAxis,
    MeasurementGrouping,
    MeasurementReadout,
)


@dataclass(frozen=True)
class MeasurementResolution:
    r"""Semantic description of measurement resolution.

    Parameters
    ----------
    axes:
        Tuple of measurement axes describing which degrees of freedom
        of the selected subsystem are resolved by the detector.

        Common values include:

        - ``"path"``: resolve by detector path or channel
        - ``"time_bin"``: resolve by temporal bins
        - ``"polarization"``: resolve by polarization subspaces
        - ``"mode"``: resolve by explicit selected modes

        Multiple axes may be combined, for example
        ``("path", "time_bin")``.

    grouping:
        Strategy for combining outcomes across multiple logical detector
        ports.

        - ``"total"``:
          Aggregate all selected inputs into a single outcome.
        - ``"per_port"``:
          Produce separate outcomes for each port.
        - ``"joint_ports"``:
          Produce a single joint outcome across all ports.

    readout:
        Family of classical outcomes reported by the detector.

        - ``"number"``:
          Exact-number readout (0, 1, 2, ...).
        - ``"threshold"``:
          Binary click / no-click readout.
        - ``"parity"``:
          Even / odd readout.
        - ``"custom"``:
          Readout defined externally (e.g. by a POVM or instrument
          specification).

    Notes
    -----
    This class does not define the measurement operators themselves.
    It only specifies how the measurement is *interpreted* at the
    semantic level.

    Validation is performed in :meth:`__post_init__` to ensure that
    the resolution is well-formed.

    """

    axes: tuple[MeasurementAxis, ...] = ("path",)
    grouping: MeasurementGrouping = "total"
    readout: MeasurementReadout = "number"

    def __post_init__(self) -> None:
        r"""Validate the measurement resolution.

        Raises
        ------
        ValueError
            If the axes tuple is empty or contains duplicate entries.

        Notes
        -----
        The validation enforces basic structural constraints:

        - At least one measurement axis must be specified.
        - Measurement axes must be unique.

        More advanced consistency checks (e.g. compatibility between
        axes and readout types) may be added in later stages of the
        measurement framework.

        """
        if not self.axes:
            raise ValueError("MeasurementResolution.axes must not be empty")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("MeasurementResolution.axes must not contain duplicates")
