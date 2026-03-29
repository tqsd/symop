r"""Measurement device models.

This package contains semantic models of measurement devices used to
extract classical information from quantum states.

Detector models define how measurement targets, resolutions, and
readouts are constructed. They do not perform the measurement
themselves; instead, they produce semantic measurement actions that are
evaluated by backend-specific measurement kernels.

Supported measurement types include:

- single-mode number detection,
- joint (coincidence) measurements across multiple ports,
- POVM-based detectors,
- postselection and conditional measurements.

Notes
-----
All detectors in this package are representation-independent and operate
at the planning level. The numerical evaluation of measurement outcomes
is delegated to measurement kernels.

Additional detector types that naturally fit here include:

- threshold detectors,
- time-resolved detectors,
- homodyne and heterodyne detectors,
- imperfect detectors (with loss, dark counts, jitter),
- adaptive or feedback-based measurement devices.

"""

from .coincidence_detector import CoincidenceDetector
from .number_detector import NumberDetector

__all__ = ["CoincidenceDetector", "NumberDetector"]
