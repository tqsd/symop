r"""Phase shifter device models.

This subpackage provides semantic device models for phase-shifting
operations in optical systems.

Currently, it includes:

- :class:`PhaseShifter`: a single-path device that applies a constant
  phase rotation to all modes associated with a selected path.

Notes
-----
Phase shifters are fundamental linear-optical components that implement
single-mode unitary transformations. In this package, they are represented
at the semantic level and rely on backend kernels to perform the actual
operator-level transformation.

"""

from .phase_shifter import PhaseShifter

__all__ = ["PhaseShifter"]
