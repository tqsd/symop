"""Primitive channel-building operations for CCR polynomial channels.

This package provides low-level building blocks used to construct
physical transformations on polynomial representations. It includes
passive linear mode rewrites, active Bogoliubov rewrites, and generic
unitary-dilation helpers that higher-level models can build upon.
"""

from __future__ import annotations

from .bogoliubov import (
    BogoliubovMap,
)
from .bogoliubov import (
    apply_to_densitypoly as apply_bogoliubov_to_densitypoly,
)
from .bogoliubov import (
    apply_to_ketpoly as apply_bogoliubov_to_ketpoly,
)
from .bogoliubov import (
    apply_to_oppoly as apply_bogoliubov_to_oppoly,
)
from .bogoliubov import (
    make_substitution as make_bogoliubov_substitution,
)
from .linear_mode_unitary import (
    LinearModeMap,
    apply_to_densitypoly,
    apply_to_ketpoly,
    apply_to_oppoly,
    make_substitution,
)
from .unitary_dilation import (
    UnitaryDilation,
    apply_unitary_dilation_densitypoly,
    apply_unitary_dilation_densitypoly_direct,
)

__all__ = [
    "LinearModeMap",
    "make_substitution",
    "apply_to_ketpoly",
    "apply_to_densitypoly",
    "apply_to_oppoly",
    "BogoliubovMap",
    "make_bogoliubov_substitution",
    "apply_bogoliubov_to_ketpoly",
    "apply_bogoliubov_to_densitypoly",
    "apply_bogoliubov_to_oppoly",
    "UnitaryDilation",
    "apply_unitary_dilation_densitypoly",
    "apply_unitary_dilation_densitypoly_direct",
]
