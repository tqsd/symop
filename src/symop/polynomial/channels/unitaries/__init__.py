"""Linear-optical unitary helpers.

Provides construction and validation utilities for small unitary matrices
commonly used in photonic models, including beamsplitters, phase shifters,
mode swaps, interferometers, and embedding operations in larger mode spaces.
"""

from __future__ import annotations

from .beamsplitter import (
    beamsplitter_u,
    loss_dilation_u,
)
from .blockdiag import block_diag, embed_1, embed_2, embed_u
from .conventions import require_dim, require_square, require_unitary_optional
from .mzi import mzi_u
from .phase import phase_u
from .swap import swap_u

__all__ = [
    "require_square",
    "require_dim",
    "require_unitary_optional",
    "phase_u",
    "swap_u",
    "beamsplitter_u",
    "loss_dilation_u",
    "block_diag",
    "embed_1",
    "embed_2",
    "embed_u",
    "mzi_u",
]
