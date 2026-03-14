"""Typing helpers for CCR protocols.

This module centralizes reusable typing aliases used across CCR protocol and
algebra modules. Keeping these aliases here avoids import cycles and keeps
type signatures consistent across the package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, TypeVar

from symop.core.protocols.ops.operators import LadderOp
from symop.core.protocols.terms.op_term import OpTerm

OpTermT = TypeVar("OpTermT", bound=OpTerm)

OpTermFactory: TypeAlias = Callable[[tuple[LadderOp, ...], complex], OpTermT]
