from __future__ import annotations

from random import Random
from typing import Protocol, runtime_checkable

from symop.core.protocols.modes.labels import Path


@runtime_checkable
class SupportsRng(Protocol):
    @property
    def rng(self) -> Random: ...


@runtime_checkable
class ApplyContext(Protocol):
    """
    Context object used during device planning.

    The planning phase may need to allocate fresh paths or reserve paths so
    later allocations do not collide.
    """

    def allocate_path(
        self,
        *,
        hint: str | None = None,
        avoid: set[Path] | None = None,
    ) -> Path:
        """
        Allocate and return a fresh path label.

        Parameters
        ----------
        hint:
            Optional human-readable prefix or hint for the allocated path.
        avoid:
            Additional paths that must not be returned.
        """
        ...

    def reserve_paths(self, paths: set[Path]) -> None:
        """
        Mark paths as reserved so future allocations avoid them.
        """
        ...
