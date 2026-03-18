from __future__ import annotations

from collections.abc import Collection
from typing import Protocol, runtime_checkable

from symop.core.protocols.modes.labels import Path as PathProtocol


@runtime_checkable
class ApplyContext(Protocol):
    def allocate_path(
        self,
        *,
        hint: str | None = None,
        avoid: Collection[PathProtocol] | None = None,
    ) -> PathProtocol: ...

    def reserve_paths(self, paths: Collection[PathProtocol]) -> None: ...
