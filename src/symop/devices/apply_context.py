r"""Simple apply context for device execution.

Provides a minimal implementation of the apply context used during
device planning and execution.

The context manages allocation of fresh path labels and ensures that
newly created paths do not collide with existing or reserved ones.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, cast

from symop.core.protocols.modes.labels import Path as PathProtocol
from symop.modes.labels.path import Path


@dataclass
class SimpleApplyContext:
    r"""Minimal apply context with path allocation support.

    Parameters
    ----------
    prefix:
        Prefix used when generating new path names.

    Attributes
    ----------
    prefix:
        Base string used for newly allocated paths.
    _counter:
        Internal counter used to generate unique suffixes.
    _reserved:
        Set of paths that must not be reused.

    Notes
    -----
    This context is intended for simple device planning scenarios where
    only path allocation and collision avoidance are required.

    """

    prefix: str = "p"
    _counter: Iterator[int] = field(default_factory=lambda: count(1))
    _reserved: set[PathProtocol] = field(default_factory=set)

    def reserve_paths(self, paths: Collection[PathProtocol]) -> None:
        r"""Mark a collection of paths as reserved.

        Parameters
        ----------
        paths:
            Paths that should be excluded from future allocations.

        Returns
        -------
        None

        Notes
        -----
        Reserved paths will not be returned by :meth:`allocate_path`.

        """
        self._reserved |= set(paths)

    def allocate_path(
        self,
        *,
        hint: str | None = None,
        avoid: Collection[PathProtocol] | None = None,
    ) -> PathProtocol:
        r"""Allocate a fresh, non-conflicting path.

        Parameters
        ----------
        hint:
            Optional base name for the path. If not provided, ``prefix`` is used.
        avoid:
            Additional paths to avoid for this allocation.

        Returns
        -------
        PathProtocol
            Newly allocated path label.

        Notes
        -----
        - Generated paths have the form ``<base>_<n>``.
        - The allocation avoids both globally reserved paths and any paths
        provided via ``avoid``.
        - The returned path is automatically added to the reserved set.

        """
        avoid_set = set(self._reserved)
        if avoid is not None:
            avoid_set |= set(avoid)

        base = self.prefix if hint is None else hint

        while True:
            n = next(self._counter)
            path = Path(f"{base}_{n}")
            if path not in avoid_set:
                self._reserved.add(path)
                return cast(PathProtocol, path)


if TYPE_CHECKING:
    from symop.devices.protocols.apply_context import (
        ApplyContext as ApplyContextProtocol,
    )

    _check_simple_apply_context: ApplyContextProtocol = SimpleApplyContext()
