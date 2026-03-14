"""Path labels for spatial or logical mode identification.

This module defines :class:`PathLabel`, a simple label used to distinguish
spatial (or logical) mode paths. Path labels are orthonormal: two labels
overlap if and only if their names are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from symop.core.types.signature import Signature


@dataclass(frozen=True)
class Path:
    r"""Path label identifying a spatial (or logical) mode path.

    Two path labels overlap if and only if their names are identical:

    .. math::

        \langle \mathrm{path}(a), \mathrm{path}(b) \rangle =
        \begin{cases}
            1, & a=b \\
            0, & a\neq b
        \end{cases}.
    """

    name: str

    def overlap(self, other: Path) -> complex:
        r"""Compute the overlap with another path label.

        Parameters
        ----------
        other:
            Another path label.

        Returns
        -------
        complex
            :math:`1` if names match, otherwise :math:`0`.

        """
        return 1.0 + 0.0j if self.name == other.name else 0.0 + 0.0j

    @property
    def signature(self) -> Signature:
        r"""Stable signature for caching/comparison.

        Returns
        -------
        Signature
            Tuple identifying the path label.

        """
        return ("path", self.name)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> Signature:
        """Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimal places used for rounding float parameters.
        ignore_global_phase:
            If True, component signatures may ignore global phase
            where applicable.

        Returns
        -------
        Signature
            Approximate signature tuple.

        """
        return self.signature


if TYPE_CHECKING:
    from symop.core.protocols.modes.labels import Path as PathProtocol

    _path_check: PathProtocol = Path("A")
