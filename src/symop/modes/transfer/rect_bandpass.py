r"""Rectangular band-pass transfer function.

This module defines an ideal rectangular band-pass amplitude transfer
implementing the :class:`~symop.modes.protocols.TransferFunctionProto`
interface.

The transfer is

.. math::

    H(\omega)
    =
    \begin{cases}
        1, & |\omega-\omega_0|\le \Delta\omega/2 \\
        0, & \text{otherwise}
    \end{cases}.

It models a hard cutoff in the frequency domain (an idealized filter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from symop.core.types.arrays import FloatArray, RCArray
from symop.core.types.signature import Signature
from symop.modes.types import (
    as_float_array,
    require_pos_finite,
)


@dataclass(frozen=True)
class RectBandpass:
    r"""Ideal rectangular band-pass amplitude transfer.

    .. math::

        H(\omega)
        =
        \begin{cases}
            1, & |\omega-\omega_0|\le \Delta\omega/2 \\
            0, & \text{otherwise}
        \end{cases}.
    """

    w0: float
    width: float  # full width

    @property
    def signature(self) -> Signature:
        """Stable signature for caching and comparison.

        Returns
        -------
        Signature
            Tuple uniquely identifying this transfer function.

        """
        return ("rect_bandpass", float(self.w0), float(self.width))

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> Signature:
        """Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimal places used for rounding float parameters.
        ignore_global_phase:
            Ignored for this transfer (no phase parameter).

        Returns
        -------
        Signature
            Approximate signature tuple.

        """
        r = round
        return (
            "rect_bandpass_approx",
            r(float(self.w0), decimals),
            r(float(self.width), decimals),
        )

    def __call__(self, w: FloatArray) -> RCArray:
        r"""Evaluate the transfer function :math:`H(\omega)` on a frequency grid.

        Parameters
        ----------
        w:
            Angular-frequency grid :math:`\omega`.

        Returns
        -------
        RCArray
            Complex samples of :math:`H(\omega)`.

        Raises
        ------
        ValueError
            If ``width`` is not positive and finite.

        """
        w = as_float_array(w)
        width = require_pos_finite("width", self.width)
        half = 0.5 * width
        mask = np.abs(w - float(self.w0)) <= half
        return cast(RCArray, mask.astype(complex))


if TYPE_CHECKING:
    from symop.core.protocols.modes import TransferFunction

    _check: TransferFunction = RectBandpass(w0=1, width=10)
