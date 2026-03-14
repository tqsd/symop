r"""Time-delay transfer function.

This module defines a frequency-domain time-delay transfer implementing
the :class:`~symop.modes.protocols.TransferFunctionProto` interface.

A transfer function acts multiplicatively in the frequency domain:

.. math::

    Z_{\mathrm{out}}(\omega)
    =
    H(\omega)\,Z_{\mathrm{in}}(\omega).

For a time delay :math:`\tau`, the transfer is

.. math::

    H(\omega) = e^{-i\omega\tau},

which corresponds to a shift in the time domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from symop.core.types.arrays import FloatArray, RCArray
from symop.core.types.signature import Signature
from symop.modes.types import as_float_array


@dataclass(frozen=True)
class TimeDelay:
    r"""Time delay in the frequency domain.

    .. math::

        H(\omega) = e^{-i\omega\tau}.
    """

    tau: float

    @property
    def signature(self) -> Signature:
        """Stable signature for caching and comparison.

        Returns
        -------
        Signature
            Tuple uniquely identifying this transfer function.

        """
        return ("time_delay", float(self.tau))

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> Signature:
        """Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimal places used for rounding float parameters.
        ignore_global_phase:
            Ignored for this transfer (no global phase parameter).

        Returns
        -------
        Signature
            Approximate signature tuple.

        """
        r = round
        return ("time_delay_approx", r(float(self.tau), decimals))

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
            If ``tau`` is not finite.

        """
        w = as_float_array(w)
        tau = float(self.tau)

        if not np.isfinite(tau):
            raise ValueError(f"tau must be finite, got {self.tau!r}")

        return cast(RCArray, np.exp(-1j * w * tau).astype(complex))


if TYPE_CHECKING:
    from symop.core.protocols.modes import TransferFunction

    _check: TransferFunction = TimeDelay(tau=1)
