r"""Super-Gaussian band-pass transfer function.

This module defines a super-Gaussian band-pass amplitude transfer
implementing the :class:`~symop.modes.protocols.TransferFunctionProto`
interface.

The transfer is

.. math::

    H(\omega)
    =
    \exp\left[
        -\frac{1}{2}
        \left(
            \frac{\omega-\omega_0}{\sigma_\omega}
        \right)^{2m}
    \right],
    \quad m \ge 1.

For :math:`m=1` this reduces to a Gaussian band-pass. Increasing
:math:`m` produces flatter passband behavior with steeper edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symop.core.protocols.signature import SignatureProto
from symop.modes.protocols import TransferFunctionProto
from symop.modes.types import (
    FloatArray,
    RCArray,
    as_float_array,
    require_pos_finite,
)


@dataclass(frozen=True)
class SuperGaussianBandpass(TransferFunctionProto):
    r"""Super-Gaussian band-pass amplitude transfer.

    .. math::

        H(\omega)
        =
        \exp\left[
            -\frac{1}{2}
            \left(
                \frac{\omega-\omega_0}{\sigma_\omega}
            \right)^{2m}
        \right],
        \quad m \ge 1.
    """

    w0: float
    sigma_w: float
    order: int = 2

    @property
    def signature(self) -> SignatureProto:
        """Stable signature for caching and comparison.

        Returns
        -------
        SignatureProto
            Tuple uniquely identifying this transfer function.

        """
        return (
            "supergauss_bandpass",
            float(self.w0),
            float(self.sigma_w),
            int(self.order),
        )

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> SignatureProto:
        """Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimal places used for rounding float parameters.
        ignore_global_phase:
            Ignored for this transfer (no phase parameter).

        Returns
        -------
        SignatureProto
            Approximate signature tuple.

        """
        r = round
        return (
            "supergauss_bandpass_approx",
            r(float(self.w0), decimals),
            r(float(self.sigma_w), decimals),
            int(self.order),
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
            If ``sigma_w`` is not positive and finite,
            or if ``order`` is less than 1.

        """
        w = as_float_array(w)
        s = require_pos_finite("sigma_w", self.sigma_w)

        m = int(self.order)
        if m < 1:
            raise ValueError(f"order must be >= 1, got {self.order!r}")

        x = (w - float(self.w0)) / s
        return np.exp(-0.5 * np.power(x * x, m)).astype(complex)
