r"""Gaussian band-pass transfer function.

This module defines a Gaussian band-pass amplitude transfer implementing
the :class:`~symop.modes.protocols.TransferFunctionProto` interface.

The transfer is given by

.. math::

    H(\omega)
    =
    \exp\left[
        -\frac{1}{2}
        \left(
            \frac{\omega - \omega_0}{\sigma_\omega}
        \right)^2
    \right].

It selects frequencies around :math:`\omega_0` with bandwidth
:math:`\sigma_\omega`. Although mathematically identical in form to a
Gaussian low-pass centered at :math:`\omega_0`, it is interpreted
semantically as a band-pass filter.
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
class GaussianBandpass(TransferFunctionProto):
    r"""Gaussian band-pass amplitude transfer.

    .. math::

        H(\omega) = \exp\left[-\frac{1}{2}\left(\frac{\omega-\omega_0}{\sigma_\omega}\right)^2\right].

    Same shape as GaussianLowpass, but semantically "band-pass" around w0.
    """

    w0: float
    sigma_w: float

    @property
    def signature(self) -> SignatureProto:
        """Stable signature for caching and comparison.

        Returns
        -------
        SignatureProto
            Tuple uniquely identifying this transfer function.

        """
        return ("gauss_bandpass", float(self.w0), float(self.sigma_w))

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> SignatureProto:
        r"""Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimals to round to.
        ignore_global_phase:
            If True, treat :math:`\phi_0` as zero for grouping.

        Returns
        -------
        SignatureProto
            Rounded/approximate signature tuple.

        """
        r = round
        return (
            "gauss_bandpass_approx",
            r(float(self.w0), decimals),
            r(float(self.sigma_w), decimals),
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
            If :math:`\sigma_\omega` is not positive and finite.

        """
        w = as_float_array(w)
        s = require_pos_finite("sigma_w", self.sigma_w)
        x = (w - float(self.w0)) / s
        return np.exp(-0.5 * x * x).astype(complex)
