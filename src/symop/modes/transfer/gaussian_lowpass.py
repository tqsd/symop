r"""Gaussian low-pass transfer function.

This module defines a Gaussian low-pass amplitude transfer implementing
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

It transmits frequencies near :math:`\omega_0` and suppresses
components away from the center with bandwidth
:math:`\sigma_\omega`.

This implementation models a purely real amplitude response
(with no dispersive phase contribution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symop.core.protocols.signature import SignatureProto
from symop.modes.protocols import TransferFunctionProto
from symop.modes.types import FloatArray, RCArray


@dataclass(frozen=True)
class GaussianLowpass(TransferFunctionProto):
    r"""Gaussian spectral amplitude transfer function.

    This implements a Gaussian low-pass response centered at :math:`\omega_0`
    with width parameter :math:`\sigma_\omega`:

    .. math::

        H(\omega) = \exp\left[-\frac{1}{2}\left(\frac{\omega-\omega_0}{\sigma_\omega}\right)^2\right].

    Notes
    -----
    - This implementation is real-valued (no dispersion), and returns complex
      dtype for convenience and composability with other transfer functions.
    - The width parameter :math:`\sigma_\omega` must be positive.

    """

    w0: float
    sigma_w: float

    @property
    def signature(self) -> SignatureProto:
        """Stable signature for caching/comparison.

        Returns
        -------
        SignatureProto
            Tuple identifying the transfer function and its parameters.

        """
        return ("gauss_lowpass", float(self.w0), float(self.sigma_w))

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
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
            "gauss_approx",
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
        w = np.asarray(w, dtype=float)
        s = float(self.sigma_w)
        if not (s > 0.0) or not np.isfinite(s):
            raise ValueError(f"sigma_w must be positive finite, got {self.sigma_w!r}")

        x = (w - float(self.w0)) / s
        return np.exp(-0.5 * x * x).astype(complex)
