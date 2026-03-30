r"""Super-Gaussian band-pass transfer function.

This module defines a super-Gaussian band-pass amplitude transfer.

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
:math:`m` produces flatter passbands and steeper edges.

Notes
-----
This transfer is not representable as a finite Gaussian expansion and
therefore does not belong to the Gaussian-closed formalism.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from symop.core.types.arrays import FloatArray, RCArray
from symop.modes.transfer.base import TransferBase
from symop.modes.types import as_float_array, require_pos_finite


@dataclass(frozen=True)
class SuperGaussianBandpass(TransferBase):
    r"""Super-Gaussian band-pass amplitude transfer.

    Parameters
    ----------
    w0:
        Center angular frequency :math:`\omega_0`.
    sigma_w:
        Width parameter :math:`\sigma_\omega`.
    order:
        Super-Gaussian order :math:`m \ge 1`.

    """

    _signature_tag = "supergauss_bandpass"

    w0: float
    sigma_w: float
    order: int = 2

    def __post_init__(self) -> None:
        """Validate parameters.

        Raises
        ------
        ValueError
            If ``sigma_w`` is not positive and finite or if ``order < 1``.

        """
        require_pos_finite("sigma_w", self.sigma_w)

        m = int(self.order)
        if m < 1:
            raise ValueError(f"order must be >= 1, got {self.order!r}")

    def __call__(self, w: FloatArray) -> RCArray:
        r"""Evaluate the transfer function.

        Parameters
        ----------
        w:
            Angular-frequency grid :math:`\omega`.

        Returns
        -------
        RCArray
            Complex samples of :math:`H(\omega)`.

        """
        w = as_float_array(w)
        s = float(self.sigma_w)
        m = int(self.order)

        x = (w - float(self.w0)) / s
        return np.exp(-0.5 * np.power(x * x, m)).astype(complex)


if TYPE_CHECKING:
    from symop.core.protocols.modes.transfer import TransferFunction

    _check: TransferFunction = SuperGaussianBandpass(
        w0=1.0,
        sigma_w=0.5,
        order=2,
    )
