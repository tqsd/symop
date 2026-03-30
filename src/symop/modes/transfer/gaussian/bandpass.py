r"""Gaussian band-pass transfer function.

This module defines a Gaussian band-pass amplitude transfer.

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
:math:`\sigma_\omega`.

Within the Gaussian-closed formalism, this transfer admits an analytic
representation and can be applied in closed form to Gaussian envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from symop.core.types.arrays import FloatArray, RCArray
from symop.modes.transfer.gaussian.base import GaussianClosedTransferBase
from symop.modes.transfer.gaussian.formalism import (
    GaussianAtom,
    GaussianTransferExpansion,
)
from symop.modes.types import as_float_array, require_pos_finite


@dataclass(frozen=True)
class GaussianBandpass(GaussianClosedTransferBase):
    r"""Gaussian band-pass amplitude transfer.

    .. math::

        H(\omega)
        =
        \exp\left[
            -\frac{1}{2}
            \left(
                \frac{\omega-\omega_0}{\sigma_\omega}
            \right)^2
        \right].

    Parameters
    ----------
    w0:
        Center angular frequency :math:`\omega_0`.
    sigma_w:
        Bandwidth parameter :math:`\sigma_\omega`.

    """

    _signature_tag = "gauss_bandpass"

    w0: float
    sigma_w: float

    def __post_init__(self) -> None:
        """Validate parameters.

        Raises
        ------
        ValueError
            If ``sigma_w`` is not positive and finite.

        """
        require_pos_finite("sigma_w", self.sigma_w)

    def __call__(self, w: FloatArray) -> RCArray:
        r"""Evaluate the transfer function on a frequency grid.

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
        x = (w - float(self.w0)) / s
        return np.exp(-0.5 * x * x).astype(complex)

    def _as_expansion(self) -> GaussianTransferExpansion:
        r"""Convert this transfer into a Gaussian expansion.

        Returns
        -------
        GaussianTransferExpansion
            Expansion consisting of a single Gaussian atom:

            .. math::

                H(\omega)
                =
                \exp\left[
                    -\frac{1}{2}
                    \left(
                        \frac{\omega-\omega_0}{\sigma_\omega}
                    \right)^2
                \right].

        Notes
        -----
        This representation enables closed-form propagation of Gaussian
        envelopes and Gaussian mixtures.

        """
        return GaussianTransferExpansion(
            c0=0.0 + 0.0j,
            atoms=(
                GaussianAtom(
                    coeff=1.0 + 0.0j,
                    w0=float(self.w0),
                    sigma_w=float(self.sigma_w),
                ),
            ),
        )


if TYPE_CHECKING:
    from symop.core.protocols.modes.transfer import TransferFunction

    _check_gaussian_bandpass: TransferFunction = GaussianBandpass(w0=1.0, sigma_w=2.0)
