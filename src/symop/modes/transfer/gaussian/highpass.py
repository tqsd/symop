r"""Gaussian high-pass transfer function.

This module defines a soft Gaussian high-pass amplitude transfer.

The transfer is constructed as the complement of a Gaussian low-pass,

.. math::

    H(\omega)
    =
    1
    -
    \exp\left[
        -\frac{1}{2}
        \left(
            \frac{\omega - \omega_0}{\sigma_\omega}
        \right)^2
    \right],

which produces a smooth high-pass characteristic rather than a sharp
physical cutoff.

Within the Gaussian-closed formalism, this transfer admits an analytic
representation as a constant term plus a single Gaussian atom, so it can
be applied in closed form to Gaussian envelopes and Gaussian mixtures.
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
class GaussianHighpass(GaussianClosedTransferBase):
    r"""Soft Gaussian high-pass amplitude transfer.

    The transfer is defined by

    .. math::

        H(\omega)
        =
        1
        -
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
        Width parameter :math:`\sigma_\omega` of the complementary
        Gaussian low-pass.

    """

    _signature_tag = "gauss_highpass"

    w0: float
    sigma_w: float

    def __post_init__(self) -> None:
        """Validate stored parameters.

        Raises
        ------
        ValueError
            If ``sigma_w`` is not positive and finite.

        """
        require_pos_finite("sigma_w", self.sigma_w)

    def __call__(self, w: FloatArray) -> RCArray:
        r"""Evaluate the transfer function on an angular-frequency grid.

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
        h_lp = np.exp(-0.5 * x * x)
        return (1.0 - h_lp).astype(complex)

    def _as_expansion(self) -> GaussianTransferExpansion:
        r"""Convert this transfer into a Gaussian expansion.

        Returns
        -------
        GaussianTransferExpansion
            Expansion of the form

            .. math::

                H(\omega)
                =
                c_0
                +
                c_1
                \exp\left[
                    -\frac{1}{2}
                    \left(
                        \frac{\omega-\omega_0}{\sigma_\omega}
                    \right)^2
                \right],

            with

            .. math::

                c_0 = 1,
                \qquad
                c_1 = -1.

        Notes
        -----
        This representation allows the transfer to be applied in closed
        form to Gaussian-closed envelopes.

        """
        return GaussianTransferExpansion(
            c0=1.0 + 0.0j,
            atoms=(
                GaussianAtom(
                    coeff=-1.0 + 0.0j,
                    w0=float(self.w0),
                    sigma_w=float(self.sigma_w),
                ),
            ),
        )


if TYPE_CHECKING:
    from symop.core.protocols.modes.transfer import TransferFunction

    _check_gaussian_highpass: TransferFunction = GaussianHighpass(
        w0=1.0,
        sigma_w=2.0,
    )
