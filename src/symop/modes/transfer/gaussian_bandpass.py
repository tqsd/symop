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
from typing import TYPE_CHECKING

import numpy as np

from symop.core.types.arrays import FloatArray, RCArray
from symop.core.types.signature import Signature
from symop.modes.protocols.envelope import SupportsGaussianClosedOverlap
from symop.modes.transfer.gaussian_formalism import (
    GaussianAtom,
    GaussianTransferExpansion,
)
from symop.modes.types import (
    as_float_array,
    require_pos_finite,
)


@dataclass(frozen=True)
class GaussianBandpass:
    r"""Gaussian band-pass amplitude transfer.

    .. math::

        H(\omega) = \exp\left[-\frac{1}{2}\left(\frac{\omega-\omega_0}{\sigma_\omega}\right)^2\right].

    Same shape as GaussianLowpass, but semantically "band-pass" around w0.
    """

    w0: float
    sigma_w: float

    @property
    def signature(self) -> Signature:
        """Stable signature for caching and comparison.

        Returns
        -------
        Signature
            Tuple uniquely identifying this transfer function.

        """
        return ("gauss_bandpass", float(self.w0), float(self.sigma_w))

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> Signature:
        r"""Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimals to round to.
        ignore_global_phase:
            If True, treat :math:`\phi_0` as zero for grouping.

        Returns
        -------
        Signature
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

    def _as_expansion(self) -> GaussianTransferExpansion:
        r"""Convert this transfer into a Gaussian transfer expansion.

        This representation is used by the ``gaussian_closed`` formalism to
        apply the transfer analytically to Gaussian envelopes.

        Returns
        -------
        GaussianTransferExpansion
            Expansion consisting of a single Gaussian atom representing the
            band-pass filter:

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
        This allows closed-form propagation of Gaussian envelopes by
        representing the filter as a sum of Gaussian atoms. In this case
        the expansion contains a single atom and no constant term.

        """
        return GaussianTransferExpansion(
            c0=0.0 + 0.0j,
            atoms=(
                GaussianAtom(
                    coeff=1.0 + 0.0j, w0=float(self.w0), sigma_w=(self.sigma_w)
                ),
            ),
        )

    def apply_to_gaussian(
        self, env: SupportsGaussianClosedOverlap
    ) -> tuple[SupportsGaussianClosedOverlap, float]:
        r"""Apply this transfer to a Gaussian-closed envelope.

        This method delegates to the Gaussian transfer expansion produced
        by :meth:`_as_expansion`, enabling closed-form filtering of Gaussian
        envelopes and Gaussian mixtures.

        Parameters
        ----------
        env:
            Input envelope belonging to the ``gaussian_closed`` family.

        Returns
        -------
        (env_out, eta):
            - env_out: resulting Gaussian-closed envelope (typically a
            Gaussian mixture).
            - eta: transmissivity

            .. math::

                \eta =
                \frac{1}{2\pi}
                \int |H(\omega)|^2 |Z(\omega)|^2 d\omega,

            representing the power transmission of the filter.

        Notes
        -----
        The returned envelope remains normalized as a **mode descriptor**.
        The transmissivity :math:`\eta` should be applied separately to the
        quantum state to represent loss.

        """
        return self._as_expansion().apply_to_gaussian(env)


if TYPE_CHECKING:
    from symop.core.protocols.modes import TransferFunction

    _check_gaussian_bandpass: TransferFunction = GaussianBandpass(w0=1, sigma_w=2)
