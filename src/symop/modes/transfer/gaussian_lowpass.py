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
from typing import TYPE_CHECKING

import numpy as np

from symop.core.types.arrays import FloatArray, RCArray
from symop.core.types.signature import Signature
from symop.modes.protocols.envelope import SupportsGaussianClosedOverlap
from symop.modes.transfer.gaussian_formalism import (
    GaussianAtom,
    GaussianTransferExpansion,
)


@dataclass(frozen=True)
class GaussianLowpass:
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
    def signature(self) -> Signature:
        """Stable signature for caching/comparison.

        Returns
        -------
        Signature
            Tuple identifying the transfer function and its parameters.

        """
        return ("gauss_lowpass", float(self.w0), float(self.sigma_w))

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
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
            "gauss_lowpass_approx",
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
            raise ValueError(f"sigma_w must be positive finite, , got {self.sigma_w!r}")

        x = (w - float(self.w0)) / s
        return np.exp(-0.5 * x * x).astype(complex)

    def _as_expansion(self) -> GaussianTransferExpansion:
        r"""Return a Gaussian transfer expansion representation.

        This expresses the transfer function as a sum of Gaussian atoms
        compatible with the closed Gaussian formalism used by
        :class:`GaussianTransferExpansion`.

        The current implementation contains a single Gaussian atom with
        coefficient ``1`` centered at :math:`\omega_0` with width
        :math:`\sigma_\omega`, which exactly reproduces the transfer
        function

        .. math::

            H(\omega) =
            \exp\left[
                -\frac{1}{2}
                \left(
                    \frac{\omega-\omega_0}{\sigma_\omega}
                \right)^2
            \right].

        Returns
        -------
        GaussianTransferExpansion
            Expansion object that can analytically act on Gaussian
            envelopes in the closed-form formalism.

        Notes
        -----
        This representation allows analytic application of the transfer
        function to Gaussian envelopes without numerical sampling.

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
        r"""Apply the transfer function to a Gaussian envelope.

        The input envelope is multiplied by the spectral transfer
        function

        .. math::

            \zeta_{out}(\omega) = H(\omega) \zeta_{in}(\omega),

        and the result is projected back into the closed Gaussian
        envelope family.

        Parameters
        ----------
        env:
            Input Gaussian envelope supporting the closed-form overlap
            formalism.

        Returns
        -------
        tuple[SupportsGaussianClosedOverlap, float]
            A pair ``(env_out, eta)`` where

            - ``env_out`` is the normalized output envelope
            - ``eta`` is the transmission efficiency

            .. math::

                \eta = \langle \zeta_{out} | \zeta_{out} \rangle.

        Notes
        -----
        The implementation converts the transfer function to a
        :class:`GaussianTransferExpansion` and delegates the analytic
        envelope transformation to that representation.

        """
        return self._as_expansion().apply_to_gaussian(env)


if TYPE_CHECKING:
    from symop.core.protocols.modes import TransferFunction

    _check_gaussian_bandpass: TransferFunction = GaussianLowpass(w0=1, sigma_w=2)
