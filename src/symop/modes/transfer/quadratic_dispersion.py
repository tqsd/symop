r"""Quadratic dispersion transfer function.

This module defines a pure quadratic spectral-phase transfer implementing the
:class:`~symop.modes.protocols.TransferFunctionProto` interface.

The transfer applies a frequency-dependent phase around a reference frequency
:math:`\omega_\mathrm{ref}`:

.. math::

    H(\omega)
    =
    \exp\left(
        -i\frac{\beta_2}{2}(\omega-\omega_\mathrm{ref})^2
    \right).

This changes the temporal shape of an envelope (dispersion) while leaving the
spectral intensity :math:`|Z(\omega)|^2` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from symop.core.protocols.signature import SignatureProto
from symop.modes.protocols import TransferFunctionProto
from symop.modes.types import FloatArray, RCArray, as_float_array


@dataclass(frozen=True)
class QuadraticDispersion(TransferFunctionProto):
    r"""Pure quadratic spectral phase around :math:`\omega_\mathrm{ref}`.

    .. math::

        H(\omega)=\exp\left(-i\frac{\beta_2}{2}(\omega-\omega_\mathrm{ref})^2\right).
    """

    beta2: float
    w_ref: float = 0.0

    @property
    def signature(self) -> SignatureProto:
        """Stable signature for caching and comparison.

        Returns
        -------
        SignatureProto
            Tuple uniquely identifying this transfer function.

        """
        return ("quad_dispersion", float(self.beta2), float(self.w_ref))

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> SignatureProto:
        """Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimal places used for rounding float parameters.
        ignore_global_phase:
            Ignored for this transfer (no global phase parameter).

        Returns
        -------
        SignatureProto
            Approximate signature tuple.

        """
        r = round
        return (
            "quad_dispersion_approx",
            r(float(self.beta2), decimals),
            r(float(self.w_ref), decimals),
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
            If ``beta2`` or ``w_ref`` is not finite.

        """
        w = as_float_array(w)
        beta2 = float(self.beta2)
        w_ref = float(self.w_ref)

        if not np.isfinite(beta2):
            raise ValueError(f"beta2 must be finite, got {self.beta2!r}")
        if not np.isfinite(w_ref):
            raise ValueError(f"w_ref must be finite, got {self.w_ref!r}")

        dw = w - w_ref
        return cast(RCArray, np.exp(-1j * 0.5 * beta2 * dw * dw).astype(complex))
