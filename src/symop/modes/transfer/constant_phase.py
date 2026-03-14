r"""Constant phase transfer function.

This module defines a global phase transfer implementing the
:class:`~symop.modes.protocols.TransferFunctionProto` interface.

The transfer multiplies all spectral components by a frequency-independent
complex phase factor:

.. math::

    H(\omega) = e^{i\phi_0}.

Since the phase is independent of :math:`\omega`, this transformation
does not change the spectral intensity :math:`|Z(\omega)|^2`, but only
rotates the complex field globally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from symop.core.types.arrays import RCArray
from symop.core.types.signature import Signature
from symop.modes.types import FloatArray, as_float_array


@dataclass(frozen=True)
class ConstantPhase:
    r"""Global phase factor.

    .. math::

        H(\omega)=e^{i\phi_0}.
    """

    phi0: float

    @property
    def signature(self) -> Signature:
        """Stable signature for caching and comparison.

        Returns
        -------
        Signature
            Tuple uniquely identifying this transfer function.

        """
        return ("const_phase", float(self.phi0))

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Signature:
        """Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimal places used for rounding float parameters.
        ignore_global_phase:
            If True, component signatures may ignore global phase where applicable.

        Returns
        -------
        Signature
            Approximate signature tuple.

        """
        if ignore_global_phase:
            return ("const_phase_approx", 0.0)
        r = round
        return ("const_phase_approx", r(float(self.phi0), decimals))

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
            If :math:`\phi_0` is not positive and finite.

        """
        w = as_float_array(w)
        phi0 = float(self.phi0)
        if not np.isfinite(phi0):
            raise ValueError(f"phi0 must be finite, got {self.phi0!r}")
        return cast(
            RCArray,
            (np.exp(1j * phi0) * np.ones_like(w, dtype=complex)).astype(complex),
        )


if TYPE_CHECKING:
    from symop.core.protocols.modes import TransferFunction

    _check: TransferFunction = ConstantPhase(phi0=1)
