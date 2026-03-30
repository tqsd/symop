r"""Quadratic dispersion transfer function.

This module defines a pure quadratic spectral-phase transfer.

The transfer applies a frequency-dependent phase around a reference
frequency :math:`\omega_\mathrm{ref}`:

.. math::

    H(\omega)
    =
    \exp\left(
        -i\frac{\beta_2}{2}(\omega-\omega_\mathrm{ref})^2
    \right).

This changes the temporal shape of an envelope through quadratic
dispersion while leaving the spectral intensity :math:`|Z(\omega)|^2`
unchanged.

Notes
-----
This transfer is kept in the generic transfer layer rather than the
Gaussian-closed transfer layer. A quadratic spectral phase generally
produces chirped Gaussian envelopes, so closed-form support would
require an extended Gaussian family that explicitly represents chirp.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from symop.core.types.arrays import FloatArray, RCArray
from symop.modes.transfer.base import TransferBase
from symop.modes.types import as_float_array


@dataclass(frozen=True)
class QuadraticDispersion(TransferBase):
    r"""Pure quadratic spectral phase around :math:`\omega_\mathrm{ref}`.

    The transfer is

    .. math::

        H(\omega)
        =
        \exp\left(
            -i\frac{\beta_2}{2}(\omega-\omega_\mathrm{ref})^2
        \right).

    Parameters
    ----------
    beta2:
        Quadratic dispersion coefficient :math:`\beta_2`.
    w_ref:
        Reference angular frequency :math:`\omega_\mathrm{ref}`.

    """

    _signature_tag = "quad_dispersion"

    beta2: float
    w_ref: float = 0.0

    def __post_init__(self) -> None:
        """Validate stored parameters.

        Raises
        ------
        ValueError
            If ``beta2`` or ``w_ref`` is not finite.

        """
        beta2 = float(self.beta2)
        w_ref = float(self.w_ref)

        if not np.isfinite(beta2):
            raise ValueError(f"beta2 must be finite, got {self.beta2!r}")
        if not np.isfinite(w_ref):
            raise ValueError(f"w_ref must be finite, got {self.w_ref!r}")

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
        beta2 = float(self.beta2)
        w_ref = float(self.w_ref)

        dw = w - w_ref
        return cast(
            RCArray,
            np.exp(-1j * 0.5 * beta2 * dw * dw).astype(complex),
        )


if TYPE_CHECKING:
    from symop.core.protocols.modes.transfer import TransferFunction

    _check: TransferFunction = QuadraticDispersion(beta2=1.0, w_ref=3.0)
