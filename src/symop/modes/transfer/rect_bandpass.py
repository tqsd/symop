r"""Rectangular band-pass transfer function.

This module defines an ideal rectangular band-pass amplitude transfer.

The transfer is

.. math::

    H(\omega)
    =
    \begin{cases}
        1, & |\omega-\omega_0|\le \Delta\omega/2 \\
        0, & \text{otherwise}
    \end{cases}.

This models an idealized hard cutoff in the frequency domain.

Notes
-----
This transfer is not compatible with the Gaussian-closed formalism and
therefore falls back to numerical filtering.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from symop.core.types.arrays import FloatArray, RCArray
from symop.modes.transfer.base import TransferBase
from symop.modes.types import as_float_array, require_pos_finite


@dataclass(frozen=True)
class RectBandpass(TransferBase):
    r"""Ideal rectangular band-pass amplitude transfer.

    Parameters
    ----------
    w0:
        Center angular frequency :math:`\omega_0`.
    width:
        Full passband width :math:`\Delta\omega`.

    """

    _signature_tag = "rect_bandpass"

    w0: float
    width: float

    def __post_init__(self) -> None:
        """Validate parameters.

        Raises
        ------
        ValueError
            If ``width`` is not positive and finite.

        """
        require_pos_finite("width", self.width)

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
        half = 0.5 * float(self.width)
        mask = np.abs(w - float(self.w0)) <= half
        return cast(RCArray, mask.astype(complex))


if TYPE_CHECKING:
    from symop.core.protocols.modes.transfer import TransferFunction

    _check: TransferFunction = RectBandpass(w0=1.0, width=10.0)
