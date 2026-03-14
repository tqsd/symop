r"""Cascade transfer function.

This module defines a cascade transfer function implementing
the :class:`~symop.modes.protocols.TransferFunctionProto` interface.

A cascade function acts as multiplication of its parts:

.. math::

    H = H_n \ldots H_2 H_1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from symop.core.protocols.modes.transfer import TransferFunction
from symop.core.types.arrays import RCArray
from symop.core.types.signature import Signature
from symop.modes.types import FloatArray, as_float_array


@dataclass(frozen=True)
class Cascade:
    r"""Pointwise product of transfer functions.

    :math:`H(\omega) = H_n(\omega)\,\cdots\,H_2(\omega)\,H_1(\omega)`.
    """

    parts: tuple[TransferFunction, ...]

    @property
    def signature(self) -> Signature:
        """Stable signature for caching and comparison.

        Returns
        -------
        Signature
            Tuple uniquely identifying this transfer function.

        """
        return ("cascade", tuple(p.signature for p in self.parts))

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
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
        return (
            "cascade_approx",
            tuple(
                p.approx_signature(
                    decimals=decimals,
                    ignore_global_phase=ignore_global_phase,
                )
                for p in self.parts
            ),
        )

    def __call__(self, w: FloatArray) -> RCArray:
        r"""Evaluate the transfer function on a frequency grid.

        Parameters
        ----------
        w:
            Angular frequency grid.

        Returns
        -------
        RCArray
            Complex-valued transfer samples :math:`H(\omega)`.

        """
        w = as_float_array(w)
        out = np.ones_like(w, dtype=complex)
        for p in self.parts:
            out *= p(w)
        return out


if TYPE_CHECKING:
    from symop.modes.transfer.gaussian_bandpass import GaussianBandpass

    g = GaussianBandpass(w0=10, sigma_w=10)
    _cascade: TransferFunction = Cascade(parts=(g,))
