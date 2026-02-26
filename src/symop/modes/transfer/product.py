r"""Product transfer function.

This module defines a transfer representing the pointwise product of two
transfer functions, implementing the
:class:`~symop.modes.protocols.TransferFunctionProto` interface.

Given two transfers :math:`H_a(\omega)` and :math:`H_b(\omega)`,
the product transfer is

.. math::

    H(\omega) = H_a(\omega)\,H_b(\omega).

This is equivalent to cascading two linear spectral responses
that act multiplicatively in the frequency domain.
"""

from __future__ import annotations

from dataclasses import dataclass

from symop.core.protocols.signature import SignatureProto
from symop.modes.protocols import TransferFunctionProto
from symop.modes.types import FloatArray, RCArray, as_float_array


@dataclass(frozen=True)
class Product(TransferFunctionProto):
    r"""Pointwise product of two transfer functions.

    :math:`H(\omega) = H_a(\omega)\,H_b(\omega)`.
    """

    a: TransferFunctionProto
    b: TransferFunctionProto

    @property
    def signature(self) -> SignatureProto:
        """Stable signature for caching and comparison.

        Returns
        -------
        SignatureProto
            Tuple uniquely identifying this transfer function.

        """
        return ("product_tf", self.a.signature, self.b.signature)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> SignatureProto:
        """Approximate signature with rounded floating parameters.

        Parameters
        ----------
        decimals:
            Number of decimal places used for rounding float parameters.
        ignore_global_phase:
            If True, component signatures may ignore global phase
            where applicable.

        Returns
        -------
        SignatureProto
            Approximate signature tuple.

        """
        return (
            "product_tf_approx",
            self.a.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
            self.b.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
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
        return (self.a(w) * self.b(w)).astype(complex)
