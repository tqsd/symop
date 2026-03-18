r"""Transfer application utilities for mode envelopes.

This module provides helper routines for applying spectral transfer
functions to mode envelopes.

The main entry point is :func:`apply_transfer`, which applies a
:class:`TransferFunction` to a :class:`BaseEnvelope` and returns the
resulting envelope together with the transmission efficiency.

Two execution paths are supported:

1. **Closed Gaussian formalism**

   If both the envelope and the transfer function support the
   Gaussian closed-form formalism, the transformation is performed
   analytically. This avoids numerical sampling and preserves the
   analytic Gaussian representation.

2. **Numerical spectral filtering**

   For general envelopes or transfer functions, the operation is
   implemented via :class:`FilteredEnvelope`, which evaluates the
   spectral product

   .. math::

       \zeta_{\mathrm{out}}(\omega)
       =
       H(\omega)\,\zeta_{\mathrm{in}}(\omega)

   using FFT-based reconstruction.

In both cases the returned efficiency

.. math::

    \eta = \langle \zeta_{\mathrm{out}} \mid \zeta_{\mathrm{out}} \rangle

represents the transmitted power of the mode after filtering.
"""

from __future__ import annotations

from symop.core.protocols.modes.transfer import TransferFunction
from symop.modes.envelopes.filtered import FilteredEnvelope
from symop.modes.protocols.envelope import (
    GaussianClosedEnvelope,
    TimeFrequencyEnvelope,
)
from symop.modes.protocols.transfer import (
    SupportsGaussianClosedTransfer,
)


def apply_transfer(
    transfer: TransferFunction,
    env: TimeFrequencyEnvelope,
    *,
    n_fft: int = 2**15,
    w_span_sigma: float = 12.0,
) -> tuple[TimeFrequencyEnvelope, float]:
    r"""Apply a spectral transfer function to a mode envelope.

    This function multiplies the envelope spectrum with a transfer
    function

    .. math::

        \zeta_{\mathrm{out}}(\omega)
        =
        H(\omega)\,\zeta_{\mathrm{in}}(\omega),

    and returns the resulting envelope together with the transmitted
    power.

    Two evaluation strategies are used depending on the available
    representations:

    * **Closed Gaussian path**

      If the envelope is a :class:`GaussianClosedEnvelope` and the
      transfer function implements
      :class:`SupportsGaussianClosedTransfer`, the operation is
      performed analytically.

    * **Numerical filtering path**

      Otherwise the operation is performed using
      :class:`FilteredEnvelope`, which evaluates the spectral product
      on a discrete frequency grid.

    Parameters
    ----------
    transfer
        Spectral transfer function :math:`H(\omega)`.
    env
        Input mode envelope.
    n_fft
        Number of FFT points used for numerical filtering.
        Ignored if the analytic Gaussian path is used.
    w_span_sigma
        Frequency span of the numerical grid expressed in multiples
        of the envelope bandwidth.

    Returns
    -------
    tuple[TimeFrequencyEnvelope, float]
        Pair ``(env_out, eta)`` where

        - ``env_out`` is the filtered envelope
        - ``eta`` is the transmission efficiency

        .. math::

            \eta
            =
            \langle \zeta_{\mathrm{out}} \mid \zeta_{\mathrm{out}} \rangle.

    Notes
    -----
    The returned envelope always implements the
    :class:`TimeFrequencyEnvelope` interface so it can be evaluated in
    both time and frequency domains.

    """
    if isinstance(env, GaussianClosedEnvelope) and isinstance(
        transfer, SupportsGaussianClosedTransfer
    ):
        out_env, eta = transfer.apply_to_gaussian(env)
        return out_env, float(eta)

    out = FilteredEnvelope(
        base=env,
        transfer=transfer,
        n_fft=int(n_fft),
        w_span_sigma=float(w_span_sigma),
    )
    return out, out.eta
