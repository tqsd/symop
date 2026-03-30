r"""Time-delay transfer function.

This module defines a frequency-domain time-delay transfer.

A transfer function acts multiplicatively in the frequency domain:

.. math::

    Z_{\mathrm{out}}(\omega)
    =
    H(\omega)\,Z_{\mathrm{in}}(\omega).

For a time delay :math:`\tau`, the transfer is

.. math::

    H(\omega) = e^{-i\omega\tau},

which corresponds to a shift in the time domain.

Notes
-----
This transfer is Gaussian-compatible, but it is not represented by a
finite Gaussian expansion. Instead, it acts by directly shifting the
time parameter of Gaussian-closed envelopes.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from symop.core.types.arrays import FloatArray, RCArray
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope
from symop.modes.protocols.envelope import GaussianClosedEnvelope
from symop.modes.transfer.gaussian.base import GaussianClosedTransferBase
from symop.modes.transfer.gaussian.formalism import GaussianTransferExpansion
from symop.modes.types import as_float_array


@dataclass(frozen=True)
class TimeDelay(GaussianClosedTransferBase):
    r"""Time delay in the frequency domain.

    .. math::

        H(\omega) = e^{-i\omega\tau}.
    """

    _signature_tag = "time_delay"

    tau: float

    def __post_init__(self) -> None:
        """Validate ``tau`` parameter.

        Raises
        ------
        ValueError
            If ``tau`` is not finite.

        """
        if not np.isfinite(self.tau):
            raise ValueError(f"tau must be finite, got {self.tau!r}")

    def __call__(self, w: FloatArray) -> RCArray:
        r"""Evaluate the transfer function :math:`H(\omega)` on a frequency grid.

        Parameters
        ----------
        w
            Angular-frequency grid :math:`\omega`.

        Returns
        -------
        RCArray
            Complex samples of :math:`H(\omega)`.

        """
        w = as_float_array(w)
        tau = float(self.tau)
        return cast(RCArray, np.exp(-1j * w * tau).astype(complex))

    def _as_expansion(self) -> GaussianTransferExpansion:
        """Raise because time delay is not a finite Gaussian expansion.

        Returns
        -------
        GaussianTransferExpansion
            Never returned.

        Raises
        ------
        NotImplementedError
            Always raised because time delay is not representable as a
            finite Gaussian expansion in the current formalism.

        """
        raise NotImplementedError(
            "TimeDelay is not representable as a finite Gaussian expansion."
        )

    def apply_to_gaussian(
        self,
        env: GaussianClosedEnvelope,
    ) -> tuple[GaussianClosedEnvelope, float]:
        r"""Apply the time delay analytically to a Gaussian-closed envelope.

        Parameters
        ----------
        env
            Input envelope in the Gaussian-closed family.

        Returns
        -------
        tuple[GaussianClosedEnvelope, float]
            Pair ``(env_out, eta)`` where ``env_out`` is the delayed
            envelope and ``eta = 1.0``.

        Notes
        -----
        A time delay shifts the Gaussian time parameter without changing
        the envelope norm.

        """
        r"""Apply the time delay analytically to a Gaussian-closed envelope."""
        dt = float(self.tau)

        if isinstance(env, GaussianEnvelope):
            return (
                GaussianEnvelope(
                    omega0=float(env.omega0),
                    sigma=float(env.sigma),
                    tau=float(env.tau + dt),
                    phi0=float(env.phi0),
                ),
                1.0,
            )

        if isinstance(env, GaussianMixtureEnvelope):
            components = tuple(
                GaussianEnvelope(
                    omega0=float(g.omega0),
                    sigma=float(g.sigma),
                    tau=float(g.tau + dt),
                    phi0=float(g.phi0),
                )
                for g in env.components
            )
            return (
                GaussianMixtureEnvelope(
                    components=components,
                    weights=env.weights,
                ),
                1.0,
            )

        raise TypeError(
            "TimeDelay supports GaussianEnvelope or GaussianMixtureEnvelope."
        )


if TYPE_CHECKING:
    from symop.core.protocols.modes.transfer import TransferFunction

    _check: TransferFunction = TimeDelay(tau=1.0)
