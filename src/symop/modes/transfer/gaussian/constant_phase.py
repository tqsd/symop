r"""Constant phase transfer function.

This module defines a frequency-independent phase transfer.

The transfer multiplies all spectral components by a constant complex
phase factor,

.. math::

    H(\omega) = e^{i\phi_0},

so it leaves the spectral intensity unchanged and only rotates the
complex field globally.

Within the Gaussian-closed envelope formalism, this transfer preserves
the Gaussian-closed family exactly. A single Gaussian remains a single
Gaussian with shifted phase, and a Gaussian mixture remains a Gaussian
mixture with all weights rotated by the same global phase factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from symop.core.types.arrays import RCArray
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope
from symop.modes.protocols.envelope import GaussianClosedEnvelope
from symop.modes.transfer.gaussian.base import GaussianClosedTransferBase
from symop.modes.transfer.gaussian.formalism import GaussianTransferExpansion
from symop.modes.types import FloatArray, as_float_array


@dataclass(frozen=True)
class ConstantPhase(GaussianClosedTransferBase):
    r"""Frequency-independent global phase transfer.

    The transfer is

    .. math::

        H(\omega) = e^{i\phi_0}.

    Since the phase factor is independent of :math:`\omega`, this
    transfer does not modify the spectral intensity profile and therefore
    does not introduce attenuation. Its transmissivity is always

    .. math::

        \eta = 1.

    Parameters
    ----------
    phi0:
        Global phase shift :math:`\phi_0` in radians.

    """

    _signature_tag = "const_phase"

    phi0: float

    def __post_init__(self) -> None:
        """Validate stored parameters.

        Raises
        ------
        ValueError
            If ``phi0`` is not finite.

        """
        phi0 = float(self.phi0)
        if not np.isfinite(phi0):
            raise ValueError(f"phi0 must be finite, got {self.phi0!r}")

    def _signature_params(
        self,
        *,
        ignore_global_phase: bool = False,
    ) -> tuple[float]:
        """Return the parameter tuple used in signatures.

        Parameters
        ----------
        ignore_global_phase:
            If True, normalize the global phase to zero for approximate
            grouping and comparison.

        Returns
        -------
        tuple[float]
            Signature parameter tuple.

        """
        if ignore_global_phase:
            return (0.0,)
        return (float(self.phi0),)

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
        phi0 = float(self.phi0)
        return cast(
            RCArray,
            (np.exp(1j * phi0) * np.ones_like(w, dtype=complex)).astype(complex),
        )

    def _as_expansion(self) -> GaussianTransferExpansion:
        r"""Convert the transfer into a Gaussian expansion.

        Returns
        -------
        GaussianTransferExpansion
            Expansion consisting only of a constant term,

            .. math::

                H(\omega) = c_0,
                \qquad
                c_0 = e^{i\phi_0}.

        Notes
        -----
        Although :meth:`apply_to_gaussian` is implemented directly for
        efficiency and clarity, providing this expansion keeps the class
        compatible with Gaussian-closed analytic machinery that may rely
        on :meth:`_as_expansion`.

        """
        return GaussianTransferExpansion(
            c0=complex(np.exp(1j * float(self.phi0))),
            atoms=(),
        )

    def apply_to_gaussian(
        self,
        env: GaussianClosedEnvelope,
    ) -> tuple[GaussianClosedEnvelope, float]:
        r"""Apply the transfer analytically to a Gaussian-closed envelope.

        Parameters
        ----------
        env:
            Input envelope in the Gaussian-closed family.

        Returns
        -------
        tuple[GaussianClosedEnvelope, float]
            Pair ``(env_out, eta)`` where ``env_out`` is the transformed
            Gaussian-closed envelope and ``eta = 1.0``.

        Notes
        -----
        For a single Gaussian envelope, the phase is absorbed into the
        Gaussian phase parameter :math:`\phi_0`.

        For a Gaussian mixture, the mixture weights are multiplied by the
        common phase factor :math:`e^{i\phi_0}`. The represented mode is
        unchanged up to a global phase, and no loss is introduced.

        """
        dphi = float(self.phi0)

        if isinstance(env, GaussianEnvelope):
            return (
                GaussianEnvelope(
                    omega0=float(env.omega0),
                    sigma=float(env.sigma),
                    tau=float(env.tau),
                    phi0=float(env.phi0 + dphi),
                ),
                1.0,
            )

        if isinstance(env, GaussianMixtureEnvelope):
            phase = complex(np.exp(1j * dphi))
            return (
                GaussianMixtureEnvelope(
                    components=env.components,
                    weights=np.asarray(env.weights, dtype=complex) * phase,
                ),
                1.0,
            )

        raise TypeError(
            "ConstantPhase supports GaussianEnvelope or GaussianMixtureEnvelope"
        )


if TYPE_CHECKING:
    from symop.core.protocols.modes.transfer import TransferFunction

    _check: TransferFunction = ConstantPhase(phi0=1.0)
