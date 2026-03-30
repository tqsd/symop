r"""Base classes for Gaussian-closed spectral transfer functions.

This module defines implementation-oriented base classes for transfer
functions that admit a closed-form analytic action on Gaussian-closed
mode envelopes.

The main class exported here is :class:`GaussianClosedTransferBase`,
which extends :class:`~symop.modes.transfer.base.TransferBase` with
support for analytic application to the Gaussian-closed envelope family.

Design
------
The Gaussian-closed transfer layer captures transfers :math:`H(\omega)`
for which filtering can be performed without numerical FFT-based
sampling. Such transfers preserve a closed analytic description of the
envelope and return the transmitted power separately.

Two broad patterns occur in this layer:

- transfers represented by a Gaussian expansion, delegated through
  :class:`~symop.modes.transfer.gaussian.formalism.GaussianTransferExpansion`
- transfers that act by directly transforming Gaussian envelope
  parameters

The base class in this module provides the shared analytic application
entry point for the expansion-based case.

Notes
-----
This module is intentionally focused on the Gaussian-closed family only.
Transfers that do not admit such a representation should inherit from
the generic :class:`~symop.modes.transfer.base.TransferBase` instead and
use the numerical filtering path.

"""

from __future__ import annotations

from abc import abstractmethod

from symop.modes.protocols.envelope import GaussianClosedEnvelope
from symop.modes.protocols.transfer import SupportsGaussianClosedTransfer
from symop.modes.transfer.base import TransferBase
from symop.modes.transfer.gaussian.formalism import GaussianTransferExpansion


class GaussianClosedTransferBase(
    TransferBase,
    SupportsGaussianClosedTransfer,
):
    r"""Base class for transfers with Gaussian-closed analytic action.

    Subclasses of this base represent transfer functions
    :math:`H(\omega)` that can be applied analytically to envelopes in
    the Gaussian-closed family.

    The only method subclasses must implement is
    :meth:`_as_expansion`, which converts the transfer into an internal
    Gaussian expansion representation. The generic
    :meth:`apply_to_gaussian` implementation then delegates the actual
    filtering to that expansion.
    """

    @abstractmethod
    def _as_expansion(self) -> GaussianTransferExpansion:
        """Convert the transfer into a Gaussian expansion.

        Returns
        -------
        GaussianTransferExpansion
            Internal expansion object representing the same transfer in a
            form suitable for analytic Gaussian filtering.

        """
        raise NotImplementedError

    def apply_to_gaussian(
        self,
        env: GaussianClosedEnvelope,
    ) -> tuple[GaussianClosedEnvelope, float]:
        r"""Apply the transfer analytically to a Gaussian-closed envelope.

        Parameters
        ----------
        env
            Input envelope belonging to the Gaussian-closed family.

        Returns
        -------
        tuple[GaussianClosedEnvelope, float]
            Pair ``(env_out, eta)`` where ``env_out`` is the filtered
            Gaussian-closed envelope and ``eta`` is the transmitted power.

        Notes
        -----
        This method delegates the actual computation to the Gaussian
        transfer expansion returned by :meth:`_as_expansion`.

        The output envelope is typically another Gaussian-closed
        descriptor, often a Gaussian mixture if the transfer broadens the
        representation beyond a single Gaussian atom.

        """
        expansion = self._as_expansion()
        out_env, eta = expansion.apply_to_gaussian(env)
        return out_env, float(eta)
