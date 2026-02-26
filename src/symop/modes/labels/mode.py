"""Composite mode labels.

This module defines :class:`ModeLabel`, a composite label combining
path, polarization, and envelope components. The total mode overlap
factorizes into the product of overlaps of its constituent labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from symop.core.protocols.signature import SignatureProto
from symop.modes.protocols.envelope import EnvelopeProto
from symop.modes.protocols.labels import (
    ModeLabelProto,
    PathLabelProto,
    PolarizationLabelProto,
)


@dataclass(frozen=True)
class ModeLabel(ModeLabelProto):
    r"""Mode label composed of a path label and a polarization label.

    The overlap factorizes into the product of the component overlaps:

    .. math::

        \langle m_1, m_2 \rangle
        =
        \langle p_1, p_2 \rangle
        \langle \pi_1, \pi_2 \rangle
        \langle \zeta_1, \zeta_2 \rangle,

    where :math:`p` denotes the path component ,:math:`\pi` denotes the
    polarization component and :math:`\zeta` denotes the envelope component.

    Attributes
    ----------
    path:
        Path label.
    pol:
        Polarization label.

    """

    path: PathLabelProto
    pol: PolarizationLabelProto
    envelope: EnvelopeProto

    def with_envelope(self, envelope: EnvelopeProto) -> ModeLabel:
        r"""Return a copy with replaced envelope.

        Parameters
        ----------
        envelope:
            New envelope.

        Returns
        -------
        ModeLabel
            Updated mode label.

        """
        return replace(self, envelope=envelope)

    def with_path(self, path: PathLabelProto) -> ModeLabel:
        r"""Return a copy with a replaced path label.

        Parameters
        ----------
        path:
            New path label.

        Returns
        -------
        ModeLabel
            Updated mode label.

        """
        return replace(self, path=path)

    def with_pol(self, pol: PolarizationLabelProto) -> ModeLabel:
        r"""Return a copy with a replaced polarization label.

        Parameters
        ----------
        pol:
            New polarization label.

        Returns
        -------
        ModeLabel
            Updated mode label.

        """
        return replace(self, pol=pol)

    def overlap(self, other: ModeLabel) -> complex:
        r"""Compute the overlap with another mode label.

        Parameters
        ----------
        other:
            Another mode label.

        Returns
        -------
        complex
            Product of path and polarization overlaps.

        """
        return self.path.overlap(other.path) * self.pol.overlap(other.pol)

    @property
    def signature(self) -> SignatureProto:
        r"""Stable signature for caching/comparison.

        Returns
        -------
        SignatureProto
            Tuple identifying the mode label and its components.

        """
        return (
            "mode_label",
            self.path.signature,
            self.pol.signature,
            self.envelope.signature,
        )

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> SignatureProto:
        r"""Approximate signature.

        Parameters are forwarded to component :meth:`approx_signature` methods.

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
            "mode_label_approx",
            self.path.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            ),
            self.pol.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            ),
            self.envelope.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            ),
        )
