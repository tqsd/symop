"""Composite mode labels.

This module defines :class:`ModeLabel`, a composite label combining
path, polarization, and envelope components. The total mode overlap
factorizes into the product of overlaps of its constituent labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Self

from symop.core.protocols.modes.labels import (
    Envelope as EnvelopeProtocol,
)
from symop.core.protocols.modes.labels import (
    Path as PathProtocol,
)
from symop.core.protocols.modes.labels import (
    Polarization as PolarizationProtocol,
)
from symop.core.types.signature import Signature


@dataclass(frozen=True)
class ModeLabel:
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
    polarization:
        Polarization label.
    envelope:
        Envelope label.

    """

    path: PathProtocol
    polarization: PolarizationProtocol
    envelope: EnvelopeProtocol

    def with_envelope(self, envelope: EnvelopeProtocol) -> Self:
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

    def with_path(self, path: PathProtocol) -> Self:
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

    def with_polarization(self, polarization: PolarizationProtocol) -> Self:
        r"""Return a copy with a replaced polarization label.

        Parameters
        ----------
        polarization:
            New polarization label.

        Returns
        -------
        ModeLabel
            Updated mode label.

        """
        return replace(self, polarization=polarization)

    def overlap(self, other: ModeLabel) -> complex:
        r"""Compute the overlap with another mode label.

        Parameters
        ----------
        other:
            Another mode label.

        Returns
        -------
        complex
            Product of path, polarization, and envelope overlaps.

        """
        return (
            self.path.overlap(other.path)
            * self.polarization.overlap(other.polarization)
            * self.envelope.overlap(other.envelope)
        )

    @property
    def signature(self) -> Signature:
        r"""Stable signature for caching/comparison.

        Returns
        -------
        Signature
            Tuple identifying the mode label and its components.

        """
        return (
            "mode_label",
            self.path.signature,
            self.polarization.signature,
            self.envelope.signature,
        )

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> Signature:
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
        Signature
            Approximate signature tuple.

        """
        return (
            "mode_label_approx",
            self.path.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            ),
            self.polarization.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            ),
            self.envelope.approx_signature(
                decimals=decimals, ignore_global_phase=ignore_global_phase
            ),
        )


if TYPE_CHECKING:
    from symop.core.protocols.modes import ModeLabel as ModeLabelProtocol
    from symop.modes.envelopes import GaussianEnvelope
    from symop.modes.labels.path import Path
    from symop.modes.labels.polarization import Polarization

    _env = GaussianEnvelope(omega0=10, sigma=10, tau=0)
    _mode_label: ModeLabelProtocol = ModeLabel(
        envelope=_env, polarization=Polarization.D(), path=Path("A")
    )
