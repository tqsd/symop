"""Polarization labels.

This module defines :class:`~symop.modes.labels.polarization.PolarizationLabel`,
a normalized Jones-vector label used to compute mode overlaps and build
stable signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symop.core.protocols.signature import SignatureProto
from symop.modes.protocols.labels import PolarizationLabelProto


@dataclass(frozen=True)
class PolarizationLabel(PolarizationLabelProto):
    r"""Polarization label represented by a normalized Jones vector.

    The Jones vector is stored as a 2-component complex vector:

    .. math::

        \mathbf{v} = \begin{pmatrix} a \\ b \end{pmatrix},

    normalized such that :math:`\|\mathbf{v}\|_2 = 1`.

    Canonicalization
    ----------------
    Jones vectors have a physically irrelevant global phase. This class
    canonicalizes the input vector by removing an overall phase so that the
    first non-negligible component is real and non-negative. This makes
    signatures stable and reduces spurious differences due to global phase.

    Attributes
    ----------
    jones:
        Tuple :math:`(a, b)` representing the Jones vector components.

    """

    jones: tuple[complex, complex]

    def __post_init__(self) -> None:
        r"""Normalize and canonicalize the Jones vector.

        Steps:
        1. Normalize :math:`(a, b)` so that :math:`|a|^2 + |b|^2 = 1`.
        2. Remove a global phase so that the first significant component is real.
        3. Enforce a non-negative leading real component.
        4. Chop tiny numerical noise in real/imag parts for stable signatures.

        Raises
        ------
        ValueError
            If the Jones vector is the zero vector.

        """
        a, b = self.jones
        v = np.array([a, b], dtype=complex)

        n = np.linalg.norm(v)
        if n == 0.0:
            raise ValueError("PolarizationLabel: Jones vector cannot be zero")

        v = v / n

        eps = 1e-15
        idx = None
        if abs(v[0]) > eps:
            idx = 0
        elif abs(v[1]) > eps:
            idx = 1

        if idx is not None:
            phase = np.exp(-1j * np.angle(v[idx]))
            v = v * phase

            v[idx] = v[idx].real + 0.0j
            if v[idx].real < 0:
                v = -v

        def _chopc(z: complex, eps: float = 1e-15) -> complex:
            r = 0.0 if abs(z.real) < eps else float(z.real)
            i = 0.0 if abs(z.imag) < eps else float(z.imag)
            return complex(r, i)

        object.__setattr__(
            self,
            "jones",
            (_chopc(complex(v[0])), _chopc(complex(v[1]))),
        )

    def overlap(self, other: PolarizationLabel) -> complex:
        r"""Compute the polarization overlap.

        This is the usual complex inner product of normalized Jones vectors:

        .. math::

            \langle \mathbf{v}_1, \mathbf{v}_2 \rangle = \mathbf{v}_1^\dagger \mathbf{v}_2.

        Parameters
        ----------
        other:
            Another polarization label.

        Returns
        -------
        complex
            Complex overlap :math:`\langle \mathbf{v}_1, \mathbf{v}_2 \rangle`.

        """
        v1 = np.array(self.jones, dtype=complex)
        v2 = np.array(other.jones, dtype=complex)
        return complex(np.vdot(v1, v2))

    @classmethod
    def H(cls) -> PolarizationLabel:
        r"""Horizontal linear polarization.

        Returns
        -------
        PolarizationLabel
            :math:`(1, 0)`.

        """
        return cls((1 + 0j, 0 + 0j))

    @classmethod
    def V(cls) -> PolarizationLabel:
        r"""Vertical linear polarization.

        Returns
        -------
        PolarizationLabel
            :math:`(0, 1)`.

        """
        return cls((0 + 0j, 1 + 0j))

    @classmethod
    def D(cls) -> PolarizationLabel:
        r"""Diagonal linear polarization.

        Returns
        -------
        PolarizationLabel
            :math:`\frac{1}{\sqrt{2}}(1, 1)`.

        """
        s = 2**-0.5
        return cls((s + 0j, s + 0j))

    @classmethod
    def A(cls) -> PolarizationLabel:
        r"""Anti-diagonal linear polarization.

        Returns
        -------
        PolarizationLabel
            :math:`\frac{1}{\sqrt{2}}(1, -1)`.

        """
        s = 2**-0.5
        return cls((s + 0j, -s + 0j))

    @classmethod
    def R(cls) -> PolarizationLabel:
        r"""Right-circular polarization (one common convention).

        Returns
        -------
        PolarizationLabel
            :math:`\frac{1}{\sqrt{2}}(1, -i)`.

        """
        s = 2**-0.5
        return cls((s + 0j, -1j * s))

    @classmethod
    def L(cls) -> PolarizationLabel:
        r"""Left-circular polarization (one common convention).

        Returns
        -------
        PolarizationLabel
            :math:`\frac{1}{\sqrt{2}}(1, i)`.

        """
        s = 2**-0.5
        return cls((s + 0j, 1j * s))

    @classmethod
    def linear(cls, theta: float) -> PolarizationLabel:
        r"""Linear polarization at angle :math:`\theta`.

        The Jones vector is:

        .. math::

            \mathbf{v}(\theta) = \begin{pmatrix}\cos\theta \\ \sin\theta\end{pmatrix}.

        Parameters
        ----------
        theta:
            Polarization angle (radians).

        Returns
        -------
        PolarizationLabel
            Linear polarization label at angle :math:`\theta`.

        """
        return cls((complex(np.cos(theta)), complex(np.sin(theta))))

    def rotated(self, theta: float) -> PolarizationLabel:
        r"""Rotate the polarization by angle :math:`\theta`.

        This applies the real rotation matrix:

        .. math::

            R(\theta)=\begin{pmatrix}\cos\theta & \sin\theta\\ -\sin\theta & \cos\theta\end{pmatrix}

        to the Jones vector.

        Parameters
        ----------
        theta:
            Rotation angle (radians).

        Returns
        -------
        PolarizationLabel
            Rotated polarization label.

        """
        a, b = self.jones
        c, s = np.cos(theta), np.sin(theta)
        return PolarizationLabel((c * a + s * b, -s * a + c * b))

    @property
    def signature(self) -> SignatureProto:
        r"""Stable signature for caching/comparison.

        The signature is based on the canonicalized Jones components.
        """
        a, b = self.jones
        return (
            "pol",
            float(a.real),
            float(a.imag),
            float(b.real),
            float(b.imag),
        )

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ) -> SignatureProto:
        r"""Approximate signature with rounded components.

        Parameters
        ----------
        decimals:
            Number of decimals to round to.
        ignore_global_phase:
            If True, component signatures may ignore global phase
            where applicable.

        Returns
        -------
        SignatureProto
            Signature tuple with rounded real/imag parts.

        """

        def r(z: complex) -> tuple[float, float]:
            return (round(z.real, decimals), round(z.imag, decimals))

        a, b = self.jones
        ra = r(a)
        rb = r(b)
        return ("pol_approx", *ra, *rb)
