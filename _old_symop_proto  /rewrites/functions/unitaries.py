from __future__ import annotations
from typing import Final

import numpy as np

_COMPLEX: Final = np.complex128


def identity_unitary(n: int) -> np.ndarray:
    r"""
    Return the :math:`n\times n` identity matrix (complex)


    Parameters:
    -----------
    - n: Dimension of the identity (must be >= 0)

    Returns:
    --------
    - ``numpy.ndarray``: Array with shape ``(n,n)`` and type ``complex128``

    Raises:
    -------
    ``ValueError``: if ``n`` is negative

    Mathematics:
    ------------
        .. math::

            U = \mathbb{I}_n

    """
    if n < 0:
        raise ValueError("identity_unitary: n must be non-negative")
    return np.eye(n, dtype=_COMPLEX)


def phase_unitary(phi: float) -> np.ndarray:
    r"""
    Single-mode phase shifter.

    Parameters:
    -----------
    - phi: Phase shift in radians.

    Returns:
    --------
    numpy.ndarray: Array with shape ``(1,1)`` and dtype ``complex128``.


    Mathematics:
    ------------
        Acts on creation operators as:

        .. math::

            a^\dagger \; \mapsto \; e^{i\phi}\,a^\dagger,
            \qquad
            a \;\mapsto\; e^{-i\phi}\, a

        So the unitary is the :math:`1\times 1` matrix

        .. math::

            U =\begin{bmatrix} e^{i\phi} \end{bmatrix}


    """
    return np.array([[np.exp(1j * phi)]], dtype=_COMPLEX)


def beamsplitter_unitary(theta: float, phi: float = 0.0) -> np.ndarray:
    r"""
    Two-mode beamsplitter unitary (passive, number-conserving)

    Parameters:
    -----------
    - theta: Mixing angleu (radians).
    - phi: Internal phase (radians). Defaults to ``0.0``.

    Returns:
    --------
    numpy.ndarray: Array with shape ``(2,2)`` and dtype ``complex128``.

    Mathematics:
    ------------
        Convention (acts on the *column* of the creation operators)
        :math:`(a_1^\dagger, a_2^\dagger)^T`:

        .. math::

            U_\text{BS}(\theta, \phi) \;=\;
            \begin{bmatrix}
                \cos\theta & e^{i\phi}\,\sin\theta\\
                -e^{-i\phi}\,\sin\theta & \cos\theta
            \end{bmatrix}

        This yields the operator map

        .. math::

            a_k^\dagger \;\mapsto\; \sum_j U_{j k}\, a_j^\dagger,
            \qquad
            a_k \; \mapsto \; \sum_j U_{j k}^*\, a_j

    Notes:
    ------
    - Transmissivity :math:`T=\cos^2\theta`, reflectivity
      :math:`R=\sin^2\theta`.
    - :math:`\phi` is the internal beamsplitter phase.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    eip = np.exp(1j * phi)
    eim = np.exp(-1j * phi)
    return np.array([[c, eip * s], [-eim * s, c]], dtype=_COMPLEX)


def pol_rotation_unitary(theta: float, chi: float) -> np.ndarray:
    r"""
    Polarization rotation / general waveplate acting on the :math:`(H,V)` basis.

    Mathematics:
    ------------

        Using the standard decomposition

        .. math::

            U(\theta,\chi) \;=\; R(\theta)\,
            \operatorname{diag}\!\big(1, e^{\,i\chi}\big)\,
            R(-\theta), \qquad
            R(\theta)=
            \begin{bmatrix}
                \cos\theta & \sin\theta \\
                -\sin\theta & \cos\theta
            \end{bmatrix}.

    Special cases
    -------------
    - Quarter-wave plate: :math:`\chi=\pi/2`.
    - Half-wave plate: :math:`\chi=\pi` (swaps :math:`H\leftrightarrow V` up to phases when
      :math:`\theta=\pi/4`).

    Parameters
    ----------
    theta
        Fast-axis orientation (radians).
    chi
        Retardance (radians).

    Returns
    -------
    numpy.ndarray
        Array with shape ``(2, 2)`` and dtype ``complex128``.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, s], [-s, c]], dtype=_COMPLEX)
    Rm = np.array([[c, -s], [s, c]], dtype=_COMPLEX)
    D = np.array([[1.0, 0.0], [0.0, np.exp(1j * chi)]], dtype=_COMPLEX)
    return (R @ D @ Rm).astype(_COMPLEX)
