"""Unitary matrices for basic two-mode linear optical elements.

This module provides small helpers for constructing 2x2 unitary
transformations used in linear optical models, such as beamsplitters
and loss dilations.
"""

from __future__ import annotations

import math

import numpy as np


def beamsplitter_u(
    *,
    t: float,
    r: float,
    phi_t: float = 0.0,
    phi_r: float = 0.0,
) -> np.ndarray:
    r"""Return a 2×2 beamsplitter unitary.

    The matrix implements a linear optical beamsplitter under the
    package Heisenberg convention, where creation operators transform as

    .. math::

        a^\dagger_{\mathrm{out},k}
        =
        \sum_j U_{k j}\, a^\dagger_{\mathrm{in},j}.

    A convenient SU(2) parameterization is

    .. math::

        U =
        \begin{pmatrix}
            t e^{i\phi_t} & r e^{i\phi_r} \\
            -r e^{-i\phi_r} & t e^{-i\phi_t}
        \end{pmatrix},

    which is unitary when :math:`t^2 + r^2 = 1`.

    Parameters
    ----------
    t :
        Transmission amplitude.
    r :
        Reflection amplitude.
    phi_t :
        Phase applied to the transmission amplitude.
    phi_r :
        Phase applied to the reflection amplitude.

    Returns
    -------
    ndarray
        Complex 2×2 unitary matrix representing the beamsplitter.

    Notes
    -----
    This function does not enforce physical constraints. Callers are
    responsible for ensuring that :math:`t^2 + r^2 = 1` if a unitary
    transformation is required.

    """
    tt = float(t)
    rr = float(r)

    et = complex(math.cos(phi_t), math.sin(phi_t))
    er = complex(math.cos(phi_r), math.sin(phi_r))

    return np.asarray(
        [
            [tt * et, rr * er],
            [-rr * np.conjugate(er), tt * np.conjugate(et)],
        ],
        dtype=np.complex128,
    )


def loss_dilation_u(*, eta: float) -> np.ndarray:
    r"""Return the beamsplitter unitary used in pure-loss dilation.

    A pure-loss channel with transmissivity :math:`\eta` can be modeled
    as a beamsplitter coupling a signal mode to a vacuum environment
    mode.

    The amplitudes are

    .. math::

        t = \sqrt{\eta}, \quad r = \sqrt{1-\eta}.

    Parameters
    ----------
    eta :
        Transmissivity of the loss channel in the interval [0, 1].

    Returns
    -------
    ndarray
        Complex 2×2 unitary implementing the loss dilation.

    """
    e = float(eta)
    t = math.sqrt(e)
    r = math.sqrt(1.0 - e)
    return beamsplitter_u(t=t, r=r, phi_t=0.0, phi_r=0.0)
