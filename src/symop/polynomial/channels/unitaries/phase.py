"""Single-mode phase unitary.

Utility for constructing the 1×1 unitary representing a phase shift
applied to a single optical mode.
"""

from __future__ import annotations

import math

import numpy as np


def phase_u(*, phi: float) -> np.ndarray:
    r"""Return the 1x1 phase-shifter unitary.

    .. math::

        U = \begin{pmatrix} e^{i\phi} \end{pmatrix}.

    Returns
    -------
    numpy.ndarray
        Complex array of shape (1, 1).

    """
    e = complex(math.cos(phi), math.sin(phi))
    return np.asarray([[e]], dtype=np.complex128)
