"""Mode-swap unitary.

Utility for constructing the 2x2 permutation unitary that exchanges
two optical modes.
"""

from __future__ import annotations

import numpy as np

from .conventions import require_unitary_optional


def swap_u(
    *,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> np.ndarray:
    r"""Return the 2x2 SWAP unitary.

    .. math::

        U = \begin{pmatrix}
            0 & 1 \\
            1 & 0
        \end{pmatrix}.

    Parameters
    ----------
    check_unitary:
        If True, validate unitarity.
    atol:
        Tolerance for optional unitary check.

    Returns
    -------
    numpy.ndarray
        Complex matrix of shape (2, 2).

    """
    U = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    require_unitary_optional(U, check_unitary=check_unitary, atol=atol)
    return U
