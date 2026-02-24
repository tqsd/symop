from __future__ import annotations

from typing import Tuple

import numpy as np


def pack_centered_ladder_K(
    N0: np.ndarray, M0: np.ndarray, G: np.ndarray
) -> np.ndarray:
    """
    Pack centered moments into K = <dr dr^T> for r = (a, a^dag).

    Convention (k modes):
      K00 = <da da^T>              = M0
      K01 = <da da^dag^T>          = G + N0^T
      K10 = <da^dag da^T>          = N0
      K11 = <da^dag da^dag^T>      = M0^*
    """
    k = G.shape[0]
    if N0.shape != (k, k) or M0.shape != (k, k):
        raise ValueError("N0, M0 must match G shape")

    K = np.zeros((2 * k, 2 * k), dtype=complex)
    K[0:k, 0:k] = M0
    K[0:k, k : 2 * k] = G + N0.T
    K[k : 2 * k, 0:k] = N0
    K[k : 2 * k, k : 2 * k] = M0.conj()
    return K


def unpack_centered_ladder_K(
    K: np.ndarray, G: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of pack_centered_ladder_K.

    Returns (N0, M0).
    """
    k = G.shape[0]
    if K.shape != (2 * k, 2 * k):
        raise ValueError("K has wrong shape")
    M0 = K[0:k, 0:k].copy()
    N0 = K[k : 2 * k, 0:k].copy()
    return N0, M0
