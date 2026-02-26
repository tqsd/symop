from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from symop_proto.core.protocols import ModeOpProto


def gram_block(
    modes_left: Sequence[ModeOpProto],
    modes_right: Sequence[ModeOpProto],
    *,
    tol: float = 0.0,
) -> np.ndarray:
    """Return the Gram block G_lr with entries [l_i, r_j^dag].

    Uses mi.ann.commutator(mj.create), consistent with ModeBasis.build().
    """
    m = len(modes_left)
    n = len(modes_right)
    G = np.zeros((m, n), dtype=complex)
    for i, mi in enumerate(modes_left):
        for j, mj in enumerate(modes_right):
            val = mi.ann.commutator(mj.create)
            if tol > 0.0 and abs(val) < tol:
                val = 0.0 + 0.0j
            G[i, j] = val
    return G


def projection_C_and_env_gram(
    modes_out: Sequence[ModeOpProto],
    modes_in: Sequence[ModeOpProto],
    *,
    rcond: float = 1e-12,
    tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the commutator-consistent projection b ~= C a + e.

    Let:
      G_aa[i,j] = [a_i, a_j^dag]
      G_bb[i,j] = [b_i, b_j^dag]
      G_ba[i,j] = [b_i, a_j^dag]

    Then:
      C = G_ba @ pinv(G_aa)
      G_e = G_bb - C @ G_aa @ C^dag

    G_e is the Gram (commutator) of the environment contribution e,
    required for CCR preservation.
    """
    G_aa = gram_block(modes_in, modes_in, tol=tol)
    G_bb = gram_block(modes_out, modes_out, tol=tol)
    G_ba = gram_block(modes_out, modes_in, tol=tol)

    G_aa_pinv = np.linalg.pinv(G_aa, rcond=float(rcond))
    C = G_ba @ G_aa_pinv

    G_e = G_bb - C @ G_aa @ C.conj().T
    G_e = 0.5 * (G_e + G_e.conj().T)
    return C, G_e


def validate_env_gram_psd(G_e: np.ndarray, *, atol: float = 1e-12) -> None:
    """Raise if G_e is not Hermitian PSD within tolerance."""
    if not np.allclose(G_e, G_e.conj().T, atol=float(atol), rtol=0.0):
        raise ValueError("G_e is not Hermitian within tolerance")
    ev = np.linalg.eigvalsh(G_e)
    if float(np.min(ev)) < -float(atol):
        raise ValueError(f"G_e not PSD: min eigenvalue {float(np.min(ev))}")
