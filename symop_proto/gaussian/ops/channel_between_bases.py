from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from symop_proto.gaussian.basis import ModeBasis
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.ops.ladder_moments import (
    pack_centered_ladder_K,
    unpack_centered_ladder_K,
)


def apply_ladder_affine_between_bases(
    core: GaussianCore,
    *,
    idx_in: Sequence[int],
    basis_out: ModeBasis,
    X: np.ndarray,
    Y: np.ndarray,
    d0: Optional[np.ndarray] = None,
    check_finite: bool = True,
) -> GaussianCore:
    """
    Apply a ladder-affine Gaussian channel from an input subset (old basis)
    to a *new* output basis.

    Input uses the subset indices idx_in in core.basis.
    Output lives on basis_out (size m).

    X,Y,d0 are defined on the ladder vector r = (a, a^dag) for the
    corresponding sizes:
      X: (2m, 2k)
      Y: (2m, 2m)
      d0: (2m,)
    """
    idx = list(idx_in)
    k = len(idx)
    m = basis_out.n

    if X.shape != (2 * m, 2 * k):
        raise ValueError(f"X must have shape {(2*m, 2*k)}, got {X.shape}")
    if Y.shape != (2 * m, 2 * m):
        raise ValueError(f"Y must have shape {(2*m, 2*m)}, got {Y.shape}")
    if d0 is None:
        d0v = np.zeros((2 * m,), dtype=complex)
    else:
        d0v = np.asarray(d0, dtype=complex).reshape(2 * m)

    if check_finite:
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite entries")
        if not np.isfinite(Y).all():
            raise ValueError("Y contains non-finite entries")
        if not np.isfinite(d0v).all():
            raise ValueError("d0 contains non-finite entries")

    idx_arr = np.asarray(idx, dtype=int)

    alpha_in = core.alpha[idx_arr].copy()
    N_in = core.N[np.ix_(idx_arr, idx_arr)].copy()
    M_in = core.M[np.ix_(idx_arr, idx_arr)].copy()
    G_in = core.basis.gram[np.ix_(idx_arr, idx_arr)].copy()

    # centered input moments
    N0_in = N_in - np.outer(alpha_in.conj(), alpha_in)
    M0_in = M_in - np.outer(alpha_in, alpha_in)

    K_in = pack_centered_ladder_K(N0_in, M0_in, G_in)

    r_in = np.concatenate([alpha_in, alpha_in.conj()], axis=0)

    r_out = X @ r_in + d0v
    K_out = X @ K_in @ X.T + Y

    alpha_out = r_out[0:m].copy()
    G_out = basis_out.gram

    N0_out, M0_out = unpack_centered_ladder_K(K_out, G_out)

    N_out = N0_out + np.outer(alpha_out.conj(), alpha_out)
    M_out = M0_out + np.outer(alpha_out, alpha_out)

    return GaussianCore.from_moments(
        basis_out, alpha=alpha_out, N=N_out, M=M_out
    )
