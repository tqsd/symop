from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.ops.common import (
    as_complex_matrix,
    as_optional_complex_vector,
    check_finite_all,
    check_is_subset_indices,
    ladder_indices,
)


def embed_subset_affine(
    n: int,
    idx: Sequence[int],
    X: np.ndarray,
    Y: np.ndarray,
    d0: np.ndarray | None = None,
    *,
    check_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Embeds a subsystem affine Gaussian map into the global ladder-operator
    space.

    This is a *basis-agnostic embedding* utility in the ladder ordering

    .. math::

        \hat r =(a_1,\ldots, a_n, a_1^\dagger, \ldots, a_n^\dagger)^{\mathsf{T}}.

    It embeds a local affine linear transformation acting only on the selected
    modes ``idx`` into the full ``2n``-dimensional ladder coordinate.

    Ladder-affine form
    ------------------

    The affine map is represented as

    .. math::

        r'=Xr +d_0

    and for any second-moment tensor expressid in the same ladder ordering
    e.g. :math:`K=\langle r r^{\mathsf T}\rangle` or its centered variant,
    the induced update has the additive-noise form

    .. math::

        K' = X K X^{\mathsf T} + Y.

    This function does not interpret :math:`X,Y,d_0` physically, it only
    embeds the provided local blocks into global matrices/vectors.

    Embedding rule
    --------------

    Let ``k=len(idx)`` and define ladder indices

    .. math::

        I = (i_1, \ldots, i_k, i_{1+n}, \ldots, i_{k+n})

    where :math:`(i_1,\ldots, i_k)` are the mode indices in ``idx``.
    Then the full objects are defined by

    - :math:`X_{\mathrm{full}}=I_{2n}` except on the block :math:`I \times I`
      where it equals the local ``X``.
    - :math:`Y_{\mathrm{full}}=0` except on the block :math:`I \times I` where
      it equals the local ``Y``.
    - :math:`d_{0, \mathrm{full}}` except on entries :math:`I` where it equals
      the local ``d0`` (or remains zero if ``d0`` is None)

    Parameters
    ----------
    n:
        Total number of modes.
    idx:
        Mode indices acted on (0-based)
    X, Y:
        Local ladder-affine matrices of shape ``(2k, 2k)`` with complex dtype.
    d0:
        Optional local ladder displacement vector of shape ``(2k,)``.
    check_finite:
        If True, rejects NaN/Inf in inputs.

    Returns
    -------
    (X_full, Y_full, d0_full):
        Embedded objects with shapes ``(2n,2n)``, ``(2n,2n)``, and ``(2n,)``


    Examples
    --------
    Embed a trivial local identity/noise-free map on mode 1 into a 2-mode space:

    .. jupyter-execute::

        import numpy as np
        from symop_proto.gaussian.ops.channel import embed_subset_affine

        n = 2
        idx = [1]  # act on mode 1 only
        X = [[2.0, 0.0], [0.0, 3.0]]
        Y = [[0.1, 0.0], [0.0, 0.2]]
        d0 = [5.0,7.0]
        X_full, Y_full, d0_full = embed_subset_affine(n, idx, X, Y, d0)

        print("X_full:")
        print(X_full)
        print("Y_full:")
        print(Y_full)
        print("d0_full:")
        print(d0_full)

    """
    check_is_subset_indices(n, idx)
    idx_list = list(idx)
    lad_idx = ladder_indices(n, idx_list)
    k2 = len(lad_idx)

    Xc = as_complex_matrix(X, shape=(k2, k2), name="X")
    Yc = as_complex_matrix(Y, shape=(k2, k2), name="Y")
    d0c = as_optional_complex_vector(d0, k2=k2, name="d0")

    if check_finite:
        check_finite_all(Xc, Yc, d0c)

    X_full = np.eye(2 * n, dtype=complex)
    Y_full = np.zeros((2 * n, 2 * n), dtype=complex)
    d0_full = np.zeros((2 * n,), dtype=complex)

    X_full[np.ix_(lad_idx, lad_idx)] = Xc
    Y_full[np.ix_(lad_idx, lad_idx)] = Yc
    if d0c is not None:
        d0_full[lad_idx] = d0c

    return X_full, Y_full, d0_full


def _pack_centered_K(core: GaussianCore) -> np.ndarray:
    n = core.basis.n
    G = core.basis.gram

    alpha = core.alpha
    N0 = core.N - np.outer(alpha.conj(), alpha)
    M0 = core.M - np.outer(alpha, alpha)

    K = np.zeros((2 * n, 2 * n), dtype=complex)
    K[0:n, 0:n] = M0
    K[0:n, n : 2 * n] = G + N0.T
    K[n : 2 * n, 0:n] = N0
    K[n : 2 * n, n : 2 * n] = M0.conj()

    return K


def _unpack_centered_K(
    core: GaussianCore, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = core.basis.n
    M0 = K[0:n, 0:n]
    N0 = K[n : 2 * n, 0:n]
    return N0, M0


def apply_ladder_affine_subset(
    core: GaussianCore,
    *,
    idx: Sequence[int],
    X: np.ndarray,
    Y: np.ndarray,
    d0: np.ndarray | None = None,
    check_finite: bool = True,
) -> GaussianCore:
    """Apply a ladder-affine Gaussian channel on a subset of modes.

    The channel is specified in ladder ordering r=(a, a^dag) as:
        r' = X r + d0
        K' = X K X^T + Y
    where K = <delta r delta r^T> is the centered second-moment matrix.

    This routine embeds the local (X,Y,d0) into the full 2n space
    (updating cross-correlations), then updates (alpha,N,M) consistently
    with GaussianCore conventions.
    """
    n = core.basis.n
    check_is_subset_indices(n, idx)

    # Embed into global ladder space
    X_full, Y_full, d0_full = embed_subset_affine(
        n=n, idx=idx, X=X, Y=Y, d0=d0, check_finite=check_finite
    )

    # Mean vector in ladder space: <r> = (alpha, alpha^*)
    r_mean = np.concatenate([core.alpha, core.alpha.conj()], axis=0)
    r_mean2 = X_full @ r_mean + d0_full
    alpha2 = r_mean2[0:n].copy()

    # Centered second moments
    K = _pack_centered_K(core)
    K2 = X_full @ K @ X_full.T + Y_full

    N0_2, M0_2 = _unpack_centered_K(core, K2)

    # Reconstruct raw moments
    N2 = N0_2 + np.outer(alpha2.conj(), alpha2)
    M2 = M0_2 + np.outer(alpha2, alpha2)

    # Stabilize algebraic symmetries
    N2 = 0.5 * (N2 + N2.conj().T)
    M2 = 0.5 * (M2 + M2.T)

    return GaussianCore(basis=core.basis, alpha=alpha2, N=N2, M=M2)
