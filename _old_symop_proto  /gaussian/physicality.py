from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from symop_proto.gaussian.core import GaussianCore


def _project_to_affine_fixed_imag(X: np.ndarray, imag_target: np.ndarray) -> np.ndarray:
    """
    Project a complex matrix onto the affine set of Hermitian matrices
    with fixed imaginary part imag_target.

    Returns a Hermitian matrix: Re(X) + 1j * imag_target, symmetrized.
    """
    Y = X.real + 1j * imag_target
    return 0.5 * (Y + Y.conj().T)


def _min_eig_hermitian(X: np.ndarray) -> float:
    Xh = 0.5 * (X + X.conj().T)
    return float(np.min(np.linalg.eigvalsh(Xh).real))


def _project_psd_fixed_imag(
    H: np.ndarray,
    *,
    atol: float,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Alternating projections to find a PSD matrix with the same imaginary part
    as H (i.e. stay on the uncertainty-matrix affine slice).

    Returns H_star approximately in:
      {X : X >= 0, Im(X) = Im(H), X Hermitian}.
    """
    imag_target = H.imag
    X = 0.5 * (H + H.conj().T)

    for _ in range(max_iter):
        # Project to PSD cone
        X_psd = _project_psd_hermitian(X, atol=atol)

        # Project back to affine slice (fixed imaginary part)
        X_next = _project_to_affine_fixed_imag(X_psd, imag_target)

        # Convergence checks
        if np.linalg.norm(X_next - X, ord="fro") <= float(tol) * max(
            1.0, np.linalg.norm(X, ord="fro")
        ):
            X = X_next
            break
        X = X_next

        # Early exit if already physical enough
        if _min_eig_hermitian(X) >= -float(atol):
            break

    return X


def _project_psd_hermitian(H: np.ndarray, *, atol: float) -> np.ndarray:
    """
    Frobenius-closest projection of a Hermitian matrix onto the PSD cone
    by eigenvalue clipping.
    """
    Hh = 0.5 * (H + H.conj().T)
    w, Q = np.linalg.eigh(Hh)
    w2 = np.maximum(w, -float(atol))
    return (Q * w2) @ Q.conj().T


def _centered_from_cov_canonical(
    V: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert (canonical) mapping from centered moments (N0, M0) to
    quadrature covariance V.

    Assumes canonical basis (G=I) and your quadrature conventions.

    For canonical centered moments:
        Vxx = N0 + Re(M0) + 0.5 I
        Vpp = N0 - Re(M0) + 0.5 I
        Vxp = Im(M0)   (symmetrized off-diagonal)
    """
    V = np.asarray(V, dtype=float)
    n2 = V.shape[0]
    if n2 % 2 != 0:
        raise ValueError("V must have shape (2n,2n)")
    n = n2 // 2

    Vxx = V[0:n, 0:n]
    Vpp = V[n : 2 * n, n : 2 * n]
    Vxp = V[0:n, n : 2 * n]

    I = np.eye(n, dtype=float)

    N0 = 0.5 * (Vxx + Vpp) - 0.5 * I
    ReM = 0.5 * (Vxx - Vpp)
    ImM = Vxp
    M0 = ReM + 1j * ImM

    # enforce algebraic symmetries
    N0 = 0.5 * (N0 + N0.T)
    M0 = 0.5 * (M0 + M0.T)
    return N0.astype(complex), M0.astype(complex)


@dataclass(frozen=True)
class ProjectionReport:
    added_noise: float
    method: str


def project_to_physical_gaussian(
    core: GaussianCore,
    *,
    method: Literal["psd_uncertainty", "add_isotropic_noise"] = "psd_uncertainty",
    atol: float = 1e-12,
    canonical_eps: float = 1e-10,
) -> Tuple[GaussianCore, ProjectionReport]:
    """
    Project a Gaussian core to a physical Gaussian core.

    Two methods:

    1) "psd_uncertainty" (recommended):
       Form H = V + i Omega/2 and project H to PSD by eigenvalue clipping.
       Then use V_proj = Re(H_proj) and reconstruct moments.
       Fully correct reconstruction is implemented for canonical bases (G ~ I).
       For non-canonical bases, it falls back to additive noise.

    2) "add_isotropic_noise":
       Add the minimal scalar t >= 0 such that H becomes PSD:
           H' = H + t I
       and map this back as N0 += t/2 I, M0 unchanged.
       Works as a conservative "repair" for any basis.

    Returns
    -------
    (GaussianCore, ProjectionReport)
    """
    if core.is_physical(atol=atol):
        return core, ProjectionReport(added_noise=0.0, method="noop")

    V = np.asarray(core.quadrature_covariance(), dtype=float)
    Omega = core.symplectic_form()
    H = V + 0.5j * Omega

    if method == "psd_uncertainty":
        # We need a physical output, i.e. H_out = V_out + i/2 Omega is PSD,
        # while Omega is fixed by the basis Gram matrix.
        #
        # A plain PSD projection changes Im(H) and breaks this constraint, so
        # we project onto the intersection of:
        #   (i) PSD cone
        #  (ii) affine set with fixed Im(H) (equivalently fixed Omega)
        H_star = _project_psd_fixed_imag(H, atol=atol)

        # Since Im(H_star) matches Im(H)=Omega/2, the projected covariance is:
        V_proj = H_star.real
        V_proj = 0.5 * (V_proj + V_proj.T)

        # Only reconstruct moments in the canonical case (G ~ I), where the
        # inverse mapping V -> (N0, M0) is implemented.
        if core.basis.is_canonical(eps=canonical_eps):
            N0, M0 = _centered_from_cov_canonical(V_proj)

            alpha = core.alpha
            N = N0 + np.outer(alpha.conj(), alpha)
            M = M0 + np.outer(alpha, alpha)

            # Stabilize algebraic symmetries
            N = 0.5 * (N + N.conj().T)
            M = 0.5 * (M + M.T)

            out = GaussianCore(basis=core.basis, alpha=alpha.copy(), N=N, M=M)

            # Diagnostic: how much the covariance changed (Frobenius norm).
            added = float(np.linalg.norm(V_proj - V, ord="fro"))

            return out, ProjectionReport(
                added_noise=added,
                method="psd_uncertainty",
            )

        # Non-canonical: fall back to additive noise repair
        method = "add_isotropic_noise"

    if method == "add_isotropic_noise":
        # minimal scalar shift to make H PSD
        Hh = 0.5 * (H + H.conj().T)
        w = np.linalg.eigvalsh(Hh).real
        min_w = float(np.min(w))
        t = max(0.0, -min_w + float(atol))

        # conservative mapping back to moments: add t/2 I to centered N0
        n = core.basis.n
        alpha = core.alpha
        N0 = core.N - np.outer(alpha.conj(), alpha)
        M0 = core.M - np.outer(alpha, alpha)

        N0_2 = N0 + t * np.eye(n, dtype=complex)
        N2 = N0_2 + np.outer(alpha.conj(), alpha)
        M2 = M0 + np.outer(alpha, alpha)

        N2 = 0.5 * (N2 + N2.conj().T)
        M2 = 0.5 * (M2 + M2.T)

        out = GaussianCore(basis=core.basis, alpha=alpha.copy(), N=N2, M=M2)
        return out, ProjectionReport(added_noise=t, method="add_isotropic_noise")

    raise ValueError(f"Unknown method: {method}")
