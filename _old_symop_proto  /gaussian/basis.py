from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from symop_proto.core.protocols import ModeOpProto, SignatureProto


@dataclass(frozen=True)
class ModeBasis:
    r"""Examples
    --------
    Orthogonal polarization labels lead to a diagonal Gram matrix:

    .. jupyter-execute::

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)

        mH = ModeOp(env=env,
                    label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        mV = ModeOp(env=env,
                    label=ModeLabel(PathLabel("A"), PolarizationLabel.V()))

        B = ModeBasis.build([mH, mV])

        print("Gram matrix:")
        print(B.gram)
        print("Off-diagonal element G(H,V) =", B.gram[0, 1])

    If two temporal modes overlap partially, the Gram matrix becomes
    non-trivial:

    .. jupyter-execute::

        env1 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        env2 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.2, phi0=0.0)

        m1 = ModeOp(env=env1,
                    label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env2,
                    label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))

        B = ModeBasis.build([m1, m2])

        print("Gram matrix:")
        print(B.gram)

    """

    modes: tuple[ModeOpProto, ...]
    gram: np.ndarray
    index_by_sig: dict[tuple, int]

    def __post_init__(self) -> None:
        n = len(self.modes)

        if self.gram.shape != (n, n):
            raise ValueError(
                f"gram matrix must have shape ({n},{n}), got {self.gram.shape}"
            )
        if len(self.index_by_sig) != n:
            raise ValueError("index_by_sig must have one entry per mode")

        idxs = sorted(self.index_by_sig.values())
        if idxs != list(range(n)):
            raise ValueError("index_by_sig values must be a permutation of 0..n-1")

        for i, m in enumerate(self.modes):
            sig = m.signature
            if sig not in self.index_by_sig:
                raise ValueError("mode signature missing from index_by_sig")
            if self.index_by_sig[sig] != i:
                raise ValueError("index_by_sig does not match modes order")

    @property
    def n(self) -> int:
        """Number of unique modes in the basis"""
        return len(self.modes)

    @staticmethod
    def build(
        modes: Iterable[ModeOpProto],
        *,
        merge_approx: bool = False,
        env_kw: dict[str, Any] | None = None,
        tol: float = 0.0,
    ) -> ModeBasis:
        """Construct a basis from an iterable of modes"""
        env_kw = env_kw or {}
        unique_modes: list[ModeOpProto] = []
        seen_modes: dict[tuple[Any, ...], int] = {}

        for m in modes:
            key = m.signature if not merge_approx else m.approx_signature(**env_kw)
            if key in seen_modes:
                continue
            seen_modes[key] = len(unique_modes)
            unique_modes.append(m)

        modes_t = tuple(unique_modes)
        n = len(modes_t)

        G = np.zeros((n, n), dtype=complex)
        for i, mi in enumerate(modes_t):
            for j, mj in enumerate(modes_t):
                if j < i:
                    continue
                val = mi.ann.commutator(mj.create)
                if tol > 0.0 and abs(val) < tol:
                    val = 0.0 + 0.0j
                G[i, j] = val
                G[j, i] = np.conjugate(val)

        index_by_sig = {m.signature: i for i, m in enumerate(modes_t)}
        return ModeBasis(modes=modes_t, gram=G, index_by_sig=index_by_sig)

    def union(
        self,
        modes: Iterable[ModeOpProto],
        *,
        merge_approx: bool = False,
        env_kw: dict[str, Any] | None = None,
        tol: float = 0.0,
    ) -> ModeBasis:
        """Return a new basis that extends this basis with additional modes.

        Preserves the current ordering; new unique modes are appended
        """
        merged = list(self.modes)

        merged.extend(list(modes))
        return ModeBasis.build(
            merged, merge_approx=merge_approx, env_kw=env_kw, tol=tol
        )

    def index_of(self, mode: ModeOpProto) -> int:
        """Return the basis index for a mode (by exact signature)"""
        return self.index_by_sig[mode.signature]

    def index_of_sig(self, sig: SignatureProto) -> int:
        return self.index_by_sig[sig]

    def require_index_of(self, mode: ModeOpProto) -> int:
        sig = mode.signature
        if sig not in self.index_by_sig:
            known = list(self.index_by_sig.keys())
            raise KeyError(f"mode not in basis: {sig}; known: {known}")
        return self.index_by_sig[sig]

    def require_index_of_sig(self, sig: SignatureProto) -> int:
        if sig not in self.index_by_sig:
            known = list(self.index_by_sig.keys())
            raise KeyError(f"signature not in basis: {sig}; known: {known}")
        return self.index_by_sig[sig]

    def is_canonical(self, eps: float = 1e-12) -> bool:
        """Checks whether basis is approximately orthonormal"""
        if self.n == 0:
            return True
        Id = np.eye(self.n, dtype=complex)
        return bool(np.max(np.abs(self.gram - Id)) <= float(eps))

    def is_hermitian(self, eps: float = 1e-12) -> bool:
        return bool(np.allclose(self.gram, self.gram.conj().T, atol=eps))

    def is_positive_semidefinite(self, eps: float = 1e-12):
        eigvals = np.linalg.eigvalsh(self.gram)
        return bool(np.min(eigvals) >= -float(eps))

    def validate(
        self, *, hermitian: bool = True, psd: bool = False, eps: float = 1e-12
    ) -> None:
        if hermitian and not self.is_hermitian(eps=eps):
            raise ValueError("gram is not Hermitian within tolerance")
        if psd and not self.is_positive_semidefinite(eps=eps):
            raise ValueError("gram is not positive semidefinite within tolerance")
