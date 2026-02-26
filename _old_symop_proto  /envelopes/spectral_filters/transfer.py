from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from symop_proto.core.protocols import SignatureProto
from symop_proto.envelopes.protocols import FloatArray, RCArray


class SpectralTransfer(Protocol):
    @property
    def signature(self) -> SignatureProto: ...

    def __call__(self, w: FloatArray) -> RCArray: ...


@dataclass(frozen=True)
class GaussianLowpass:
    r"""Gaussian spectral amplitude transfer:

    H(w) = exp(- (w - w0)^2 / (2 * sigma_w^2)) * exp(i * phi(w))

    For now we implement a real-valued lowpass (no dispersion):
    H(w) = exp(-0.5 * ((w - w0)/sigma_w)^2).
    """

    w0: float
    sigma_w: float

    @property
    def signature(self) -> SignatureProto:
        return ("gauss_lowpass", float(self.w0), float(self.sigma_w))

    def __call__(self, w: FloatArray) -> RCArray:
        w = np.asarray(w, dtype=float)
        x = (w - float(self.w0)) / float(self.sigma_w)
        return np.exp(-0.5 * x * x).astype(complex)
