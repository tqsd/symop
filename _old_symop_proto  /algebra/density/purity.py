from __future__ import annotations

from symop_proto.core.protocols import DensityTermProto

from .inner import density_inner


def density_purity(terms: tuple[DensityTermProto, ...]) -> float:
    return float(density_inner(terms, terms).real)
