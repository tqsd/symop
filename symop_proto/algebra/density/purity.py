from __future__ import annotations
from typing import Tuple
from symop_proto.core.protocols import DensityTermProto
from .inner import density_inner


def density_purity(terms: Tuple[DensityTermProto, ...]) -> float:
    return float(density_inner(terms, terms).real)
