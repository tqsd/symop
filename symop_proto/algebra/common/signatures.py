from __future__ import annotations
from typing import Any, Tuple

from symop_proto.core.monomial import Monomial
from symop_proto.core.terms import DensityTerm, KetTerm


def sig_mono(
    m: Monomial, *, approx: bool = False, **env_kw
) -> Tuple[Any, ...]:
    return m.signature if not approx else m.approx_signature(**env_kw)


def sig_ket(t: KetTerm, *, approx: bool = False, **env_kw) -> Tuple[Any, ...]:
    return t.signature if not approx else t.approx_signature(**env_kw)


def sig_density(
    t: DensityTerm, *, approx: bool = False, **env_kw
) -> Tuple[Any, ...]:
    return t.signature if not approx else t.approx_signature(**env_kw)
