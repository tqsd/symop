from __future__ import annotations

from typing import Any

from symop_proto.core.protocols import (
    DensityTermProto,
    KetTermProto,
    MonomialProto,
)


def sig_mono(m: MonomialProto, *, approx: bool = False, **env_kw) -> tuple[Any, ...]:
    return m.signature if not approx else m.approx_signature(**env_kw)


def sig_ket(t: KetTermProto, *, approx: bool = False, **env_kw) -> tuple[Any, ...]:
    return t.signature if not approx else t.approx_signature(**env_kw)


def sig_density(
    t: DensityTermProto, *, approx: bool = False, **env_kw
) -> tuple[Any, ...]:
    return t.signature if not approx else t.approx_signature(**env_kw)
