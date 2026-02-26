from __future__ import annotations

from typing import Protocol, Union

from symop_proto.algebra.protocols import (
    DensityPolyProto,
    KetPolyProto,
    OpPolyProto,
)
from symop_proto.state.protocols import (
    DensityPolyStateProto,
    KetPolyStateProto,
)

StateLike = Union[KetPolyStateProto, DensityPolyStateProto]


class RewriteDeviceProto(Protocol):
    def on_ketpoly(self, poly: KetPolyProto) -> KetPolyProto: ...
    def on_density(self, rho: DensityPolyProto) -> DensityPolyProto: ...
    def on_oppoly(self, op: OpPolyProto) -> OpPolyProto: ...
    def on_state(self, state: StateLike) -> StateLike: ...
