from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from symop_proto.core.protocols import SignatureProto
from symop_proto.labels.protocols import (
    ModeLabelProto,
    PathLabelProto,
    PolraizationLabelProto,
)


@dataclass(frozen=True)
class ModeLabel(ModeLabelProto):
    path: PathLabelProto
    pol: PolraizationLabelProto

    def with_path(self, path: PathLabelProto) -> ModeLabel:
        return replace(self, path=path)

    def with_pol(self, pol: PolraizationLabelProto) -> ModeLabel:
        return replace(self, pol=pol)

    def overlap(self, other: ModeLabel) -> complex:
        return self.path.overlap(other.path) * self.pol.overlap(other.pol)

    @property
    def signature(self) -> SignatureProto:
        return ("mode_label", self.path.signature, self.pol.signature)

    def approx_signature(self, **kw: Any) -> SignatureProto:
        return (
            "mode_label_approx",
            self.path.approx_signature(**kw),
            self.pol.approx_signature(**kw),
        )
