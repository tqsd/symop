from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple
from symop_proto.core.label_protocol import LabelProto
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel


@dataclass(frozen=True)
class ModeLabel(LabelProto):
    path: PathLabel
    pol: PolarizationLabel

    def overlap(self, other: ModeLabel) -> complex:
        return self.path.overlap(other.path) * self.pol.overlap(other.pol)

    def signature(self) -> Tuple[Any, ...]:
        return ("mode_label", self.path.signature(), self.pol.signature())

    def approx_signature(self, **kw: Any) -> Tuple[Any, ...]:
        return (
            "mode_label_approx",
            self.path.approx_signature(*kw),
            self.pol.approx_signature(*kw),
        )
