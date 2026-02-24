from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


from symop_proto.core.protocols import ModeOpProto
from symop_proto.gaussian.core import GaussianCore


class GaussianMap(ABC):
    @abstractmethod
    def apply(self, core: GaussianCore) -> GaussianCore:
        """
        Returns a new GaussianCore with this map applied
        """

    def __call__(self, core: GaussianCore) -> GaussianCore:
        return self.apply(core)


class GaussianSubsetMap(GaussianMap, ABC):
    modes: Tuple[ModeOpProto, ...]
    check_unitary: bool
    atol: float

    def _idx(self, core: GaussianCore) -> Tuple[int, ...]:
        return tuple(core.basis.require_index_of(m) for m in self.modes)
