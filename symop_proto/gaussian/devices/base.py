from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from symop_proto.core.protocols import ModeOpProto
from symop_proto.devices.base import BaseDevice, DeviceApplyOptions
from symop_proto.devices.io import DeviceIO, DeviceResult
from symop_proto.gaussian.basis import ModeBasis
from symop_proto.gaussian.core import GaussianCore


@dataclass(frozen=True)
class GaussianDevice(BaseDevice[GaussianCore]):
    """
    Base class for Gaussian devices.

    A Gaussian device is a state transformer operating on
    :class:`symop_proto.gaussian.core.GaussianCore`.
    Concrete devices implement :meth:`_apply_gaussian`.


    Notes
    -----
    - This base class is thin and avoid import cycles
    - Device-level logic (selecting modes, creating env modes, routing)
      belongs here, not in gaussian map kernels.
    """

    def _apply_gaussian(
        self, state: GaussianCore, *, options: Optional[DeviceApplyOptions]
    ) -> DeviceResult[GaussianCore]:
        raise NotImplementedError

    def _trace_out(
        self, state: GaussianCore, modes: Sequence[ModeOpProto]
    ) -> GaussianCore:
        return state.trace_out(tuple(modes))

    def _keep(
        self, state: GaussianCore, modes: Sequence[ModeOpProto]
    ) -> GaussianCore:
        return state.keep(tuple(modes))

    def _relabel(
        self,
        state: GaussianCore,
        mode_map: Sequence[Tuple[ModeOpProto, ModeOpProto]],
        *,
        atol: float = 1e-12,
    ) -> GaussianCore:
        old_basis = state.basis
        new_modes = list(old_basis.modes)

        for in_mode, out_mode in mode_map:
            i = old_basis.require_index_of(in_mode)
            new_modes[i] = out_mode

        sigs = [m.signature for m in new_modes]
        if len(set(sigs)) != len(sigs):
            raise ValueError(
                "Relabel produced duplicate modes (signature collision)"
            )

        new_basis = ModeBasis.build(tuple(new_modes))

        if not np.allclose(
            new_basis.gram, old_basis.gram, atol=atol, rtol=0.0
        ):
            raise ValueError(
                "Relabel changed Gram matrix. This is not a pure relabel;"
                "you likely changed env or a mode-defining attribute."
            )

        return GaussianCore.from_moments(
            new_basis, alpha=state.alpha, N=state.N, M=state.M
        )

    def resolve_io(self, state: GaussianCore) -> DeviceIO:
        raise NotImplementedError

    def do_apply(self, state: GaussianCore, io: DeviceIO):
        raise NotImplementedError
