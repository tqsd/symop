from __future__ import annotations

from dataclasses import dataclass

from symop_proto.gaussian.maps.base import GaussianSubsetMap


@dataclass(frozen=True)
class LossChannel(GaussianSubsetMap):
    r"""Pure-loss (attenuator) channel on a single mode implemented
    by a beamsplitter with a vacuum environment mode.

    Dilation:
        - add env mode e in vacuum
        - apply beamsplitter between (signal, env)
        - trace out env
    """
