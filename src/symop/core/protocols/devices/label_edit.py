from __future__ import annotations

from dataclasses import dataclass

from symop.core.protocols.base.signature import Signature
from symop.core.protocols.modes.labels import ModeLabel


@dataclass(frozen=True)
class LabelEdit:
    pass


@dataclass(frozen=True)
class SetModeLabel(LabelEdit):
    mode_sig: Signature
    label: ModeLabel


@dataclass(frozen=True)
class DeleteModeLabel(LabelEdit):
    mode_sig: Signature
