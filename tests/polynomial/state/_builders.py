# tests/polynomial/state/_builders.py

from __future__ import annotations

from symop.core.operators import ModeOp
from symop.modes.labels import Path, Polarization, ModeLabel
from symop.modes.envelopes import GaussianEnvelope
from symop.polynomial.state.ket import KetPolyState


def make_test_mode(name: str, path: str) -> ModeOp:
    polarization = Polarization.H()
    mode_path = Path(path)
    envelope = GaussianEnvelope(1, 1, 0, 0)
    mode_label = ModeLabel(path=mode_path, polarization=polarization, envelope=envelope)
    mode_op = ModeOp(label=mode_label, user_label=name)
    return mode_op

def make_single_photon_ket(name: str = "m0", path: str = "p0") -> KetPolyState:
    mode = make_test_mode(name=name, path=path)
    return KetPolyState.from_creators([mode.cre])


def make_two_mode_ket() -> tuple[KetPolyState, object, object]:
    mode_a = make_test_mode(name="a", path="p0")
    mode_b = make_test_mode(name="b", path="p1")
    state = KetPolyState.from_creators([mode_a.cre, mode_b.cre])
    return state, mode_a, mode_b
