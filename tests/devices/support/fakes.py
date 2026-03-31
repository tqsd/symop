from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np

from symop.devices.action import DeviceAction
from symop.devices.measurement.outcomes import MeasurementOutcome
from symop.devices.ports import PortSpec
from symop.devices.types.device_kind import DeviceKind


@dataclass(frozen=True)
class FakePath:
    name: str

    @property
    def signature(self) -> tuple[str, str]:
        return ("fake_path", self.name)

    def overlap(self, other: object) -> complex:
        if not isinstance(other, FakePath):
            return 0.0 + 0.0j
        return 1.0 + 0.0j if self.name == other.name else 0.0 + 0.0j

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> tuple[str, str, int, bool]:
        return ("fake_path_approx", self.name, decimals, ignore_global_phase)


@dataclass
class FakeState:
    rep_kind: str = "poly"
    state_kind: str = "ket"


@dataclass
class FakeEditableState:
    rep_kind: str = "poly"
    state_kind: str = "ket"
    mode_labels: Mapping[object, object] = field(default_factory=dict)
    modes: tuple[object, ...] = ()
    applied_edits: list[object] = field(default_factory=list)

    def label_for_mode(self, mode_sig: object) -> object:
        return self.mode_labels[mode_sig]

    def apply_label_edits(self, edits: Sequence[object]) -> FakeEditableState:
        return FakeEditableState(
            rep_kind=self.rep_kind,
            state_kind=self.state_kind,
            mode_labels=self.mode_labels,
            modes=self.modes,
            applied_edits=self.applied_edits + list(edits),
        )

    def modes_on_path(self, path: object) -> tuple[object, ...]:
        return tuple(
            mode
            for mode in self.modes
            if getattr(mode.label, "path", None) == path
        )


@dataclass(frozen=True)
class FakeTimeFrequencyEnvelope:
    formalism: ClassVar[str] = "generic"

    name: str
    tau: float = 0.0
    phi: float = 0.0
    sigma: float = 1.0
    omega0: float = 0.0
    omega_sigma: float = 1.0

    @property
    def signature(self) -> tuple[str, str, float, float, float, float, float]:
        return (
            "fake_time_frequency_envelope",
            self.name,
            self.tau,
            self.phi,
            self.sigma,
            self.omega0,
            self.omega_sigma,
        )

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> tuple[str, str, float, float | str, float, float, float, int, bool]:
        phi = 0.0 if ignore_global_phase else round(self.phi, decimals)
        return (
            "fake_time_frequency_envelope_approx",
            self.name,
            round(self.tau, decimals),
            phi,
            round(self.sigma, decimals),
            round(self.omega0, decimals),
            round(self.omega_sigma, decimals),
            decimals,
            ignore_global_phase,
        )

    def overlap(self, other: object) -> complex:
        if not isinstance(other, FakeTimeFrequencyEnvelope):
            return 0.0 + 0.0j
        return 1.0 + 0.0j if self.signature == other.signature else 0.0 + 0.0j

    def time_eval(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        return np.exp(-((t_arr - self.tau) ** 2) / (2.0 * self.sigma ** 2)) * np.exp(
            1j * self.phi
        )

    def freq_eval(self, w: np.ndarray) -> np.ndarray:
        w_arr = np.asarray(w, dtype=float)
        return np.exp(
            -((w_arr - self.omega0) ** 2) / (2.0 * self.omega_sigma ** 2)
        ) * np.exp(1j * self.phi)

    def delayed(self, dt: float) -> FakeTimeFrequencyEnvelope:
        return replace(self, tau=self.tau + dt)

    def phased(self, dphi: float) -> FakeTimeFrequencyEnvelope:
        return replace(self, phi=self.phi + dphi)

    def center_and_scale(self) -> tuple[float, float]:
        return (self.tau, self.sigma)


@dataclass
class FakeDensityState:
    rep_kind: str = "poly"
    state_kind: str = "density"


@dataclass
class FakeStateWithToDensity:
    rep_kind: str = "poly"
    state_kind: str = "ket"
    density_state: object | None = None
    to_density_calls: int = 0

    def to_density(self) -> object:
        self.to_density_calls += 1
        if self.density_state is None:
            raise RuntimeError("density_state not configured")
        return self.density_state


@dataclass
class RecordingKernel:
    result: object
    calls: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, *, state: object, action: object, ctx: object) -> object:
        self.calls.append(
            {
                "state": state,
                "action": action,
                "ctx": ctx,
            }
        )
        return self.result


@dataclass
class FakeDevice:
    kind: DeviceKind
    port_specs: Sequence[PortSpec]
    action: DeviceAction
    plan_calls: list[dict[str, object]] = field(default_factory=list)

    def plan(
        self,
        *,
        state: object,
        ports: Mapping[str, object],
        selection: object | None,
        ctx: object,
    ) -> DeviceAction:
        self.plan_calls.append(
            {
                "state": state,
                "ports": ports,
                "selection": selection,
                "ctx": ctx,
            }
        )
        return self.action


@dataclass(frozen=True)
class FakeMeasurementAction:
    intent: str
    target: object = None
    outcome: object | None = None
    destructive: bool = False


@dataclass
class FakeMeasurementDevice:
    kind: DeviceKind
    port_specs: Sequence[PortSpec]
    observe_action: object
    detect_action: object
    postselect_action: object
    observe_calls: list[dict[str, object]] = field(default_factory=list)
    detect_calls: list[dict[str, object]] = field(default_factory=list)
    postselect_calls: list[dict[str, object]] = field(default_factory=list)

    def plan_observe(
        self,
        *,
        state: object,
        ports: Mapping[str, object],
        selection: object | None = None,
        ctx: object | None = None,
    ) -> object:
        self.observe_calls.append(
            {
                "state": state,
                "ports": ports,
                "selection": selection,
                "ctx": ctx,
            }
        )
        return self.observe_action

    def plan_detect(
        self,
        *,
        state: object,
        ports: Mapping[str, object],
        selection: object | None = None,
        ctx: object | None = None,
    ) -> object:
        self.detect_calls.append(
            {
                "state": state,
                "ports": ports,
                "selection": selection,
                "ctx": ctx,
            }
        )
        return self.detect_action

    def plan_postselect(
        self,
        *,
        state: object,
        outcome: MeasurementOutcome,
        ports: Mapping[str, object],
        selection: object | None = None,
        ctx: object | None = None,
    ) -> object:
        self.postselect_calls.append(
            {
                "state": state,
                "outcome": outcome,
                "ports": ports,
                "selection": selection,
                "ctx": ctx,
            }
        )
        return self.postselect_action


@dataclass
class FakeMeasurementRuntime:
    observe_result: object = None
    detect_result: object = None
    postselect_result: object = None
    observe_calls: list[dict[str, object]] = field(default_factory=list)
    detect_calls: list[dict[str, object]] = field(default_factory=list)
    postselect_calls: list[dict[str, object]] = field(default_factory=list)

    def observe(
        self,
        *,
        device: object,
        state: object,
        ports: Mapping[str, object],
        selection: object | None = None,
        ctx: object | None = None,
    ) -> object:
        self.observe_calls.append(
            {
                "device": device,
                "state": state,
                "ports": ports,
                "selection": selection,
                "ctx": ctx,
            }
        )
        return self.observe_result

    def detect(
        self,
        *,
        device: object,
        state: object,
        ports: Mapping[str, object],
        selection: object | None = None,
        ctx: object | None = None,
    ) -> object:
        self.detect_calls.append(
            {
                "device": device,
                "state": state,
                "ports": ports,
                "selection": selection,
                "ctx": ctx,
            }
        )
        return self.detect_result

    def postselect(
        self,
        *,
        device: object,
        state: object,
        outcome: MeasurementOutcome,
        ports: Mapping[str, object],
        selection: object | None = None,
        ctx: object | None = None,
    ) -> object:
        self.postselect_calls.append(
            {
                "device": device,
                "state": state,
                "outcome": outcome,
                "ports": ports,
                "selection": selection,
                "ctx": ctx,
            }
        )
        return self.postselect_result
