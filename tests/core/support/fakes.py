from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

from symop.core.operators import ModeOp
from symop.core.protocols.modes.labels import (
    Envelope as EnvelopeProtocol,
)
from symop.core.protocols.modes.labels import (
    ModeLabel as ModeLabelProtocol,
)
from symop.core.protocols.modes.labels import (
    Path as PathProtocol,
)
from symop.core.protocols.modes.labels import (
    Polarization as PolarizationProtocol,
)
from symop.core.types.signature import Signature


@dataclass(frozen=True)
class FakeComponentLabel:
    kind: str
    name: str
    overlap_table: dict[tuple[Signature, Signature], complex] | None = None
    payload: Any = None

    def overlap(self, other: object) -> complex:
        other_sig = getattr(other, "signature", None)
        if other_sig is None:
            return 0.0 + 0.0j

        if self.overlap_table is not None:
            key = (self.signature, other_sig)
            if key in self.overlap_table:
                return self.overlap_table[key]

        if self.signature == other_sig:
            return 1.0 + 0.0j
        return 0.0 + 0.0j

    @property
    def signature(self) -> Signature:
        return (self.kind, self.name)

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Signature:
        return (
            f"{self.kind}_approx",
            self.name,
            decimals,
            ignore_global_phase,
        )

    def with_name(self, name: str) -> FakeComponentLabel:
        return replace(self, name=name)


@dataclass(frozen=True)
class FakeModeLabel:
    path: PathProtocol
    polarization: PolarizationProtocol
    envelope: EnvelopeProtocol

    def with_envelope(self, envelope: EnvelopeProtocol) -> FakeModeLabel:
        return replace(self, envelope=envelope)

    def with_path(self, path: PathProtocol) -> FakeModeLabel:
        return replace(self, path=path)

    def with_polarization(
        self,
        polarization: PolarizationProtocol,
    ) -> FakeModeLabel:
        return replace(self, polarization=polarization)

    def overlap(self, other: ModeLabelProtocol) -> complex:
        path_overlap = self.path.overlap(other.path)
        if path_overlap == 0:
            return 0.0 + 0.0j

        polarization_overlap = self.polarization.overlap(other.polarization)
        if polarization_overlap == 0:
            return 0.0 + 0.0j

        envelope_overlap = self.envelope.overlap(other.envelope)
        return path_overlap * polarization_overlap * envelope_overlap

    @property
    def signature(self) -> Signature:
        return (
            "fake_mode_label",
            self.path.signature,
            self.polarization.signature,
            self.envelope.signature,
        )

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Signature:
        return (
            "fake_mode_label_approx",
            self.path.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
            self.polarization.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
            self.envelope.approx_signature(
                decimals=decimals,
                ignore_global_phase=ignore_global_phase,
            ),
        )


def set_symmetric_overlap(
    table: dict[tuple[Signature, Signature], complex],
    left: FakeComponentLabel,
    right: FakeComponentLabel,
    value: complex,
) -> None:
    table[(left.signature, right.signature)] = value
    table[(right.signature, left.signature)] = value

def set_hermitian_overlap(
    table: dict[tuple[Signature, Signature], complex],
    left: EnvelopeProtocol,
    right: EnvelopeProtocol,
    value: complex,
) -> None:
    table[(left.signature, right.signature)] = value
    table[(right.signature, left.signature)] = value.conjugate()

def make_mode_label(
    *,
    path: str = "p",
    polarization: str = "pol",
    envelope: str = "env",
    path_table: dict[tuple[Signature, Signature], complex] | None = None,
    polarization_table: dict[tuple[Signature, Signature], complex] | None = None,
    envelope_table: dict[tuple[Signature, Signature], complex] | None = None,
) -> FakeModeLabel:
    return FakeModeLabel(
        path=cast(PathProtocol, FakeComponentLabel("path", path, path_table)),
        polarization=cast(
            PolarizationProtocol,
            FakeComponentLabel("polarization", polarization, polarization_table),
        ),
        envelope=cast(
            EnvelopeProtocol,
            FakeComponentLabel("envelope", envelope, envelope_table),
        ),
    )


def make_mode(
    *,
    path: str = "p",
    polarization: str = "pol",
    envelope: str = "env",
    user_label: str | None = None,
    display_index: int | None = None,
    path_table: dict[tuple[Signature, Signature], complex] | None = None,
    polarization_table: dict[tuple[Signature, Signature], complex] | None = None,
    envelope_table: dict[tuple[Signature, Signature], complex] | None = None,
) -> ModeOp:
    label = make_mode_label(
        path=path,
        polarization=polarization,
        envelope=envelope,
        path_table=path_table,
        polarization_table=polarization_table,
        envelope_table=envelope_table,
    )
    return ModeOp(label=cast(ModeLabelProtocol, label), user_label=user_label, display_index=display_index)
