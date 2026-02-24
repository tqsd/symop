from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from .signature import HasSignature


@runtime_checkable
class LabelProto(HasSignature, Protocol):
    def overlap(self, other: Self) -> complex: ...


@runtime_checkable
class PathProto(HasSignature, Protocol):
    def overlap(self, other: Self) -> complex: ...


@runtime_checkable
class PolarizationProto(HasSignature, Protocol):
    def overlap(self, other: Self) -> complex: ...


@runtime_checkable
class ModeLabelLike(LabelProto, Protocol):
    path: PathProto
    pol: PolarizationProto

    def with_pol(self, pol: LabelProto) -> ModeLabelLike: ...
    def with_path(self, path: PathProto) -> ModeLabelLike: ...
