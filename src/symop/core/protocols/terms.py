from __future__ import annotations

from typing import Protocol, runtime_checkable

from .monomials import MonomialProto
from .operators import ModeOpProto
from .signature import HasSignature


@runtime_checkable
class KetTermProto(HasSignature, Protocol):
    coeff: complex
    monomial: MonomialProto

    @staticmethod
    def identity() -> KetTermProto: ...

    def adjoint(self) -> KetTermProto: ...
    def scaled(self, s: complex) -> KetTermProto: ...

    @property
    def is_creator_only(self) -> bool: ...

    @property
    def is_annihilator_only(self) -> bool: ...

    @property
    def is_identity(self) -> bool: ...

    @property
    def creation_count(self) -> int: ...

    @property
    def annihilation_count(self) -> int: ...

    @property
    def total_degree(self) -> int: ...

    @property
    def mode_ops(self) -> tuple[ModeOpProto, ...]: ...


@runtime_checkable
class DensityTermProto(HasSignature, Protocol):
    coeff: complex
    left: MonomialProto
    right: MonomialProto

    @staticmethod
    def identity() -> DensityTermProto: ...

    def adjoint(self) -> DensityTermProto: ...
    def scaled(self, s: complex) -> DensityTermProto: ...

    @property
    def is_creator_only_left(self) -> bool: ...

    @property
    def is_creator_only_right(self) -> bool: ...

    @property
    def is_creator_only(self) -> bool: ...

    @property
    def is_annihilator_only_left(self) -> bool: ...

    @property
    def is_annihilator_only_right(self) -> bool: ...

    @property
    def is_annihilator_only(self) -> bool: ...

    @property
    def is_diagonal_in_monomials(self) -> bool: ...

    @property
    def creation_count_left(self) -> int: ...

    @property
    def creation_count_right(self) -> int: ...

    @property
    def annihilation_count_left(self) -> int: ...

    @property
    def annihilation_count_right(self) -> int: ...

    @property
    def mode_ops_left(self) -> tuple[ModeOpProto, ...]: ...

    @property
    def mode_ops_right(self) -> tuple[ModeOpProto, ...]: ...
