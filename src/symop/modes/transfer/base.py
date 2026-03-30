r"""Base classes for spectral transfer functions.

This module provides small implementation-oriented base classes for
transfer functions used in the modes package.

The design separates two concerns:

1. Generic transfer identity and evaluation contract.
2. Optional analytic capabilities layered on top of the generic transfer.

The primary class exported here is :class:`TransferBase`, which provides
shared logic for stable signatures and approximate signatures. Concrete
transfer functions typically inherit from this class and implement only
their spectral evaluation rule.

Notes
-----
The base class is intentionally lightweight. It does not impose any
particular analytic formalism and does not assume that a transfer
supports closed-form application to specific envelope families.

For dataclass-based transfers, the default implementation derives
signature parameters from dataclass fields in declaration order. This
keeps concrete transfer implementations compact and uniform.

Subclasses may override :meth:`_signature_params` when the identity of a
transfer should differ from its raw dataclass field tuple.

Examples
--------
A simple constant phase transfer can inherit from :class:`TransferBase`
and only implement three things:

- the class attribute ``_signature_tag``
- the spectral call operator ``__call__``
- optional constructor-time validation

Then the exact and approximate signatures are inherited automatically.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import Any, ClassVar

from symop.core.types.arrays import FloatArray, RCArray
from symop.core.types.signature import Signature


class TransferBase(ABC):
    r"""Common implementation base for spectral transfer functions.

    This base class centralizes signature generation for concrete
    transfer-function implementations.

    A transfer function :math:`H(\omega)` is identified by a stable
    signature used for caching, comparison, and expression reuse. In many
    cases the identity is fully determined by the transfer type together
    with a tuple of scalar parameters. For frozen dataclass-based
    transfers, this class derives those parameters automatically.

    Subclasses are expected to provide:

    - a class attribute :attr:`_signature_tag`
    - an implementation of :meth:`__call__`

    They may optionally override :meth:`_signature_params` if their
    logical identity differs from their raw stored parameters.

    Notes
    -----
    This class does not enforce a specific numerical representation for
    the returned samples beyond what is required by the public transfer
    protocol used elsewhere in the package.

    The approximate signature is constructed by recursively rounding
    floating-point values contained in the exact parameter structure.

    Attributes
    ----------
    _signature_tag : ClassVar[str]
        Stable symbolic tag identifying the concrete transfer type.

    """

    _signature_tag: ClassVar[str]

    @property
    def signature(self) -> Signature:
        """Return the exact stable signature of the transfer.

        The exact signature is a tuple beginning with the subclass tag
        followed by the exact signature parameter structure.

        Returns
        -------
        Signature
            Exact signature tuple used for caching and structural
            comparison.

        Notes
        -----
        The default implementation uses :meth:`_signature_params` with
        ``ignore_global_phase=False``.

        """
        return (self._signature_tag, *self._signature_params())

    def approx_signature(
        self,
        *,
        decimals: int = 12,
        ignore_global_phase: bool = False,
    ) -> Signature:
        """Return an approximate signature with rounded numeric values.

        This method is useful for grouping or caching transfer functions
        whose parameters differ only by negligible floating-point noise.

        Parameters
        ----------
        decimals
            Number of decimal places used when rounding floating-point
            values.
        ignore_global_phase
            Whether subclasses should omit or normalize parameters
            representing a physically irrelevant global phase.

        Returns
        -------
        Signature
            Approximate signature tuple with recursively rounded numeric
            parameters.

        Notes
        -----
        The exact meaning of ``ignore_global_phase`` depends on the
        subclass. For transfers without a global-phase degree of freedom,
        the flag usually has no effect.

        """
        params = self._signature_params(ignore_global_phase=ignore_global_phase)
        rounded = self._round_for_signature(params, decimals=decimals)
        return (f"{self._signature_tag}_approx", *rounded)

    def _signature_params(
        self,
        *,
        ignore_global_phase: bool = False,
    ) -> tuple[Any, ...]:
        """Return the logical parameter tuple used in signatures.

        Parameters
        ----------
        ignore_global_phase
            Whether a physically irrelevant global phase should be
            excluded or canonicalized by subclasses that support that
            behavior.

        Returns
        -------
        tuple[Any, ...]
            Parameter tuple describing the transfer.

        Notes
        -----
        The default implementation derives parameters from dataclass
        fields in declaration order. This is appropriate for most simple
        immutable transfer classes.

        Subclasses can override this method if their signature should
        depend on a transformed, reduced, or reordered set of values.

        """
        del ignore_global_phase
        return self._dataclass_field_values()

    def _dataclass_field_values(self) -> tuple[Any, ...]:
        """Return dataclass field values in declaration order.

        Returns
        -------
        tuple[Any, ...]
            Tuple of stored dataclass field values.

        Raises
        ------
        TypeError
            If the instance is not a dataclass instance and the subclass
            did not override :meth:`_signature_params`.

        Notes
        -----
        This helper supports the common case where transfer functions are
        implemented as frozen dataclasses with scalar parameters.

        """
        if not is_dataclass(self):
            raise TypeError(
                f"{type(self).__name__} must be a dataclass instance or "
                "override _signature_params()."
            )
        return tuple(getattr(self, f.name) for f in fields(self))

    @classmethod
    def _round_for_signature(
        cls,
        value: Any,
        *,
        decimals: int,
    ) -> Any:
        r"""Recursively round numeric values for approximate signatures.

        Parameters
        ----------
        value
            Arbitrary nested value appearing inside a signature
            parameter structure.
        decimals
            Number of decimal places used for rounding.

        Returns
        -------
        Any
            Value with floating-point leaves rounded recursively.

        Notes
        -----
        The transformation preserves the container shape for tuples and
        lists. Complex numbers are rounded componentwise as

        .. math::

            z = x + i y
            \mapsto
            \operatorname{round}(x) + i \operatorname{round}(y).

        Booleans are preserved unchanged.

        """
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return round(value, decimals)
        if isinstance(value, complex):
            return complex(
                round(value.real, decimals),
                round(value.imag, decimals),
            )
        if isinstance(value, tuple):
            return tuple(cls._round_for_signature(v, decimals=decimals) for v in value)
        if isinstance(value, list):
            return [cls._round_for_signature(v, decimals=decimals) for v in value]
        return value

    @abstractmethod
    def __call__(self, w: FloatArray) -> RCArray:
        r"""Evaluate the transfer function on an angular-frequency grid.

        Parameters
        ----------
        w
            Angular-frequency grid :math:`\omega`.

        Returns
        -------
        RCArray
            Complex samples of :math:`H(\omega)`.

        Notes
        -----
        Concrete subclasses should return an array-compatible complex
        representation matching the public transfer protocol used by the
        package.

        """
        raise NotImplementedError
