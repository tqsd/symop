r"""Symbolic density-operator algebra for bosonic CCR systems.

This package implements purely symbolic manipulations of density
polynomials of the form

.. math::

    \rho \;=\; \sum_i c_i \, \lvert L_i \rangle \langle R_i \rvert,

where each term consists of a complex coefficient and a pair of
normally ordered ladder-operator monomials. All computations are
performed using canonical commutation relations (CCR) and symbolic
normal ordering. No matrix representations or finite-dimensional
truncations are constructed.

Functionality provided in this package includes:

- Construction of pure states

  .. math::

      \rho = \lvert \psi \rangle \langle \psi \rvert.

- Trace evaluation and trace normalization.
- Hilbert–Schmidt inner products

  .. math::

      \langle A, B \rangle = \mathrm{Tr}(A^\dagger B),

  and purity

  .. math::

      \mathrm{Tr}(\rho^2).

- Left and right application of operator words.
- Partial trace over selected modes.
- Canonicalization by combining like terms.
- Symbolic expansion of monomial and word products.

The module-level API is functional and stateless. The class
:class:`DensityPoly` provides a thin immutable wrapper exposing these
operations in object-oriented form while delegating the actual algebra
to the underlying functional routines.

Design principles
-----------------

- Fully symbolic CCR-based manipulation.
- Support for non-orthogonal modes via overlap factors.
- Deterministic canonical ordering using monomial signatures.
- Strict separation between algebraic logic and representation.

Public API
----------

- :class:`DensityPoly`
"""

from .poly import DensityPoly

__all__ = ["DensityPoly"]
