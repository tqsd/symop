r"""Symbolic ket algebra for bosonic CCR systems.

This package implements purely symbolic manipulations of ket polynomials
of the form

.. math::

    \lvert \psi \rangle \;=\; \sum_i c_i \, \lvert M_i \rangle,

where each :math:`\lvert M_i \rangle` is a normally ordered ladder-operator
monomial acting on the vacuum. All computations are performed using
canonical commutation relations (CCR) and symbolic normal ordering.
No matrix representations or Hilbert-space truncations are constructed.

Functionality provided in this package includes:

- Construction from creators/annihilators or arbitrary operator words.
- Symbolic normal ordering of operator products.
- Linear combination and canonicalization (combining like terms).
- Scalar multiplication and ket–ket multiplication.
- Symbolic inner products

  .. math::

      \langle \phi \mid \psi \rangle,

  computed as identity coefficients of normal-ordered products.
- Application of operator words

  .. math::

      W \lvert \psi \rangle,

  including linear combinations of words.

All routines operate purely at the algebraic level using commutator
relations

.. math::

    [a_i, a_j^\dagger] = \langle m_i \mid m_j \rangle,

where the mode overlap may be non-orthogonal.

The module-level API is functional and stateless. The class
:class:`KetPoly` provides a thin immutable wrapper exposing these
operations in object-oriented form while delegating the algebra to
the underlying functional implementations.

Design principles
-----------------

- Fully symbolic CCR-based manipulation.
- Support for non-orthogonal modes via overlap factors.
- Deterministic canonical ordering using monomial signatures.
- Separation of algebraic logic and representation.

Public API
----------

- :class:`KetPoly`
"""

from .poly import KetPoly

__all__ = ["KetPoly"]
