"""Symbolic CCR polynomial framework.

This package provides symbolic representations of quantum states and
operators based on polynomials in bosonic ladder operators obeying the
canonical commutation relations (CCR).

The polynomial layer acts as the core symbolic manipulation engine used
throughout the library. It supports rewriting ladder-operator expressions
under linear optical transformations, channel constructions, and other
device-level operations.

Subpackages
-----------
state
    State containers for polynomial representations (e.g. density states).
channels
    Quantum channels implemented via symbolic polynomial rewrites.
rewrites
    Core substitution and transformation utilities for ladder operators.
protocols
    Protocol interfaces defining the polynomial object contracts.

Notes
-----
This layer operates purely symbolically. Transformations are implemented
as algebraic rewrites of ladder operators and polynomial terms rather
than through explicit matrix representations of Hilbert spaces.

"""
