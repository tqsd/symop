Architecture
============

This project implements a CCR framework with support for continuous 
variable states (Gaussian), where the individual modes may be
non-orthogonal. 

The project has a strict structure to ensure maintainability,
exstensibility and long-term stability. Foundational abstractions
are defined in lower layers and reused by higher layers through
well-defined interfaces. Each layer has a clearly defined responsibility
and a restricted dependency surface.

This document defines the architectural principles of the project.
Detailed responsibilities of individual packages are described in
:doc:`packages`.

Dependency Principle
--------------------

The project follows a unidirectional dependency model.

- Dependencies must form an acyclic graph.
- Lower layers must not depend on higher layers.
- Foundational packages define primitives and protocols.
- Higher-level packages build structured models on top of these primitives.

Foundational Packages
---------------------


Core Layer
^^^^^^^^^^

The ``core`` package forms the root of the dependency graph.

Its responsibilities include:

- Defining mathematical primitives of the CCR algebra.
- Representing operator atoms (e.g. mode operators and ladder operators).
- Defining monomials and elementary term structures.
- Providing signature and structural identity machinery.
- Defining typing protocols used across the project.
- Enforcing immutability and structural consistency accross the project.

The ``core`` packages does not import any other internal package.

All other internal packages are required to depend on ``core`` rather than 
the revers. This constraint is enforced through static analysis and linting
contracts.

Modes Layer
^^^^^^^^^^^

The ``modes`` package forms the second foundational layer above ``core``.
It provides the analytic and numerical realizations of abstract CCR modes
defined in ``core``.

While ``core`` defines algebraic primitives (mode operators, ladder operatrors
monomials, signatures), it is intentionally agnostic to any specific functional
representation of modes. The ``modes`` layer assigns these abstract modes a concrete
continuous-variable structure.


Its architectural role is to:

- Represent complex time-domain envelopes.
- Provide frequency-domain representations consistent with the project
  Fourier convenvion.
- Define overlap structure between (possibly non-orthogonal) modes.
- Implement analytic envelope models (e.g. Gaussian envelopes).
- Define transfer functions acting multiplicatively in frequency space.
- Provide controlled plotting and inspection utilities.

This layer introduces functional structure without modifying the underlying
CCR algebra.

CCR Layer
^^^^^^^^^

The ``ccr`` package builds on top of ``core`` (and uses ``modes`` where overlap)
and continuous-variable structure are required) to provide the symbolic
operator algebra for canonical commutation relations.

Its responsibility is to represent and manipulate CCR expression in a
purely symbolic form, without constructing matrices.

Architectural role
~~~~~~~~~~~~~~~~~~

The ``ccr`` layer providec:

- Word- and polynomial-based symbolic objects:
  operator polynomials (``OpPoly``), ket polynomials (``KetPoly``),
  and density polynomials (``DensityPoly``).
- Linear-algebra-like operations at the symbolic level:
  addition, scalar scaling, adjoints, composition, and left/right actions.
- Canonicalization utilities based on structural signatures:
  combining like terms and stable hashing for caching and deduplication.
- Domain-specific operations for densities where defined:
  trace, purity, partial trace, and Hilbert-Schmidt inner products.

Design constraints
~~~~~~~~~~~~~~~~~~

- No numeriacl state vectors or matrices are created in this layer.
- No commutations-based rewriting is applied implicitly.
  Normal ordering and CCR rewrite rules are modeled explicitly as
  separate transformations to keep semantics predictable and testable.
- All operations are linear and signature-driven unless stated otherwise.



Design Goals
------------

The architecture is designed to achieve the following goals:

- Clear separation of concerns.
- Strict layering and acyclic dependencies.
- Protocol-based interfaces between layers.
- Static typing in foundational components.
- Isolation of mathematical structure from physical modeling.

Public API Strategy
-------------------

The project exposes a clearly defined public import surface.
