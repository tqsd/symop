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
