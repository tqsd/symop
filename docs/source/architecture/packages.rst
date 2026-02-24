Packages
========

This document defines what belongs in each top-level package and serves as
the authoritative reference for package boundaries.


``core``
--------

Purpose
^^^^^^^

- Provide the foundational primitives and interfaces used throughout 
  the project.
- Serve as the root of the internal dependency graph.

Scope
^^^^^

- Mathematical primitives of the *CCR* operator framework.
- Structural identity and signature machinery.
- Typing protocols that define cross-package interfaces.
- Immutable, composable datastructures that higher layers build upon.

Primary Objects
^^^^^^^^^^^^^^^

- Mode and ladder operator primitives.
- Normally ordered monomials.
- Elementary term objects used in ket and density representations.
- Signature objects used for identity, ordering, and grouping

Protocols
^^^^^^^^^

- Signature-bearing intirfacse (exact and approximate signatures).
- Envelope-like interface required by mode operators.
- Label-like interfaces required for overlap structure and mode labeling.
- Operator interfaces (mode operators, ladder operators).
- Term interfaces (monomials and elementary ket/density terms).

Invariants and Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Core objects are immutable and hashable when appropriate.
- Monomials represent normally ordered products (creators then annihilators).
- The empty monomial represents the identity operator :math:`\mathbb{I}`.
- ``signature`` identifies exact structural identity.
- ``approx_signature`` provides a relaxed identifier for numerical merging and
  approximate equivalence (e.g. envelope tolerances).

Dependencies
^^^^^^^^^^^^
- The ``core`` package must not import any other internal package.
- External dependencies should remain minimal.
