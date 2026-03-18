"""LaTeX rendering dispatch layer for symop objects.

This package defines LaTeX renderers for the symbolic objects used in
the ``symop`` framework. Rendering is implemented via a central
dispatcher (``symop.viz._dispatch.latex``), with individual modules
registering handlers for specific object types.

The rendering pipeline is layered:

- low-level primitives (operators, labels, envelopes)
- algebraic structures (monomials, terms)
- polynomial objects (ket, density, operator polynomials)
- state-level objects

Each module in this package contributes renderer registrations for a
specific abstraction level, ensuring a compositional and extensible
LaTeX representation across the entire symbolic stack.

Notes
-----
Importing this package ensures that all LaTeX renderers are registered
with the global dispatcher.

"""
