r"""Text rendering registry for symbolic objects.

This package provides text-dispatch implementations for various symbolic
objects in the symop framework, including states, operators, terms,
labels, and envelopes.

Importing this module registers all available text renderers via the
central ``text`` dispatcher, enabling consistent string representations
across the codebase.

Notes
-----
- Modules are imported for their side effects (registration).
- Rendering is type-driven via the dispatch system.
- Output is intended for debugging, logging, and lightweight inspection.

"""
