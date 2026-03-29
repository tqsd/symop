r"""Beamsplitter device models.

This package contains semantic models of beamsplitter devices used in
linear optical systems.

Beamsplitters are passive two-mode transformations that mix input modes
according to a unitary transformation. At the semantic level, these
models describe:

- how input paths are paired,
- how output paths are assigned,
- and how device parameters (e.g., reflectivity, phase) are encoded.

The actual transformation of quantum states is performed by
representation-specific kernels. The models defined here only construct
semantic device actions and label edits required for execution.

Typical devices in this package include:

- balanced and unbalanced beamsplitters,
- parametrized two-mode couplers,
- components used to build interferometers such as Mach–Zehnder setups.

Notes
-----
These models are backend-agnostic and operate purely at the symbolic or
semantic planning level.

"""

from .beamsplitter import BeamSplitter

__all__ = ["BeamSplitter"]
