r"""
OpPoly basics: word polynomials and adjoints
============================================

This example demonstrates the operator-polynomial layer :class:`~symop.ccr.algebra.op.poly.OpPoly`.

The important point: **OpPoly does not apply CCR rewriting or normal ordering**.
It represents a finite linear combination of *words* in ladder operators.

We show:

1. Building basic operators :math:`a`, :math:`a^\dagger`, :math:`n`.
2. Polynomial multiplication as word concatenation.
3. Combining identical words with :meth:`~symop.ccr.op.poly.OpPoly.combine_like_terms`.
4. Adjoint identities, e.g. :math:`(AB)^\dagger = B^\dagger A^\dagger`.
"""

from __future__ import annotations

import numpy as np

from symop.ccr.algebra.op.poly import OpPoly

from symop.core.operators import ModeOp
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel


def make_mode(tag: str) -> ModeOp:
    env = GaussianEnvelope(omega0=10.0, sigma=0.4, tau=0.0, phi0=0.0)
    lab = ModeLabel(
        path=PathLabel(tag), pol=PolarizationLabel.H(), envelope=env
    )
    return ModeOp(label=lab)


# %%
# 1) Basic building blocks
# ------------------------
m = make_mode("A")

A = OpPoly.a(m)
Ad = OpPoly.adag(m)
N = OpPoly.n(m)

assert len(A) == 1
assert len(Ad) == 1
assert len(N) == 1

# %%
# 2) Word concatenation (no CCR rewriting)
# ----------------------------------------
# a * adag is a different word than adag * a.
# (Composition uses @, matching your __matmul__ semantics for OpPoly @ OpPoly.)
AA_dag = A @ Ad
AdA = Ad @ A

assert AA_dag != AdA

# %%
# 3) Linear combination and combining like words
# ----------------------------------------------
# Build the same word twice and merge:
O = OpPoly.from_words([[m.ann, m.create]]) + OpPoly.from_words(
    [[m.ann, m.create]]
)
assert len(O) == 2  # not merged yet
O2 = O.combine_like_terms()
assert len(O2) == 1
assert abs(O2.terms[0].coeff - 2.0) == 0.0

# %%
# 4) Adjoint identities
# ----------------------
# (AB)^\dagger = B^\dagger A^\dagger
lhs = (A @ Ad).adjoint()
rhs = Ad.adjoint() @ A.adjoint()
assert lhs == rhs

# (A^\dagger)^\dagger = A
assert A.adjoint().adjoint() == A

# Rotated quadrature example (multi-term poly with complex coefficients)
theta = 0.123
X = OpPoly.X_theta(m, theta)
assert X.adjoint().adjoint() == X

# Scalar conjugation rule: (c A)^\dagger = c^* A^\dagger
c = 1.0 + 2.0j
assert (X * c).adjoint() == X.adjoint() * np.conjugate(c)
