r"""
DensityPoly right action: rho @ OpPoly
======================================

This example demonstrates the symbolic right action of operator polynomials
on a density polynomial:

.. math::

    (rho @ O) = rho O.

This uses :meth:`~symop.ccr.algebra.density.poly.DensityPoly.__matmul__` and shows:

1. Identity action: ``rho @ I == rho``.
2. Linearity in ``O``.
3. The effect on monomials: right words are appended symbolically.
"""

from __future__ import annotations


from symop.ccr.algebra.density import DensityPoly
from symop.ccr.algebra.op import OpPoly

from symop.core.monomial import Monomial
from symop.core.terms import DensityTerm

from symop.core.operators import ModeOp
from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel


def make_mode(tag: str, *, tau: float = 0.0) -> ModeOp:
    env = GaussianEnvelope(omega0=10.0, sigma=0.4, tau=float(tau), phi0=0.0)
    lab = ModeLabel(
        path=PathLabel(tag), pol=PolarizationLabel.H(), envelope=env
    )
    return ModeOp(label=lab)


# %%
# Setup: a simple density term
# -----------------------------
m = make_mode("A")

# rho = 2 * |adag><a|
L = Monomial(creators=(m.create,), annihilators=())
R = Monomial(creators=(), annihilators=(m.ann,))
rho = DensityPoly((DensityTerm(coeff=2.0 + 0.0j, left=L, right=R),))

# %%
# 1) Identity action
# -------------------
I = OpPoly.identity()
assert rho @ I == rho

# %%
# 2) Linearity in the operator polynomial
# ---------------------------------------
O1 = OpPoly.from_words([[m.ann]])
O2 = OpPoly.from_words([[m.create]]) * (0.5 + 0.0j)

left = (rho @ (O1 + O2)).combine_like_terms()
right = ((rho @ O1) + (rho @ O2)).combine_like_terms()
assert left == right

# %%
# 3) Inspect the structural effect: words are appended on the right
# -----------------------------------------------------------------
# After applying @ a, the right monomial should gain an extra annihilator.
out = (rho @ O1).combine_like_terms()
assert len(out.terms) == 1
dt = out.terms[0]

# Left should be unchanged structurally in this right action.
assert dt.left.signature == L.signature

# Right should now represent |...>< a a | in the sense of appended word.
# Since Monomial stores annihilators as a tuple, check the tail.
assert dt.right.annihilators[-1].signature == m.ann.signature

# %%
# 4) A quick numeric-like check: trace is linear under right multiplication by identity
# ------------------------------------------------------------------------------------
assert abs((rho @ I).trace() - rho.trace()) <= 1e-14
