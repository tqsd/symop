Normal ordering, expectation values, and trace
==============================================

This tutorial explains how the CCR layer rewrites operator expressions into
normally ordered form and how scalar quantities such as overlaps, expectation
values, traces, and partial traces are extracted symbolically.

The key idea is that Symop does not evaluate matrices directly. Instead, it
rewrites ladder-operator words using commutation relations and keeps the result
as symbolic sums of normally ordered monomials.

Background
----------

In the CCR layer, ladder operators satisfy commutation relations of the form

.. math::

   [a_i, a_j^\dagger] = \langle m_i \mid m_j \rangle,

where the overlap on the right-hand side may be nontrivial when the underlying
modes are not orthogonal.

A general operator word is rewritten into a sum of normally ordered monomials,

.. math::

   W = o_1 o_2 \cdots o_L
   \;\longmapsto\;
   \sum_k c_k M_k,

where each monomial :math:`M_k` stores all creators on the left and all
annihilators on the right.

This symbolic representation is the basis for:

- ket construction and ket multiplication
- overlaps and inner products
- density operators and traces
- symbolic operator actions

Setup
-----

.. jupyter-execute::

   import symop.viz as viz

   from symop.core.operators import ModeOp
   from symop.modes.labels import ModeLabel
   from symop.modes.labels.path import Path
   from symop.modes.labels.polarization import Polarization
   from symop.modes.envelopes import GaussianEnvelope

   from symop.ccr.algebra.ket.poly import KetPoly
   from symop.ccr.algebra.op.poly import OpPoly
   from symop.ccr.algebra.density.poly import DensityPoly

   env = GaussianEnvelope(omega0=1.0, sigma=1.0, tau=0.0)

   mode = ModeOp(
       label=ModeLabel(
           path=Path("A"),
           polarization=Polarization.H(),
           envelope=env,
       ),
       user_label="a",
   )

Normal ordering example: :math:`a a^\dagger`
--------------------------------------------

A simple example is the word

.. math::

   a a^\dagger.

From the commutation relation we expect

.. math::

   a a^\dagger = a^\dagger a + 1.

.. jupyter-execute::

   expr = KetPoly.from_word(ops=(mode.ann, mode.cre))

.. jupyter-execute::

   viz.display(expr)

The result contains:

- an identity contribution (scalar term)
- the reordered monomial :math:`a^\dagger a`

This illustrates the core mechanism: non-normal-ordered expressions expand into
normally ordered terms plus contraction contributions.

Algorithmic picture
-------------------

The normal-ordering algorithm proceeds left-to-right.

For each operator:

- annihilators are appended to the annihilation block
- creators are appended to the creation block
- each creator also contracts with existing annihilators

Each contraction contributes a scalar

.. math::

   \langle m_i \mid m_j \rangle.

This produces the exact normal-ordered expansion without constructing matrices.

Step-by-step normal ordering
----------------------------

It is useful to see explicitly how the symbolic rewrite proceeds on a small
word. Consider

.. math::

   a_a \, a_b^\dagger.

There are two cases:

- if the modes are orthogonal, the commutator vanishes and the word is only
  reordered
- if the modes overlap, an additional scalar contraction term appears

For orthogonal modes, the result is just the reordered monomial:

.. jupyter-execute::

   mode_b = ModeOp(
       label=ModeLabel(
           path=Path("B"),
           polarization=Polarization.H(),
           envelope=env,
       ),
       user_label="b",
   )

   expr_orth = KetPoly.from_word(ops=(mode.ann, mode_b.cre))

.. jupyter-execute::

   viz.display(expr_orth)

Since the modes are distinct, there is no contraction and the output contains
only the normally ordered term.

For the same mode, the contraction is nonzero:

.. jupyter-execute::

   expr_same = KetPoly.from_word(ops=(mode.ann, mode.cre))

.. jupyter-execute::

   viz.display(expr_same)

This corresponds to

.. math::

   a_a a_a^\dagger = a_a^\dagger a_a + 1.

Algorithmically, the rewrite can be read as:

1. start from the empty monomial
2. append :math:`a_a` to the annihilator list
3. insert :math:`a_a^\dagger`
4. keep the reordered term :math:`a_a^\dagger a_a`
5. contract once with the existing annihilator to produce the scalar term

In other words, normal ordering produces both:

- the pass-through reordered monomial
- every allowed single contraction contribution

Two-mode expectation values
---------------------------

Expectation values become especially informative when two modes are involved.
Consider the number operator in mode :math:`a` acting on a one-particle state
in mode :math:`b`:

.. math::

   \hat n_a = a_a^\dagger a_a,
   \qquad
   |\psi_b\rangle = a_b^\dagger.

If the two modes are orthogonal, the expectation value should vanish:

.. math::

   \langle \psi_b | \hat n_a | \psi_b \rangle = 0.

Construct the state and operator symbolically:

.. jupyter-execute::

   psi_b = KetPoly.from_ops(creators=(mode_b.cre,))
   n_a = OpPoly.n(mode)
   psi_b_out = n_a @ psi_b

.. jupyter-execute::

   viz.display(psi_b)

.. jupyter-execute::

   viz.display(psi_b_out)

Now compute the expectation value:

.. jupyter-execute::

   psi_b.inner(psi_b_out)

The result is zero because no identity contribution survives after symbolic
normal ordering: the operator counts excitations in mode :math:`a`, but the
state occupies mode :math:`b`.

For comparison, the number operator in the matching mode gives a nonzero value:

.. jupyter-execute::

   psi_a = KetPoly.from_ops(creators=(mode.cre,))
   psi_a_out = n_a @ psi_a

.. jupyter-execute::

   viz.display(psi_a_out)

.. jupyter-execute::

   psi_a.inner(psi_a_out)

This shows the symbolic logic of expectation-value evaluation:

1. apply the operator polynomial to the ket
2. expand the result into normally ordered monomials
3. form the symbolic inner product with the original ket
4. extract the identity contribution

Already normal-ordered input
----------------------------

If the input is already normal ordered, no contraction occurs.

.. jupyter-execute::

   ordered = KetPoly.from_word(ops=(mode.cre, mode.ann))

.. jupyter-execute::

   viz.display(ordered)

Constructing from operators vs words
------------------------------------

- ``KetPoly.from_ops`` assumes normal ordering
- ``KetPoly.from_word`` performs CCR rewriting

.. jupyter-execute::

   psi_ops = KetPoly.from_ops(creators=(mode.cre,), annihilators=(mode.ann,))
   psi_word = KetPoly.from_word(ops=(mode.cre, mode.ann))

.. jupyter-execute::

   viz.display(psi_ops)

.. jupyter-execute::

   viz.display(psi_word)

Scalar overlaps
---------------

Scalar quantities come from the identity monomial.

.. jupyter-execute::

   psi = KetPoly.from_ops(creators=(mode.cre,))
   psi.inner(psi)

Only the identity contribution survives.

Operator action on kets
-----------------------

Operator polynomials act using ``@``.

.. jupyter-execute::

   a = OpPoly.a(mode)
   adag = OpPoly.adag(mode)

   vacuum = KetPoly.from_ops(coeff=1.0)
   created = adag @ vacuum

.. jupyter-execute::

   viz.display(created)

.. jupyter-execute::

   acted = a @ created

.. jupyter-execute::

   viz.display(acted)

This reflects:

- creation → append operator
- annihilation → contraction + reordered term

Expectation values
------------------

Expectation values are computed via symbolic contraction.

.. math::

   \langle \psi | \hat n | \psi \rangle,
   \qquad
   \hat n = a^\dagger a.

For a single-photon state in the matching mode, the expected value is 1.

.. jupyter-execute::

   psi = KetPoly.from_ops(creators=(mode.cre,))
   n_op = OpPoly.n(mode)
   psi_out = n_op @ psi

.. jupyter-execute::

   viz.display(psi)

.. jupyter-execute::

   viz.display(psi_out)

.. jupyter-execute::

   psi.inner(psi_out)

The scalar result comes from identity extraction after normal ordering.

From kets to density operators
------------------------------

A pure density operator is

.. math::

   \rho = |\psi\rangle\langle\psi|.

.. jupyter-execute::

   rho = DensityPoly.pure(created)

.. jupyter-execute::

   viz.display(rho)

Trace
-----

The trace is

.. math::

   \mathrm{Tr}(\rho)
   =
   \sum_i c_i \langle R_i \mid L_i \rangle.

.. jupyter-execute::

   rho.trace()

Purity
------

.. math::

   \mathrm{Tr}(\rho^2)

.. jupyter-execute::

   rho.purity()

Left and right operator action
------------------------------

.. math::

   \rho \mapsto W\rho,
   \qquad
   \rho \mapsto \rho W.

Right action is implemented internally via daggered reversed words.

.. jupyter-execute::

   rho_left = rho.apply_left((mode.cre,))

.. jupyter-execute::

   viz.display(rho_left)

.. jupyter-execute::

   rho_right = rho.apply_right((mode.ann,))

.. jupyter-execute::

   viz.display(rho_right)

Partial trace
-------------

Partial trace contracts a subset of modes.

.. math::

   \mathrm{Tr}_T(\rho)
   =
   \sum_i c_i \langle R_i^T \mid L_i^T \rangle
   |L_i^K\rangle\langle R_i^K|.

.. jupyter-execute::

   mode_b = ModeOp(
       label=ModeLabel(
           path=Path("B"),
           polarization=Polarization.H(),
           envelope=env,
       ),
       user_label="b",
   )

   psi = KetPoly.from_ops(creators=(mode.cre, mode_b.cre))
   rho = DensityPoly.pure(psi)

.. jupyter-execute::

   viz.display(rho)

.. jupyter-execute::

   rho_reduced = rho.partial_trace((mode_b,))

.. jupyter-execute::

   viz.display(rho_reduced)

The traced mode is contracted, leaving a reduced operator on the remaining mode.

End-to-end example
------------------

.. jupyter-execute::

   psi0 = KetPoly.from_ops(coeff=1.0)
   psi1 = adag @ psi0
   rho1 = DensityPoly.pure(psi1)

.. jupyter-execute::

   viz.display(psi1)

.. jupyter-execute::

   viz.display(rho1)

.. jupyter-execute::

   rho1.trace(), rho1.purity()

Summary
-------

The CCR layer operates entirely symbolically:

- operator words are rewritten via commutation
- results are stored as normally ordered monomials
- scalar quantities come from identity contributions
- density operations are built on the same mechanism

This enables overlaps, expectation values, traces, and reductions without
matrix representations.

See also
--------

- :class:`symop.ccr.algebra.ket.poly.KetPoly`
- :class:`symop.ccr.algebra.density.poly.DensityPoly`
- :class:`symop.ccr.algebra.op.poly.OpPoly`
- :func:`symop.ccr.algebra.ket.from_word.ket_from_word`
- :func:`symop.ccr.algebra.ket.inner.ket_inner`
- :func:`symop.ccr.algebra.density.trace.density_trace`
- :func:`symop.ccr.algebra.density.partial_trace.density_partial_trace`
