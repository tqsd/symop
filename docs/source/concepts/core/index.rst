Core symbolic terms
===================

This section introduces the symbolic term layer at the heart of Symop.

These objects provide the low-level algebraic building blocks used throughout
the package to represent ordered operator words, normal-ordered monomials, and
coefficient-bearing ket- and density-like terms.

The main concepts covered here are:

- :class:`symop.core.terms.op_term.OpTerm`
- :class:`symop.core.monomial.Monomial`
- :class:`symop.core.terms.ket_term.KetTerm`
- :class:`symop.core.terms.density_term.DensityTerm`

The canonical commutation relations and rewriting rules built on top of these
objects are documented separately in the CCR section.

.. toctree::
   :maxdepth: 1

   terms
