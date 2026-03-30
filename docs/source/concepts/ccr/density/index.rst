Density algebra
===============

The density layer represents symbolic density-like operators as finite sums of
outer products,

.. math::

   \rho \sim \sum_i c_i |L_i\rangle\langle R_i|,

where both :math:`L_i` and :math:`R_i` are normally ordered monomials.

The central object is :class:`symop.ccr.density.poly.DensityPoly`.

Main ideas
----------

Density terms keep left and right monomials explicit. This makes it possible to
define symbolic operator actions, traces, purity, Hilbert-Schmidt inner
products, and partial traces without constructing matrices.

Constructing a pure density from a ket
--------------------------------------

.. jupyter-execute::

   import symop.viz as viz

   from symop.core.operators import ModeOp
   from symop.ccr.algebra.ket.poly import KetPoly
   from symop.ccr.algebra.density.poly import DensityPoly
   from symop.modes.envelopes import GaussianEnvelope
   from symop.modes.labels import ModeLabel, Path, Polarization

   env = GaussianEnvelope(omega0=1.0, sigma=1.0, tau=0.0)

   mode = ModeOp(
       label=ModeLabel(
           envelope=env,
           path=Path("A"),
           polarization=Polarization.H(),
           ),
       user_label="a",
   )
           

   psi = KetPoly.from_ops(creators=(mode.cre,))
   rho = DensityPoly.pure(psi)
   viz.display(rho)

Left and right actions
----------------------

Left and right operator actions are kept distinct.

.. jupyter-execute::

   rho_left = rho.apply_left((mode.cre,))
   rho_right = rho.apply_right((mode.ann,))

   viz.display(rho_left)
   viz.display(rho_right)

Main operations
---------------

- :meth:`symop.ccr.density.poly.DensityPoly.trace`
- :meth:`symop.ccr.density.poly.DensityPoly.normalize_trace`
- :meth:`symop.ccr.density.poly.DensityPoly.inner`
- :meth:`symop.ccr.density.poly.DensityPoly.purity`
- :meth:`symop.ccr.density.poly.DensityPoly.partial_trace`

Example: trace and purity
-------------------------

.. jupyter-execute::

   rho.trace(), rho.purity()

API links
---------

- :class:`symop.ccr.algebra.density.poly.DensityPoly`
- :func:`symop.ccr.algebra.density.pure.density_pure`
- :func:`symop.ccr.algebra.density.trace.density_trace`
- :func:`symop.ccr.algebra.density.inner.density_inner`
- :func:`symop.ccr.algebra.density.partial_trace.density_partial_trace`
