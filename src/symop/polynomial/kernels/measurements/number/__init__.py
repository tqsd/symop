r"""Polynomial number-measurement kernels.

This package provides backend implementations of photon-number
measurement for polynomial state representations.

It includes support for three measurement modes:

- observation: compute full probability distributions without modifying
  the state
- detection: sample outcomes and return post-measurement states
- postselection: condition on a specified outcome and return the
  corresponding branch

Both single-target number measurement and joint per-port number
measurement are supported. The behavior is controlled by the measurement
resolution specified in the semantic action.

Structure
---------
The implementation is organized into:

- ``common``:
    Shared structural logic for resolving targets, counting quanta, and
    projecting states onto number sectors.

- ``observe``:
    Non-destructive evaluation of number statistics.

- ``detect``:
    Stochastic sampling of number outcomes and state collapse.

- ``postselect``:
    Deterministic conditioning on specified number outcomes.

Notes
-----
The implementation operates on symbolic polynomial representations of
quantum states. Number measurement is evaluated structurally by counting
creation operators associated with selected mode signatures.

For density states, probabilities are obtained from traces of projected
density operators. For ket states, probabilities are computed from
squared norms of projected states.

Joint per-port measurement extends the outcome space to tuples of number
outcomes, one per selected port.

"""
