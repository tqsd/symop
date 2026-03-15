"""Concrete photonic channel models for symbolic CCR polynomials.

This package provides ready-to-use implementations of common linear
optical transformations expressed in terms of the symbolic polynomial
rewrite primitives. The models correspond to standard photonic devices
such as beamsplitters, phase shifters, interferometers, and loss
channels.

Each model constructs the appropriate linear or dilation map and
applies it to ket, density, or operator polynomial representations.

Modules
-------
beamsplitter
    Passive two-mode beamsplitter transformations.
phase
    Single-mode phase-shift transformations.
mzi
    Mach–Zehnder interferometer built from beamsplitters and phase shifts.
pure_loss
    Bosonic pure-loss channel implemented via beamsplitter dilation.
amplifier
    Phase-insensitive bosonic amplification channel.

Notes
-----
These models are thin wrappers around the primitive rewrite operations
defined in ``symop.polynomial.channels.primitives`` and are intended to
provide convenient high-level building blocks for photonic simulations.

"""
