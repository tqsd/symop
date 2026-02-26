"""Envelope implementations.

This subpackage defines time- and frequency-domain mode envelopes,
including analytic envelopes (e.g., Gaussian) and numerically
constructed envelopes (e.g., filtered envelopes).

Envelopes provide evaluation, overlap computation, and plotting
utilities.
"""

from .filtered import FilteredEnvelope
from .gaussian import GaussianEnvelope

__all__ = ["GaussianEnvelope", "FilteredEnvelope"]
