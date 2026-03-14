r"""Utility helpers for transmissivity normalization.

This module provides small utilities for handling transmissivity values
(:math:`\eta`) produced by spectral filtering operations. In particular,
it includes helpers to ensure numerical results respect the physical
constraint :math:`0 \le \eta \le 1`.
"""

import math


def clip_eta(x: float) -> float:
    r"""Clip a scalar to :math:`[0,1]` with non-finite mapped to 0.

    Parameters
    ----------
    x:
        Candidate transmissivity.

    Returns
    -------
    float
        Value clipped to :math:`[0, 1]`.

    Notes
    -----
    In exact arithmetic, the transmissivity

    .. math::

        \eta = \frac{1}{2\pi}\int |H(\omega)|^2 |Z(\omega)|^2 d\omega

    should lie in :math:`[0,1]` for passive amplitude transfers with
    :math:`|H|\le 1`. In floating-point arithmetic and for "semantic" filters
    (e.g. complements), tiny violations can occur; clipping keeps the
    higher-level semantics stable.

    """
    if not math.isfinite(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
