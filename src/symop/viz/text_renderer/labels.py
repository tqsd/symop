r"""Text rendering for mode labels.

Provides text-dispatch implementations for mode-related label types,
including paths, polarizations, and full mode labels.

The output is intended for debugging and inspection, exposing key label
attributes such as path identifiers, Jones vectors, and envelope types.
"""

from __future__ import annotations

from typing import Any

from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import Path
from symop.modes.labels.polarization import Polarization
from symop.viz._dispatch import text


def _fmt_complex(z: complex, *, decimals: int = 6) -> str:
    r"""Format a complex number for compact label display.

    Parameters
    ----------
    z:
        Complex number to format.
    decimals:
        Number of decimal places used for rounding.

    Returns
    -------
    str
        String representation of ``z`` in ``a±bj`` form.

    Notes
    -----
    - Real and imaginary parts are rounded independently.
    - Purely real numbers are returned without an imaginary part.

    """
    r = round(float(z.real), decimals)
    i = round(float(z.imag), decimals)
    if i == 0.0:
        return str(r)
    sign = "+" if i >= 0 else "-"
    return f"{r}{sign}{abs(i)}j"


@text.register
def _(p: Path, /, **kwargs: Any) -> str:
    r"""Render a path label as a text string.

    Parameters
    ----------
    p:
        Path label instance.
    **kwargs:
        Additional keyword arguments (ignored).

    Returns
    -------
    str
        String representation of the form ``PathLabel(name)``.

    """
    return f"PathLabel({p.name})"


@text.register
def _(pol: Polarization, /, **kwargs: Any) -> str:
    r"""Render a polarization label using its Jones vector.

    Parameters
    ----------
    pol:
        Polarization label instance.
    **kwargs:
        Additional keyword arguments (ignored).

    Returns
    -------
    str
        String representation including the Jones vector components.

    Notes
    -----
    The Jones vector entries are formatted using :func:`_fmt_complex`.

    """
    a, b = pol.jones
    return f"PolarizationLabel(jones=({_fmt_complex(a)}, {_fmt_complex(b)}))"


@text.register
def _(m: ModeLabel, /, **kwargs: Any) -> str:
    r"""Render a mode label as a structured text representation.

    Parameters
    ----------
    m:
        Mode label instance.
    **kwargs:
        Additional keyword arguments forwarded to the ``text`` dispatcher.

    Returns
    -------
    str
        String representation including path, polarization, and envelope type.

    Notes
    -----
    - Path and polarization are rendered via the ``text`` dispatcher.
    - The envelope is represented by its class name only.

    """
    return (
        "ModeLabel("
        + f"path={text(m.path)}, "
        + f"pol={text(m.polarization)}, "
        + f"envelope={type(m.envelope).__name__}"
        + ")"
    )
