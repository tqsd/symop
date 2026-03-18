r"""Optional visualization dependencies.

Provides helper utilities to access optional visualization backends
such as matplotlib and IPython display, with graceful fallbacks when
these dependencies are not installed.
"""

from __future__ import annotations

from typing import Any


def require_matplotlib_pyplot() -> Any:
    r"""Import and return ``matplotlib.pyplot``.

    Returns
    -------
    Any
        The ``matplotlib.pyplot`` module.

    Raises
    ------
    RuntimeError
        If matplotlib is not available.

    Notes
    -----
    This helper centralizes optional dependency handling for plotting.
    Users can install the required extras via ``pip install symop[viz]``.

    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "Plotting requires optional dependencies. "
            "Install with: pip install symop[viz]"
        ) from e
    return plt


def try_ipython_display() -> tuple[Any, Any] | None:
    r"""Attempt to import IPython display utilities.

    Returns
    -------
    tuple[Any, Any] or None
        A tuple ``(display, Latex)`` if IPython is available, otherwise ``None``.

    Notes
    -----
    This function enables optional Jupyter integration without requiring
    IPython as a hard dependency.

    """
    try:
        from IPython.display import Latex as IPyLatex
        from IPython.display import display as ipy_display
    except Exception:
        return None
    return ipy_display, IPyLatex
