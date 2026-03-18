r"""Dispatch-based visualization utilities.

Defines generic dispatchers for rendering and displaying symbolic objects
in different formats, including text, LaTeX, and plots.

The dispatch system allows type-specific implementations to be registered
externally, enabling extensible visualization across the symop framework.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any


@singledispatch
def plot(obj: Any, /, **kwargs: Any) -> Any:
    r"""Plot an object using a registered handler.

    Parameters
    ----------
    obj:
        Object to be plotted.
    **kwargs:
        Additional keyword arguments forwarded to the registered plot handler.

    Raises
    ------
    TypeError
        If no plot handler is registered for the object's type.

    Notes
    -----
    Implementations should be registered via ``plot.register`` for specific types.

    """
    raise TypeError(f"No plot() handler registered for type {type(obj)!r}")


@singledispatch
def latex(obj: Any, /, **kwargs: Any) -> str:
    r"""Render an object as a LaTeX string.

    Parameters
    ----------
    obj:
        Object to render.
    **kwargs:
        Additional keyword arguments forwarded to the registered handler.

    Returns
    -------
    str
        LaTeX representation of the object, or an empty string if no
        specialized renderer is available.

    Notes
    -----
    Implementations should be registered via ``latex.register``.

    """
    return ""


@singledispatch
def text(obj: Any, /, **kwargs: Any) -> str:
    r"""Render an object as a plain-text string.

    Parameters
    ----------
    obj:
        Object to render.
    **kwargs:
        Additional keyword arguments forwarded to the registered handler.

    Returns
    -------
    str
        Textual representation of the object. Defaults to ``repr(obj)`` if
        no specialized renderer is registered.

    Notes
    -----
    Implementations should be registered via ``text.register``.

    """
    return repr(obj)


def display(obj: Any, /, **kwargs: Any) -> None:
    r"""Display an object in a Jupyter-friendly way.

    Parameters
    ----------
    obj:
        Object to display.
    **kwargs:
        Additional keyword arguments forwarded to rendering functions.

    Returns
    -------
    None

    Notes
    -----
    - If IPython is available, attempts to render using LaTeX.
    - Falls back to plain-text output otherwise.
    - Uses :func:`latex` and :func:`text` dispatchers internally.

    """
    from symop.viz._optional import try_ipython_display

    ipy = try_ipython_display()
    if ipy is None:
        print(text(obj, **kwargs))
        return

    ipy_display, IPyLatex = ipy
    s = latex(obj, **kwargs)
    if s:
        ipy_display(IPyLatex("$" + s + "$"))
    else:
        print(text(obj, **kwargs))
