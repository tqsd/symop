r"""Text rendering for mode and ladder operators.

Provides text-dispatch implementations for mode-level and ladder
operators, producing compact, terminal-friendly representations with
optional unicode formatting (e.g., subscripts and daggers).

The output is intended for debugging and inspection of CCR-based
operator structures.
"""

from __future__ import annotations

from typing import Any

from symop.core.operators import LadderOp, ModeOp
from symop.viz._dispatch import text

_DAGGER = "\u2020"  # †

_SUBSCRIPT_DIGITS = {
    "0": "\N{SUBSCRIPT ZERO}",
    "1": "\N{SUBSCRIPT ONE}",
    "2": "\N{SUBSCRIPT TWO}",
    "3": "\N{SUBSCRIPT THREE}",
    "4": "\N{SUBSCRIPT FOUR}",
    "5": "\N{SUBSCRIPT FIVE}",
    "6": "\N{SUBSCRIPT SIX}",
    "7": "\N{SUBSCRIPT SEVEN}",
    "8": "\N{SUBSCRIPT EIGHT}",
    "9": "\N{SUBSCRIPT NINE}",
    "-": "\N{SUBSCRIPT MINUS}",
}


def _to_subscript(n: int) -> str:
    r"""Convert an integer to a unicode subscript string.

    Parameters
    ----------
    n:
        Integer to convert.

    Returns
    -------
    str
        String representation of ``n`` using unicode subscript digits.

    Notes
    -----
    Characters not present in the subscript mapping are left unchanged.

    """
    s = str(int(n))
    return "".join(_SUBSCRIPT_DIGITS.get(ch, ch) for ch in s)


def _mode_tag(mode: Any) -> str:
    r"""Extract a display tag for a mode.

    Parameters
    ----------
    mode:
        Mode-like object with optional labeling attributes.

    Returns
    -------
    str
        Preferred tag derived from ``user_label`` if present, otherwise
        from ``display_index`` (as a subscript), or an empty string.

    Notes
    -----
    The function prioritizes user-defined labels over generated indices.

    """
    # Prefer user_label, else display_index if present, else fallback.
    lab = getattr(mode, "user_label", None)
    if lab:
        return str(lab)
    idx = getattr(mode, "display_index", None)
    if isinstance(idx, int):
        return _to_subscript(idx)
    return ""


@text.register
def _text_modeop(obj: Any, /, **kwargs: Any) -> str:
    r"""Render a mode operator as a labeled text string.

    Parameters
    ----------
    obj:
        Object expected to be a ``ModeOp`` instance.
    **kwargs:
        Additional keyword arguments forwarded to the ``text`` dispatcher.

    Returns
    -------
    str
        String representation including a base mode identifier and
        associated label information.

    Notes
    -----
    - If ``obj`` is not a ``ModeOp``, ``repr(obj)`` is returned.
    - Label rendering is delegated to the ``text`` dispatcher.
    - Falls back to a minimal representation if label rendering fails.

    """
    if not isinstance(obj, ModeOp):
        return repr(obj)

    tag = _mode_tag(obj)
    base = "mode" + (tag if tag else "")
    try:
        label_s = text(obj.label, **kwargs)
        return f"{base}: {label_s}"
    except Exception:
        return base


@text.register
def _text_ladderop(obj: Any, /, **kwargs: Any) -> str:
    r"""Render a ladder operator in compact symbolic form.

    Parameters
    ----------
    obj:
        Object expected to be a ``LadderOp`` instance.
    **kwargs:
        Additional keyword arguments (ignored).

    Returns
    -------
    str
        String representation such as ``a``, ``a†``, or with mode
        subscripts.

    Notes
    -----
    - Creation operators (``adag``) are rendered using a dagger symbol.
    - Annihilation operators (``a``) are rendered without modification.
    - Mode tags are appended as subscripts when available.
    - Falls back to a generic representation for unknown operator kinds.

    """
    if not isinstance(obj, LadderOp):
        return repr(obj)

    # kind.value is "a" or "adag" in your OperatorKind
    kind = getattr(obj.kind, "value", str(obj.kind))
    tag = _mode_tag(obj.mode)

    if kind == "adag":
        # a† with subscript
        return "a" + (tag if tag else "") + _DAGGER
    if kind == "a":
        return "a" + (tag if tag else "")
    # fallback for any other operator kind
    return f"{kind}" + (tag if tag else "")
