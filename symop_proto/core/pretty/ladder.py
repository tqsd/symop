from __future__ import annotations


def _text_subscript(op) -> str:
    if op.mode.user_label is not None:
        return str(op.mode.user_label)
    if op.mode.display_index is not None:
        return str(op.mode.display_index)
    return "?"


def _latex_subscript(op) -> str:
    if op.mode.user_label is not None:
        s = op.mode.user_label.replace("_", r"\_")
        return rf"\text{{{s}}}"
    if op.mode.display_index is not None:
        return f"{op.mode.display_index}"
    return "?"


def ladder_text(op, base: str = "m") -> str:
    """
    Plain-text: m<label> or m<index>, with † for creation.
    """
    sub = _text_subscript(op)
    return f"{base}{sub}†" if op.is_creation else f"{base}{sub}"


def ladder_latex(op, base: str = "a") -> str:
    r"""
    LaTeX (no $...$): \hat{a}_{sub} or \hat{a}_{sub}^{\dagger}.
    """
    sub = _latex_subscript(op)
    core = rf"\hat{{{base}}}_{{{sub}}}"
    return core + r"^{\dagger}" if op.is_creation else core
