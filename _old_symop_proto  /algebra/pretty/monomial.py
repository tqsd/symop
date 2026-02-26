from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from fractions import Fraction
from math import fabs, gcd, isfinite, sqrt

from symop_proto.core.monomial import MonomialProto
from symop_proto.core.protocols import LadderOpProto

IDENTITY_LATEX = r"\mathbb{I}"  # or r"\mathbf{1}" if you prefer bold
IDENTITY_TEXT = "I"  # plain-text fallback


def _is_identity_monomial(m) -> bool:
    return (len(m.creators) == 0) and (len(m.annihilators) == 0)


def collect_mode_order(terms: Iterable) -> dict[tuple, int]:
    """Build a stable index for modes across a set of terms,
    deduped by content (mode.signature).
    """
    order: dict[tuple, int] = {}
    next_idx = 1
    for t in terms:
        mono = t.monomial if hasattr(t, "monomial") else t
        for op in (*mono.creators, *mono.annihilators):
            sig = op.mode.signature  # content-based
            if sig not in order:
                order[sig] = next_idx
                next_idx += 1
    return order


def _op_symbol(op: LadderOpProto, idx: int, dagger: bool) -> str:
    base = "m"
    return f"{base}{idx}†" if dagger else f"{base}{idx}"


def monomial_to_str(m: MonomialProto, mode_index: dict[tuple, int]) -> str:
    c_count = Counter(op.mode.signature for op in m.creators)
    a_count = Counter(op.mode.signature for op in m.annihilators)

    def by_idx(sig):
        return mode_index[sig]

    c_parts: list[str] = []
    for sig in sorted(c_count, key=by_idx):
        n = c_count[sig]
        rep = next(op for op in m.creators if op.mode.signature == sig)
        sym = _op_symbol(rep, mode_index[sig], dagger=True)
        c_parts.append(sym if n == 1 else f"{sym}^{n}")
    a_parts: list[str] = []
    for sig in sorted(a_count, key=by_idx):
        n = a_count[sig]
        rep = next(op for op in m.annihilators if op.mode.signature == sig)
        sym = _op_symbol(rep, mode_index[sig], dagger=False)
        a_parts.append(sym if n == 1 else f"{sym}^{n}")

    if not c_parts and not a_parts:
        return IDENTITY_TEXT
    if c_parts and a_parts:
        return f"{' '.join(c_parts)} {' '.join(a_parts)}"
    return " ".join(c_parts) or " ".join(a_parts)


# ---------- LaTeX helpers ----------


# ---------- scalar pretty-printers ----------


def _approx_int(v: float, tol: float) -> int | None:
    """Return k if v ≈ k within tol, else None."""
    if not isfinite(v):
        return None
    k = round(v)
    return k if abs(v - k) <= tol else None


def _match_inv_sqrt(v: float, tol: float = 1e-12, nmax: int = 16) -> int | None:
    """Return n if v ≈ 1/sqrt(n) within tol, else None."""
    if not isfinite(v):
        return None
    for n in range(2, nmax + 1):
        if abs(v - 1.0 / sqrt(n)) <= tol:
            return n
    return None


def _match_rational(
    v: float, tol: float = 1e-12, qmax: int = 16
) -> tuple[int, int] | None:
    """Return simplified (p,q) if v ≈ p/q with q ≤ qmax, else None."""
    if not isfinite(v):
        return None
    frac = Fraction(v).limit_denominator(qmax)
    if abs(v - float(frac)) <= tol:
        p, q = frac.numerator, frac.denominator
        g = gcd(abs(p), q)
        p //= g
        q //= g
        return (p, q)
    return None


def _scalar_to_latex(x: float, *, tol: float = 1e-12, use_tfrac: bool = False) -> str:
    """Pretty LaTeX for a real scalar (no $...$)."""
    if fabs(x) < tol:
        return "0"

    # integers first
    k = _approx_int(x, tol)
    if k is not None:
        return f"{k}"

    sgn = "-" if x < 0 else ""
    v = abs(x)

    # 1/sqrt(n)
    n = _match_inv_sqrt(v, tol=tol)
    if n is not None:
        frac = r"\tfrac" if use_tfrac else r"\frac"
        return sgn + rf"{frac}{{1}}{{\sqrt{{{n}}}}}"

    # rational p/q
    rat = _match_rational(v, tol=tol)
    if rat is not None:
        p, q = rat
        if q == 1:
            return sgn + f"{p}"
        frac = r"\tfrac" if use_tfrac else r"\frac"
        return sgn + rf"{frac}{{{p}}}{{{q}}}"

    # fallback
    return f"{x:g}"


# ---------- complex pretty-printer ----------


def complex_to_latex(c: complex, eps: float = 1e-15, *, use_tfrac: bool = False) -> str:
    r"""Render a complex a+ib without $...$.
    Examples: '1', '\frac{1}{\sqrt{2}}', 'i', '-i', '1 \pm i', '1 \pm \tfrac{1}{\sqrt{2}}\,i'
    """
    a = 0.0 if abs(c.real) < eps else c.real
    b = 0.0 if abs(c.imag) < eps else c.imag

    if b == 0.0:
        return _scalar_to_latex(a, use_tfrac=use_tfrac)

    if a == 0.0:
        if b == 1.0:
            return r"\mathrm{i}"
        if b == -1.0:
            return r"-\mathrm{i}"
        return _scalar_to_latex(b, use_tfrac=use_tfrac) + r"\,\mathrm{i}"

    sign = "+" if b > 0 else "-"
    b_abs = abs(b)
    b_tex = (
        r"\mathrm{i}"
        if _approx_int(b_abs, eps) == 1
        else _scalar_to_latex(b_abs, use_tfrac=use_tfrac) + r"\,\mathrm{i}"
    )
    return rf"{_scalar_to_latex(a, use_tfrac=use_tfrac)}{sign}{b_tex}"


# ---------- Operators ----------


def _op_symbol_latex(op: LadderOpProto, idx: int, dagger: bool) -> str:
    r"""\hat{a}_idx  or  \hat{a}_idx^{\dagger}
    (no $...$, no displaystyle)
    """
    base = "a"
    return (
        rf"\hat{{{base}}}_{{{idx}}}^{{\dagger}}"
        if dagger
        else rf"\hat{{{base}}}_{{{idx}}}"
    )


def _pow_group(s: str, n: int) -> str:
    r"""Wrap the entire operator BEFORE raising to a power:
        (\hat a_i^\dagger)^n
    not   \hat a_i^{\dagger^n}
    """
    return s if n == 1 else rf"\left({s}\right)^{{{n}}}"


def monomial_to_latex(m: MonomialProto, mode_index: dict[tuple, int]) -> str:
    r"""Build a LaTeX string for a normally-ordered monomial:
      (a_i^\dagger)^r ... (a_j)^{s}
    Identity is \mathbf{1} for MathJax safety.
    """
    c_count = Counter(op.mode.signature for op in m.creators)
    a_count = Counter(op.mode.signature for op in m.annihilators)

    def by_idx(sig):
        return mode_index[sig]

    # creators (left)
    c_parts: list[str] = []
    for sig in sorted(c_count, key=by_idx):
        n = c_count[sig]
        rep = next(op for op in m.creators if op.mode.signature == sig)
        sym = _op_symbol_latex(rep, mode_index[sig], dagger=True)
        c_parts.append(_pow_group(sym, n))

    # annihilators (right)
    a_parts: list[str] = []
    for sig in sorted(a_count, key=by_idx):
        n = a_count[sig]
        rep = next(op for op in m.annihilators if op.mode.signature == sig)
        sym = _op_symbol_latex(rep, mode_index[sig], dagger=False)
        a_parts.append(_pow_group(sym, n))
    if not c_parts and not a_parts:
        return IDENTITY_LATEX

    if c_parts and a_parts:
        # small spacing between creation and annihilation blocks
        return rf"{' '.join(c_parts)} \;{' '.join(a_parts)}"

    return " ".join(c_parts) or " ".join(a_parts)
