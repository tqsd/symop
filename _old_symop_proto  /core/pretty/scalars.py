from __future__ import annotations
from math import isfinite, sqrt, fabs
from fractions import Fraction


def _approx_int(v: float, tol: float) -> int | None:
    if not isfinite(v):
        return None
    k = round(v)
    return k if abs(v - k) <= tol else None


def _match_inv_sqrt(v: float, tol: float = 1e-12, nmax: int = 16) -> int | None:
    if not isfinite(v):
        return None
    for n in range(2, nmax + 1):
        if abs(v - 1.0 / sqrt(n)) <= tol:
            return n
    return None


def _match_rational(
    v: float, tol: float = 1e-12, qmax: int = 32
) -> tuple[int, int] | None:
    if not isfinite(v):
        return None
    frac = Fraction(v).limit_denominator(qmax)
    if abs(v - float(frac)) <= tol:
        return frac.numerator, frac.denominator
    return None


def scalar_to_text(x: float, *, tol: float = 1e-12) -> str:
    if fabs(x) < tol:
        return "0"
    k = _approx_int(x, tol)
    if k is not None:
        return f"{k}"
    sgn = "-" if x < 0 else ""
    v = abs(x)
    n = _match_inv_sqrt(v, tol)
    if n is not None:
        return f"{sgn}1/sqrt({n})"
    rat = _match_rational(v, tol)
    if rat is not None:
        p, q = rat
        return f"{sgn}{p}/{q}"
    return f"{x:g}"


def complex_to_text(c: complex, *, tol: float = 1e-12) -> str:
    a = 0.0 if abs(c.real) < tol else c.real
    b = 0.0 if abs(c.imag) < tol else c.imag
    if b == 0.0:
        return scalar_to_text(a, tol=tol)
    if a == 0.0:
        if _approx_int(b, tol) == 1:
            return "i"
        if _approx_int(b, tol) == -1:
            return "-i"
        return scalar_to_text(b, tol=tol) + "·i"
    sgn = "+" if b > 0 else "-"
    b_abs = abs(b)
    b_txt = (
        "i" if _approx_int(b_abs, tol) == 1 else scalar_to_text(b_abs, tol=tol) + "·i"
    )
    return f"{scalar_to_text(a, tol=tol)}{sgn}{b_txt}"


def scalar_to_latex(x: float, *, tol: float = 1e-12, use_tfrac: bool = False) -> str:

    if fabs(x) < tol:
        return "0"
    k = _approx_int(x, tol)
    if k is not None:
        return f"{k}"
    sgn = "-" if x < 0 else ""
    v = abs(x)
    n = _match_inv_sqrt(v, tol)
    if n is not None:
        frac = r"\tfrac" if use_tfrac else r"\frac"
        return sgn + rf"{frac}{{1}}{{\sqrt{{{n}}}}}"
    rat = _match_rational(v, tol)
    if rat is not None:
        p, q = rat
        if q == 1:
            return sgn + f"{p}"
        frac = r"\tfrac" if use_tfrac else r"\frac"
        return sgn + rf"{frac}{{{p}}}{{{q}}}"
    return f"{x:g}"


def complex_to_latex(c: complex, *, tol: float = 1e-12, use_tfrac: bool = False) -> str:
    a = 0.0 if abs(c.real) < tol else c.real
    b = 0.0 if abs(c.imag) < tol else c.imag
    if b == 0.0:
        return scalar_to_latex(a, tol=tol, use_tfrac=use_tfrac)
    if a == 0.0:
        if _approx_int(b, tol) == 1:
            return r"\mathrm{i}"
        if _approx_int(b, tol) == -1:
            return r"-\mathrm{i}"
        return scalar_to_latex(b, tol=tol, use_tfrac=use_tfrac) + r"\,\mathrm{i}"
    sign = "+" if b > 0 else "-"
    b_abs = abs(b)
    b_tex = (
        r"\mathrm{i}"
        if _approx_int(b_abs, tol) == 1
        else scalar_to_latex(b_abs, tol=tol, use_tfrac=use_tfrac) + r"\,\mathrm{i}"
    )
    return rf"{scalar_to_latex(a, tol=tol, use_tfrac=use_tfrac)}{sign}{b_tex}"
