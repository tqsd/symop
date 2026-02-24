from __future__ import annotations
import unittest
from dataclasses import dataclass
from typing import Tuple, Any

from symop_proto.core.protocols import LadderOpProto
from tests.utils.case import ExtendedTestCase

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp, LadderOp
from symop_proto.algebra.op_poly.combine import op_combine_like_terms


def make_mode(
    path: str = "A",
    *,
    omega=1.0,
    sigma=0.3,
    tau=0.0,
    phi=0.0,
    pol: PolarizationLabel | None = None,
) -> ModeOp:
    pol = pol or PolarizationLabel.H()
    env = GaussianEnvelope(omega0=omega, sigma=sigma, tau=tau, phi0=phi)
    label = ModeLabel(PathLabel(path), pol)
    return ModeOp(env=env, label=label)


@dataclass(frozen=True)
class _TestTerm:
    ops: Tuple[LadderOpProto, ...]
    coeff: complex
    _sig: tuple
    _approx_sig: tuple

    @property
    def signature(self) -> tuple:
        return self._sig

    def approx_signature(self, **env_kw: Any) -> tuple:
        return self._approx_sig


def _factory(ops: Tuple[LadderOp, ...], coeff: complex) -> _TestTerm:
    return _TestTerm(
        ops=ops,
        coeff=coeff,
        _sig=("out", len(ops)),
        _approx_sig=("outa", len(ops)),
    )


class TestOpCombineLikeTerms(ExtendedTestCase):
    def test_combine_exact_signature_and_sum_coeffs(self):
        mA = make_mode("A")
        t1 = _TestTerm(ops=(mA.create,), coeff=2.0, _sig=("S", 1), _approx_sig=("A", 1))
        t2 = _TestTerm(ops=(mA.ann,), coeff=3.0, _sig=("S", 1), _approx_sig=("A", 1))
        t3 = _TestTerm(
            ops=(
                mA.create,
                mA.ann,
            ),
            coeff=-1.0,
            _sig=("S", 2),
            _approx_sig=("A", 2),
        )
        out = op_combine_like_terms((t1, t2, t3), term_factory=_factory)
        self.assertEqual(len(out), 2)
        d = {o.ops: o.coeff for o in out}
        self.assertIn(t1.ops, d)
        self.assertAlmostEqual(d[t1.ops], 5.0)
        self.assertIn(t3.ops, d)
        self.assertAlmostEqual(d[t3.ops], -1.0)

    def test_zero_sum_bucket_is_dropped(self):
        mA = make_mode("A")
        t1 = _TestTerm(
            ops=(mA.create,), coeff=1.25, _sig=("Z", 0), _approx_sig=("Za", 0)
        )
        t2 = _TestTerm(
            ops=(mA.create,), coeff=-1.25, _sig=("Z", 0), _approx_sig=("Za", 0)
        )
        out = op_combine_like_terms((t1, t2), term_factory=_factory)
        self.assertEqual(out, ())

    def test_use_approx_groups_same_mode_same_key(self):
        mA = make_mode("A")
        t1 = _TestTerm(
            ops=(mA.create,),
            coeff=1.0,
            _sig=("E", 1),
            _approx_sig=("Ap", "A", "C"),
        )
        t2 = _TestTerm(
            ops=(mA.create,),
            coeff=2.0,
            _sig=("E", 2),
            _approx_sig=("Ap", "A", "C"),
        )
        self.assertEqual(t1.approx_signature(), t2.approx_signature())
        out = op_combine_like_terms((t1, t2), approx=True, term_factory=_factory)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].ops, t1.ops)
        self.assertAlmostEqual(out[0].coeff, 3.0)

    def test_use_approx_respects_different_modes(self):
        mA = make_mode("A")
        mB = make_mode("B")
        t1 = _TestTerm(
            ops=(mA.create,),
            coeff=1.0,
            _sig=("E", 1),
            _approx_sig=("Ap", "A", "C"),
        )
        t2 = _TestTerm(
            ops=(mB.create,),
            coeff=2.0,
            _sig=("E", 2),
            _approx_sig=("Ap", "B", "C"),
        )
        out = op_combine_like_terms((t1, t2), approx=True, term_factory=_factory)
        self.assertEqual(len(out), 2)

    def test_env_kwargs_are_forwarded_and_preserve_word(self):
        mA = make_mode("A")

        class _EnvTerm(_TestTerm):
            def approx_signature(self, **env_kw: Any) -> tuple:
                w = tuple(
                    ("C" if op.is_creation else "A", op.mode.label.signature)
                    for op in self.ops
                )
                return ("ApTol", env_kw.get("tol", None), w)

        t1 = _EnvTerm(ops=(mA.create,), coeff=1.0, _sig=("E", 1), _approx_sig=("X",))
        t2 = _EnvTerm(ops=(mA.create,), coeff=2.0, _sig=("E", 2), _approx_sig=("X",))
        k1 = t1.approx_signature(tol=42)
        k2 = t2.approx_signature(tol=42)
        self.assertEqual(k1, k2)
        out = op_combine_like_terms(
            (t1, t2), approx=True, term_factory=_factory, tol=42
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].ops, t1.ops)
        self.assertAlmostEqual(out[0].coeff, 3.0)


if __name__ == "__main__":
    unittest.main()
