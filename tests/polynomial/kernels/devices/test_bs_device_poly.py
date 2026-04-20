from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

from symop.polynomial.kernels.devices.beamsplitter import (
    _parse_beamsplitter_params,
    _require_output_paths,
    beamsplitter_poly_ket,
)


class TestBeamSplitterHelpers(unittest.TestCase):
    def test_parse_top_level_single_pair(self) -> None:
        action = SimpleNamespace(
            params={
                "mode0": "m0",
                "mode1": "m1",
                "theta": 0.3,
                "out0": "p_out_0",
                "out1": "p_out_1",
            }
        )

        parsed = _parse_beamsplitter_params(action)

        self.assertEqual(len(parsed.pairs), 1)
        spec = parsed.pairs[0]
        self.assertEqual(spec.mode0_sig, "m0")
        self.assertEqual(spec.mode1_sig, "m1")
        self.assertEqual(spec.theta, 0.3)
        self.assertEqual(spec.out0, "p_out_0")
        self.assertEqual(spec.out1, "p_out_1")
        self.assertEqual(spec.phi_t, 0.0)
        self.assertEqual(spec.phi_r, 0.0)

    def test_require_output_paths_raises_when_missing(self) -> None:
        spec = SimpleNamespace(out0="p0", out1=None)

        with self.assertRaises(ValueError):
            _require_output_paths(spec)


class TestBeamSplitterKetKernel(unittest.TestCase):
    @patch("symop.polynomial.kernels.devices.beamsplitter.KetPolyState.from_ketpoly")
    @patch("symop.polynomial.kernels.devices.beamsplitter.beamsplitter_ketpoly")
    def test_calls_model_with_existing_modes(
        self,
        mock_bs: MagicMock,
        mock_from_ketpoly: MagicMock,
    ) -> None:
        mode0 = MagicMock()
        mode1 = MagicMock()

        state = MagicMock()
        state.ket = sentinel.ket_in
        state.mode_by_signature = {"m0": mode0, "m1": mode1}

        action = SimpleNamespace(
            params={
                "mode0": "m0",
                "mode1": "m1",
                "theta": 0.25,
                "phi_t": 0.1,
                "phi_r": 0.2,
                "out0": "out_a",
                "out1": "out_b",
            }
        )
        ctx = MagicMock()

        mock_bs.return_value = sentinel.ket_out
        mock_from_ketpoly.return_value = sentinel.state_out

        out = beamsplitter_poly_ket(state=state, action=action, ctx=ctx)

        self.assertIs(out, sentinel.state_out)
        mock_bs.assert_called_once_with(
            sentinel.ket_in,
            mode0=mode0,
            mode1=mode1,
            out0="out_a",
            out1="out_b",
            theta=0.25,
            phi_t=0.1,
            phi_r=0.2,
        )
        mock_from_ketpoly.assert_called_once_with(sentinel.ket_out)

    @patch("symop.polynomial.kernels.devices.beamsplitter.KetPolyState.from_ketpoly")
    @patch("symop.polynomial.kernels.devices.beamsplitter.beamsplitter_ketpoly")
    def test_synthesizes_missing_mode0_from_mode1(
        self,
        mock_bs: MagicMock,
        mock_from_ketpoly: MagicMock,
    ) -> None:
        mode1 = MagicMock()
        synthesized_mode0 = MagicMock()
        mode1.with_path.return_value = synthesized_mode0

        state = MagicMock()
        state.ket = sentinel.ket_in
        state.mode_by_signature = {"m1": mode1}

        action = SimpleNamespace(
            params={
                "mode0": "missing",
                "mode1": "m1",
                "theta": 0.5,
                "in0": "in_path_0",
                "out0": "out0",
                "out1": "out1",
            }
        )
        ctx = MagicMock()

        mock_bs.return_value = sentinel.ket_out
        mock_from_ketpoly.return_value = sentinel.state_out

        out = beamsplitter_poly_ket(state=state, action=action, ctx=ctx)

        self.assertIs(out, sentinel.state_out)
        mode1.with_path.assert_called_once_with("in_path_0")
        mock_bs.assert_called_once()
        _, kwargs = mock_bs.call_args
        self.assertIs(kwargs["mode0"], synthesized_mode0)
        self.assertIs(kwargs["mode1"], mode1)

    @patch("symop.polynomial.kernels.devices.beamsplitter.KetPolyState.from_ketpoly")
    @patch("symop.polynomial.kernels.devices.beamsplitter.beamsplitter_ketpoly")
    def test_skips_pair_when_both_modes_missing(
        self,
        mock_bs: MagicMock,
        mock_from_ketpoly: MagicMock,
    ) -> None:
        state = MagicMock()
        state.ket = sentinel.ket_in
        state.mode_by_signature = {}

        action = SimpleNamespace(
            params={
                "mode0": "m0",
                "mode1": "m1",
                "theta": 0.5,
                "out0": "out0",
                "out1": "out1",
            }
        )
        ctx = MagicMock()

        mock_from_ketpoly.return_value = sentinel.state_out

        out = beamsplitter_poly_ket(state=state, action=action, ctx=ctx)

        self.assertIs(out, sentinel.state_out)
        mock_bs.assert_not_called()
        mock_from_ketpoly.assert_called_once_with(sentinel.ket_in)
