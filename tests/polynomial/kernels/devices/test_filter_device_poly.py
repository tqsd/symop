from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

from symop.polynomial.kernels.devices.filter import (
    _make_env_mode_for_loss,
    _parse_filter_params,
    filter_poly_ket,
)


class TestFilterHelpers(unittest.TestCase):
    def test_parse_filter_params_requires_mapping(self) -> None:
        action = SimpleNamespace(params={"eta_by_mode": {"m0": 0.8}})
        parsed = _parse_filter_params(action)
        self.assertEqual(parsed.eta_by_mode, {"m0": 0.8})

    def test_make_env_mode_for_loss_uses_fresh_env_path_and_same_envelope(self) -> None:
        env_mode_after_path = MagicMock()
        final_env_mode = MagicMock()

        signal_mode = MagicMock()
        signal_mode.label.envelope = sentinel.env
        signal_mode.with_path.return_value = env_mode_after_path
        env_mode_after_path.with_envelope.return_value = final_env_mode

        ctx = MagicMock()
        ctx.allocate_path.return_value = "env_1"

        out = _make_env_mode_for_loss(ctx=ctx, signal_mode=signal_mode)

        self.assertIs(out, final_env_mode)
        ctx.allocate_path.assert_called_once_with(hint="env")
        signal_mode.with_path.assert_called_once_with("env_1")
        env_mode_after_path.with_envelope.assert_called_once_with(sentinel.env)


class TestFilterKetKernel(unittest.TestCase):
    @patch("symop.polynomial.kernels.devices.filter.filter_poly_density")
    def test_ket_wrapper_converts_to_density_and_delegates(
        self,
        mock_filter_density: MagicMock,
    ) -> None:
        density_state = MagicMock()
        ket_state = MagicMock()
        ket_state.to_density.return_value = density_state

        action = SimpleNamespace(params={"eta_by_mode": {"m0": 0.5}})
        ctx = MagicMock()

        mock_filter_density.return_value = sentinel.out

        out = filter_poly_ket(state=ket_state, action=action, ctx=ctx)

        self.assertIs(out, sentinel.out)
        ket_state.to_density.assert_called_once_with()
        mock_filter_density.assert_called_once_with(
            state=density_state,
            action=action,
            ctx=ctx,
        )
