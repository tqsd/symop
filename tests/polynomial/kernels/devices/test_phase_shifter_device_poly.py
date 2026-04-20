from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

from symop.polynomial.kernels.devices.phase_shifter import (
    _parse_phase_shifter_params,
    phase_shifter_poly_density,
)


class TestPhaseShifterHelpers(unittest.TestCase):
    def test_parse_requires_phi(self) -> None:
        action = SimpleNamespace(params={"path": "p0"})
        with self.assertRaises(KeyError):
            _parse_phase_shifter_params(action)

    def test_parse_requires_real_phi(self) -> None:
        action = SimpleNamespace(params={"path": "p0", "phi": "bad"})
        with self.assertRaises(TypeError):
            _parse_phase_shifter_params(action)


class TestPhaseShifterDensityKernel(unittest.TestCase):
    @patch("symop.polynomial.kernels.devices.phase_shifter.DensityPolyState.from_densitypoly")
    @patch("symop.polynomial.kernels.devices.phase_shifter.phase_densitypoly")
    def test_applies_to_all_modes_on_selected_path(
        self,
        mock_phase: MagicMock,
        mock_from_densitypoly: MagicMock,
    ) -> None:
        mode0 = MagicMock()
        mode1 = MagicMock()

        state = MagicMock()
        state.rho = sentinel.rho0
        state.modes_on_path.return_value = [mode0, mode1]

        mock_phase.side_effect = [sentinel.rho1, sentinel.rho2]
        mock_from_densitypoly.return_value = sentinel.out_state

        action = SimpleNamespace(params={"path": "p0", "phi": 0.7})

        out = phase_shifter_poly_density(
            state=state,
            action=action,
            ctx=MagicMock(),
        )

        self.assertIs(out, sentinel.out_state)
        state.modes_on_path.assert_called_once_with("p0")
        self.assertEqual(mock_phase.call_count, 2)
        self.assertEqual(mock_phase.call_args_list[0].kwargs["mode"], mode0)
        self.assertEqual(mock_phase.call_args_list[1].kwargs["mode"], mode1)
        mock_from_densitypoly.assert_called_once_with(sentinel.rho2)
