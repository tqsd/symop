from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from symop.core.types.operator_kind import OperatorKind
from symop.polynomial.kernels.devices.number_state_source import (
    _creator_word_for_number_state,
    number_state_source_poly_ket,
)


class TestNumberStateSourceHelpers(unittest.TestCase):
    def test_creator_word_repeats_creation_ops_in_mode_order(self) -> None:
        cre0 = MagicMock()
        cre1 = MagicMock()

        mode0 = MagicMock()
        mode0.signature = "m0"
        mode0.cre = cre0
        cre0.kind = OperatorKind.CRE

        mode1 = MagicMock()
        mode1.signature = "m1"
        mode1.cre = cre1
        cre1.kind = OperatorKind.CRE

        word = _creator_word_for_number_state(
            source_modes=(mode0, mode1),
            excitations_by_mode={"m0": 2, "m1": 1},
        )

        self.assertEqual(word, (cre0, cre0, cre1))


class TestNumberStateSourceKetKernel(unittest.TestCase):
    @patch("symop.polynomial.kernels.devices.number_state_source.KetPolyState.from_creators")
    def test_emits_expected_number_state(
        self,
        mock_from_creators: MagicMock,
    ) -> None:
        cre = MagicMock()
        mode = MagicMock()
        mode.signature = "m0"
        mode.cre = cre
        cre.kind = OperatorKind.CRE

        action = SimpleNamespace(
            params={
                "source_modes": [mode],
                "excitations_by_mode": {"m0": 2},
            }
        )

        out = number_state_source_poly_ket(
            state=MagicMock(),
            action=action,
            ctx=MagicMock(),
        )

        self.assertIs(out, mock_from_creators.return_value)

        args, kwargs = mock_from_creators.call_args
        self.assertEqual(args[0], (cre, cre))
        self.assertEqual(kwargs["coeff"], 1.0 / math.sqrt(2))
    def test_creator_word_raises_for_non_creation_operator(self) -> None:
        op = MagicMock()

        mode = MagicMock()
        mode.signature = "m0"
        mode.cre = op
        op.kind = object()

        with self.assertRaises(ValueError):
            _creator_word_for_number_state(
                source_modes=(mode,),
                excitations_by_mode={"m0": 1},
            )
