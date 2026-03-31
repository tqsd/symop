from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from symop.devices.models.filters.spectral_filter import SpectralFilter
from symop.devices.types.device_kind import DeviceKind
from tests.core.support.fakes import FakeComponentLabel, make_mode
from tests.devices.support.fakes import (
    FakeTimeFrequencyEnvelope,
    FakeEditableState,
    FakePath,
    FakeState,
)


@dataclass(frozen=True)
class FakeTransfer:
    name: str = "tf"

    @property
    def signature(self) -> tuple[str, str]:
        return ("fake_transfer", self.name)


class TestSpectralFilter(unittest.TestCase):
    def test_kind(self) -> None:
        device = SpectralFilter(transfer=FakeTransfer())
        self.assertEqual(device.kind, DeviceKind.SPECTRAL_FILTER)

    def test_port_specs(self) -> None:
        device = SpectralFilter(transfer=FakeTransfer())

        self.assertEqual(len(device.port_specs), 2)
        self.assertEqual(device.port_specs[0].name, "in")
        self.assertEqual(device.port_specs[1].name, "out")

    def test_plan_requires_label_editable_state(self) -> None:
        device = SpectralFilter(transfer=FakeTransfer())

        with self.assertRaises(TypeError):
            device.plan(
                state=FakeState(),
                ports={"in": FakePath("in0"), "out": FakePath("out0")},
                selection=None,
                ctx=None,
            )

    def test_plan_rejects_incompatible_envelope(self) -> None:
        in_path = FakeComponentLabel("path", "in0")
        out_path = FakeComponentLabel("path", "out0")

        mode = make_mode(path="in0", polarization="pol", envelope="env")
        state = FakeEditableState(
            mode_labels={mode.signature: mode.label},
            modes=(mode,),
        )
        device = SpectralFilter(transfer=FakeTransfer())

        with self.assertRaises(TypeError):
            device.plan(
                state=state,
                ports={"in": in_path, "out": out_path},
                selection=None,
                ctx=None,
            )

    @patch("symop.devices.models.filters.spectral_filter.apply_transfer")
    def test_plan_records_eta_and_emits_edit_when_eta_is_nonzero(
        self,
        mock_apply_transfer,
    ) -> None:
        in_path = FakeComponentLabel("path", "in0")
        out_path = FakeComponentLabel("path", "out0")

        env_in = FakeTimeFrequencyEnvelope("env", tau=0.0)
        env_out = FakeTimeFrequencyEnvelope("env_filtered", tau=0.0)
        mock_apply_transfer.return_value = (env_out, 0.25)

        mode = make_mode(path="in0", polarization="pol", envelope="unused")
        mode = mode.with_label(mode.label.with_envelope(env_in))

        state = FakeEditableState(
            mode_labels={mode.signature: mode.label},
            modes=(mode,),
        )

        device = SpectralFilter(transfer=FakeTransfer())

        action = device.plan(
            state=state,
            ports={"in": in_path, "out": out_path},
            selection=None,
            ctx=None,
        )

        eta_by_mode = action.params["eta_by_mode"]
        self.assertAlmostEqual(eta_by_mode[mode.signature], 0.25)
        self.assertEqual(len(action.edits), 1)

        edit = action.edits[0]
        self.assertEqual(edit.mode_sig, mode.signature)
        self.assertEqual(edit.label.path, out_path)
        self.assertEqual(edit.label.envelope, env_out)

    @patch("symop.devices.models.filters.spectral_filter.apply_transfer")
    def test_plan_skips_edit_when_eta_is_zero(self, mock_apply_transfer) -> None:
        in_path = FakeComponentLabel("path", "in0")
        out_path = FakeComponentLabel("path", "out0")

        env_in = FakeTimeFrequencyEnvelope("env", tau=0.0)
        env_out = FakeTimeFrequencyEnvelope("env_filtered", tau=0.0)
        mock_apply_transfer.return_value = (env_out, 0.0)

        mode = make_mode(path="in0", polarization="pol", envelope="unused")
        mode = mode.with_label(mode.label.with_envelope(env_in))

        state = FakeEditableState(
            mode_labels={mode.signature: mode.label},
            modes=(mode,),
        )

        device = SpectralFilter(transfer=FakeTransfer())

        action = device.plan(
            state=state,
            ports={"in": in_path, "out": out_path},
            selection=None,
            ctx=None,
        )

        eta_by_mode = action.params["eta_by_mode"]
        self.assertAlmostEqual(eta_by_mode[mode.signature], 0.0)
        self.assertEqual(tuple(action.edits), ())

    def test_apply_forces_density_out_kind(self) -> None:
        class RecordingRuntime:
            def __init__(self) -> None:
                self.calls = []

            def apply(
                self,
                *,
                device: object,
                state: object,
                ports: object,
                selection: object,
                ctx: object,
                out_kind: object,
            ) -> object:
                self.calls.append({"out_kind": out_kind})
                return "ok"

        rt = RecordingRuntime()
        device = SpectralFilter(transfer=FakeTransfer())

        result = device.apply(
            FakeState(),
            ports={"in": FakePath("in0"), "out": FakePath("out0")},
            runtime=rt,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(rt.calls[0]["out_kind"], "density")
