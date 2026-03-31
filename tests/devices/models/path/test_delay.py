from __future__ import annotations

import unittest

from symop.devices.models.paths.delay import Delay
from symop.devices.types.device_kind import DeviceKind
from tests.core.support.fakes import FakeComponentLabel, make_mode
from tests.devices.support.fakes import FakeTimeFrequencyEnvelope, FakeEditableState, FakeState


class TestDelay(unittest.TestCase):
    def test_kind(self) -> None:
        device = Delay(dt=1.5)
        self.assertEqual(device.kind, DeviceKind.DELAY)

    def test_port_specs(self) -> None:
        device = Delay(dt=1.5)

        self.assertEqual(len(device.port_specs), 2)
        self.assertEqual(device.port_specs[0].name, "in")
        self.assertEqual(device.port_specs[0].direction, "in")
        self.assertEqual(device.port_specs[1].name, "out")
        self.assertEqual(device.port_specs[1].direction, "out")

    def test_plan_requires_label_editable_state(self) -> None:
        device = Delay(dt=2.0)

        with self.assertRaises(TypeError):
            device.plan(
                state=FakeState(),
                ports={
                    "in": FakeComponentLabel("path", "in0"),
                    "out": FakeComponentLabel("path", "out0"),
                },
                selection=None,
                ctx=None,
            )

    def test_plan_returns_edit_only_action(self) -> None:
        in_path = FakeComponentLabel("path", "in0")
        out_path = FakeComponentLabel("path", "out0")

        env = FakeTimeFrequencyEnvelope("env", tau=0.0)
        mode = make_mode(
            path="in0",
            polarization="pol",
            envelope="unused",
        )
        mode = mode.with_label(mode.label.with_envelope(env))

        state = FakeEditableState(
            mode_labels={mode.signature: mode.label},
            modes=(mode,),
        )

        device = Delay(dt=2.5)

        action = device.plan(
            state=state,
            ports={"in": in_path, "out": out_path},
            selection=None,
            ctx=None,
        )

        self.assertEqual(action.kind, DeviceKind.DELAY)
        self.assertFalse(action.requires_kernel)
        self.assertEqual(action.params["dt"], 2.5)
        self.assertEqual(len(action.edits), 1)

        edit = action.edits[0]
        self.assertEqual(edit.mode_sig, mode.signature)
        self.assertEqual(edit.label.path, out_path)
        self.assertEqual(edit.label.envelope.tau, 2.5)

    def test_plan_rejects_non_delayable_envelope(self) -> None:
        in_path = FakeComponentLabel("path", "in0")
        out_path = FakeComponentLabel("path", "out0")

        mode = make_mode(path="in0", polarization="pol", envelope="env")
        state = FakeEditableState(
            mode_labels={mode.signature: mode.label},
            modes=(mode,),
        )
        device = Delay(dt=1.0)

        with self.assertRaises(TypeError):
            device.plan(
                state=state,
                ports={"in": in_path, "out": out_path},
                selection=None,
                ctx=None,
            )
