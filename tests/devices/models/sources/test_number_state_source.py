from __future__ import annotations

import unittest

from symop.devices.models.sources.number_state_source import NumberStateSource
from symop.devices.types.device_kind import DeviceKind
from tests.core.support.fakes import FakeComponentLabel
from tests.devices.support.fakes import FakePath, FakeState


class TestNumberStateSource(unittest.TestCase):
    def test_init_rejects_negative_n(self) -> None:
        envelope = FakeComponentLabel("envelope", "env")
        polarization = FakeComponentLabel("polarization", "pol")

        with self.assertRaises(ValueError):
            NumberStateSource(
                envelope=envelope,
                polarization=polarization,
                n=-1,
            )

    def test_kind(self) -> None:
        device = NumberStateSource(
            envelope=FakeComponentLabel("envelope", "env"),
            polarization=FakeComponentLabel("polarization", "pol"),
            n=2,
        )
        self.assertEqual(device.kind, DeviceKind.NUMBER_STATE_SOURCE)

    def test_port_specs(self) -> None:
        device = NumberStateSource(
            envelope=FakeComponentLabel("envelope", "env"),
            polarization=FakeComponentLabel("polarization", "pol"),
            n=2,
        )

        self.assertEqual(len(device.port_specs), 1)
        self.assertEqual(device.port_specs[0].name, "out")
        self.assertEqual(device.port_specs[0].direction, "out")
        self.assertTrue(device.port_specs[0].required)

    def test_plan_returns_source_mode_and_excitation_map(self) -> None:
        envelope = FakeComponentLabel("envelope", "env")
        polarization = FakeComponentLabel("polarization", "pol")
        device = NumberStateSource(
            envelope=envelope,
            polarization=polarization,
            n=3,
        )
        out = FakePath("out0")

        action = device.plan(
            state=FakeState(),
            ports={"out": out},
            selection={"tag": "keep"},
            ctx=None,
        )

        self.assertEqual(action.kind, DeviceKind.NUMBER_STATE_SOURCE)
        self.assertEqual(action.ports["out"], out)
        self.assertEqual(action.selection, {"tag": "keep"})
        self.assertEqual(tuple(action.edits), ())

        source_modes = action.params["source_modes"]
        excitations_by_mode = action.params["excitations_by_mode"]

        self.assertEqual(len(source_modes), 1)
        mode = source_modes[0]

        self.assertEqual(mode.label.path, out)
        self.assertEqual(mode.label.envelope, envelope)
        self.assertEqual(mode.label.polarization, polarization)
        self.assertEqual(excitations_by_mode[mode.signature], 3)
