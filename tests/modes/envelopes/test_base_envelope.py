import unittest

import numpy as np

from symop.modes.envelopes.base import BaseEnvelope


class StubEnvelope(BaseEnvelope):
    def __init__(
        self,
        amplitude: complex = 1.0 + 0.0j,
        center: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        object.__setattr__(self, "_amplitude", amplitude)
        object.__setattr__(self, "_center", center)
        object.__setattr__(self, "_scale", scale)

    def time_eval(self, t):
        x = (t - self._center) / self._scale
        return self._amplitude * np.exp(-(x**2))

    def freq_eval(self, w):
        return np.exp(-(w**2))

    def delayed(self, dt: float):
        return type(self)(
            amplitude=self._amplitude,
            center=self._center + dt,
            scale=self._scale,
        )

    def phased(self, dphi: float):
        return type(self)(
            amplitude=self._amplitude * np.exp(1j * dphi),
            center=self._center,
            scale=self._scale,
        )

    @property
    def signature(self):
        return ("stub_env", self._amplitude, self._center, self._scale)

    def approx_signature(
        self, *, decimals: int = 12, ignore_global_phase: bool = False
    ):
        amp = self._amplitude
        if ignore_global_phase:
            amp = abs(amp) + 0.0j
        return (
            "stub_env_approx",
            round(amp.real, decimals),
            round(amp.imag, decimals),
            round(self._center, decimals),
            round(self._scale, decimals),
        )

    def center_and_scale(self):
        return self._center, self._scale


class TestBaseEnvelope(unittest.TestCase):
    def assertComplexAlmostEqual(
        self,
        left: complex,
        right: complex,
        places: int = 10,
    ) -> None:
        self.assertAlmostEqual(left.real, right.real, places=places)
        self.assertAlmostEqual(left.imag, right.imag, places=places)

    def test_default_center_and_scale_on_base_stub(self) -> None:
        env = StubEnvelope()
        self.assertEqual(env.center_and_scale(), (0.0, 1.0))

    def test_overlap_of_identical_stub_envelopes_is_positive(self) -> None:
        env = StubEnvelope()
        value = env.overlap(env)
        self.assertGreater(value.real, 0.0)
        self.assertAlmostEqual(value.imag, 0.0, places=10)

    def test_norm2_matches_self_overlap_real_part(self) -> None:
        env = StubEnvelope()
        overlap = env.overlap(env)
        self.assertAlmostEqual(env.norm2(), overlap.real, places=8)

    def test_delayed_returns_shifted_copy(self) -> None:
        env = StubEnvelope(center=1.0, scale=2.0)
        shifted = env.delayed(0.5)

        self.assertEqual(shifted.center_and_scale(), (1.5, 2.0))
        self.assertEqual(env.center_and_scale(), (1.0, 2.0))

    def test_phased_changes_phase_but_not_norm(self) -> None:
        env = StubEnvelope(amplitude=1.0 + 0.0j)
        phased = env.phased(0.7)

        self.assertAlmostEqual(env.norm2(), phased.norm2(), places=8)

    def test_overlap_is_conjugate_symmetric_for_stub_envelopes(self) -> None:
        env1 = StubEnvelope(amplitude=1.0 + 0.5j, center=0.0, scale=1.0)
        env2 = StubEnvelope(amplitude=0.3 - 0.2j, center=0.4, scale=1.5)

        left = env1.overlap(env2)
        right = env2.overlap(env1)

        self.assertComplexAlmostEqual(left, right.conjugate(), places=8)
