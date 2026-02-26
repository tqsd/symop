import numpy as np
import pytest

from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.labels.path_label import PathLabel
from symop_proto.labels.polarization_label import PolarizationLabel
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.core.operators import ModeOp
from symop_proto.gaussian.basis import ModeBasis
from symop_proto.gaussian.core import GaussianCore

from symop_proto.gaussian.maps.channel import (
    Displacement,
    PureLoss,
    ThermalLoss,
    PhaseInsensitiveAmplifier,
)


def single_mode_core():
    env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
    m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
    B = ModeBasis.build([m])
    return m, B


def test_displacement_on_vacuum():
    m, B = single_mode_core()
    core = GaussianCore.vacuum(B)

    beta = np.array([1.2 + 0.5j])
    core2 = Displacement(modes=(m,), beta=beta).apply(core)

    assert np.allclose(core2.alpha, beta)
    assert np.allclose(core2.centered_moments()[0], 0.0)
    assert np.allclose(core2.centered_moments()[1], 0.0)


def test_pure_loss_identity_limit():
    m, B = single_mode_core()
    core = GaussianCore.coherent(B, np.array([1.0 + 0.0j]))

    core2 = PureLoss(modes=(m,), eta=1.0).apply(core)

    assert np.allclose(core2.alpha, core.alpha)
    assert np.allclose(core2.N, core.N)
    assert np.allclose(core2.M, core.M)


def test_pure_loss_scales_coherent():
    m, B = single_mode_core()
    alpha = np.array([2.0 + 0.0j])
    core = GaussianCore.coherent(B, alpha)

    eta = 0.25
    core2 = PureLoss(modes=(m,), eta=eta).apply(core)

    assert np.allclose(core2.alpha, np.sqrt(eta) * alpha)


def test_thermal_loss_adds_noise():
    m, B = single_mode_core()
    core = GaussianCore.vacuum(B)

    eta = 0.5
    nbar = 3.0

    core2 = ThermalLoss(modes=(m,), eta=eta, nbar=nbar).apply(core)

    # For vacuum input:
    # N = (1-eta)*nbar
    expected = (1 - eta) * nbar
    assert np.allclose(core2.N[0, 0].real, expected)
    assert np.allclose(core2.M[0, 0], 0.0)


def test_amplifier_gain_scaling():
    m, B = single_mode_core()
    alpha = np.array([1.5 + 0.0j])
    core = GaussianCore.coherent(B, alpha)

    g = 4.0
    core2 = PhaseInsensitiveAmplifier(modes=(m,), gain=g).apply(core)

    assert np.allclose(core2.alpha, np.sqrt(g) * alpha)


def test_amplifier_vacuum_noise():
    m, B = single_mode_core()
    core = GaussianCore.vacuum(B)

    g = 3.0
    core2 = PhaseInsensitiveAmplifier(modes=(m,), gain=g, nbar=0.0).apply(core)

    # vacuum -> thermal with N = (g-1)
    assert np.allclose(core2.N[0, 0].real, g - 1.0)
    assert np.allclose(core2.M[0, 0], 0.0)
