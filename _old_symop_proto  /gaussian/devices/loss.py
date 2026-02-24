from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from symop_proto.devices.io import DeviceIO, DeviceResult
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.devices.base import GaussianDevice
from symop_proto.gaussian.devices._utils import select_modes, require_nonempty
from symop_proto.gaussian.maps.channel import PureLoss


@dataclass(frozen=True)
class PathPureLoss(GaussianDevice):
    r"""
    Apply the same pure-loss channel to all modes whose PathLabel matches ``path``.

    Mathematical model
    ------------------

    For each selected mode :math:`a`, the quantum-limited attenuator acts as

    .. math::

        a' = \sqrt{\eta}\, a + \sqrt{1-\eta}\, e,

    where :math:`e` is a vacuum environment mode and :math:`0\le \eta\le 1`.
    Consequently,

    .. math::

        \langle a\rangle' = \sqrt{\eta}\,\langle a\rangle,

    and the mean photon number scales as

    .. math::

        \langle a^\dagger a\rangle' = \eta \,\langle a^\dagger a\rangle.

    Examples
    --------

    Example 1: Photon number on a path scales by ``eta``.
    ====================================================

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.evaluator import GaussianEvaluator
        from symop_proto.gaussian.devices.loss import PathPureLoss

        env = GaussianEnvelope(omega0=1.0, sigma=0.5, tau=0.0, phi0=0.0)

        # Two modes on path A (H/V) + one mode on path B
        mAH = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        mAV = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.V()))
        mBH = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))

        B = ModeBasis.build([mAH, mAV, mBH])

        # coherent in all three modes
        alpha = np.array([1.0 + 0.0j, 0.5 + 0.2j, 2.0 + 0.0j], dtype=complex)
        core = GaussianCore.coherent(B, alpha)

        ev0 = GaussianEvaluator(core)

        nA0 = ev0.number(mAH) + ev0.number(mAV)
        nB0 = ev0.number(mBH)

        eta = 0.25
        res = PathPureLoss(path=PathLabel("A"), eta=eta).apply(core)
        core2 = res.state

        ev1 = GaussianEvaluator(core2)

        nA1 = ev1.number(mAH) + ev1.number(mAV)
        nB1 = ev1.number(mBH)

        print("n(A) before:", nA0)
        print("n(A) after :", nA1)
        print("expected  :", eta * nA0)

        print("n(B) before:", nB0)
        print("n(B) after :", nB1)
        print("expected   : unchanged")

    Example 2: Mean field amplitude scales by ``sqrt(eta)`` on the path.
    ===================================================================

    .. jupyter-execute::

        import numpy as np

        from symop_proto.gaussian.evaluator import GaussianEvaluator
        from symop_proto.gaussian.devices.loss import PathPureLoss
        from symop_proto.labels.path_label import PathLabel

        # Reuse core, mAH, mAV, mBH from Example 1

        eta = 0.36
        core2 = PathPureLoss(path=PathLabel("A"), eta=eta).apply(core).state

        ev0 = GaussianEvaluator(core)
        ev1 = GaussianEvaluator(core2)

        s = np.sqrt(eta)

        print("<a_AH> before:", ev0.mean(mAH))
        print("<a_AH> after :", ev1.mean(mAH))
        print("expected     :", s * ev0.mean(mAH))

        print("<a_BH> before:", ev0.mean(mBH))
        print("<a_BH> after :", ev1.mean(mBH))
        print("expected     : unchanged")

    Example 3: "Energy" proxy before/after loss using evaluator.
    ============================================================

    Here we define two diagnostics for the set of modes on path A:

    - photon-number proxy: :math:`E_\mathrm{phot} = \sum_i \langle n_i\rangle`
    - frequency-weighted proxy (units :math:`\hbar=1`):
      :math:`E_\omega = \sum_i \omega_i \langle n_i\rangle`

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.evaluator import GaussianEvaluator
        from symop_proto.gaussian.devices.loss import PathPureLoss

        # two modes on the same path with different carrier frequencies
        env1 = GaussianEnvelope(omega0=1.0, sigma=0.5, tau=0.0, phi0=0.0)
        env2 = GaussianEnvelope(omega0=2.0, sigma=0.5, tau=0.0, phi0=0.0)

        m1 = ModeOp(env=env1, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env2, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))

        B = ModeBasis.build([m1, m2])
        core = GaussianCore.coherent(B, np.array([1.0 + 0.0j, 0.5 + 0.0j], dtype=complex))

        def energy_proxies(ev):
            n1 = ev.number(m1)
            n2 = ev.number(m2)
            E_phot = n1 + n2
            E_w = float(m1.env.omega0) * n1 + float(m2.env.omega0) * n2
            return E_phot, E_w

        ev0 = GaussianEvaluator(core)
        E0, Ew0 = energy_proxies(ev0)

        eta = 0.5
        core2 = PathPureLoss(path=PathLabel("A"), eta=eta).apply(core).state
        ev1 = GaussianEvaluator(core2)
        E1, Ew1 = energy_proxies(ev1)

        print("E_phot before:", E0)
        print("E_phot after :", E1)
        print("expected     :", eta * E0)

        print("E_w before:", Ew0)
        print("E_w after :", Ew1)
        print("expected  :", eta * Ew0)

    """

    path: object
    eta: float
    pol: Optional[object] = None
    allow_empty: bool = False

    def __post_init__(self) -> None:
        # Initialize BaseDevice internals for frozen dataclass subclasses.
        self._init_base()

    def resolve_io(self, state: GaussianCore) -> DeviceIO:
        modes = select_modes(state, path=self.path, pol=self.pol)

        if not self.allow_empty:
            modes = require_nonempty(modes, what="PathPureLoss")

        return DeviceIO(
            input_modes=modes,
            output_modes=modes,  # in-place
            env_modes=(),
            meta={"eta": float(self.eta)},
        )

    def do_apply(self, state: GaussianCore, io: DeviceIO) -> GaussianCore:
        # io.input_modes already resolved from the state basis
        return PureLoss(modes=io.input_modes, eta=self.eta).apply(state)

    def _apply_gaussian(
        self, state: GaussianCore, *, options=None
    ) -> DeviceResult[GaussianCore]:
        io = self.resolve_io(state)
        out = self.do_apply(state, io)
        return DeviceResult(state=out, io=io)
