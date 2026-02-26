from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from symop_proto.core.protocols import ModeOpProto
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.maps.base import GaussianSubsetMap
from symop_proto.gaussian.ops.bogoliubov import apply_bogoliubov_subset


@dataclass(frozen=True)
class Bogoliubov(GaussianSubsetMap):
    r"""Linear Bogoliubov map on a chosen mode subset.

    This map represents the most general linear CCR-preserving transformation
    on ladder operators (Gaussian-preserving in the Schrödinger picture):

    .. math::

        a_S \mapsto U a_S + V a_S^\dagger,

    acting on a selected subset of modes ``modes``. Modes outside the subset
    are unchanged.

    What this enables
    -----------------

    * **Squeezing and parametric processes**: single-mode and two-mode
      squeezing are Bogoliubov maps with nonzero ``V``.
    * **Gaussian unitaries in general**: any linear symplectic transformation
      can be expressed via some pair ``(U,V)``.
    * **Amplifier / loss models via auxiliary modes**: by extending the basis
      with environmental vacuum modes, coupling + tracing produces standard
      Gaussian channels (amplification requires ``V != 0`` at the joint-unitary
      level).

    Moment update
    -------------

    The map updates the Gaussian core by transforming first/second moments.
    Importantly, it updates *all* correlation blocks, including
    cross-correlations with untouched modes, because it is applied as a
    full-matrix action with ``U_full`` and ``V_full`` embedded into the
    global basis.

    Parameters
    ----------
    modes:
        Ordered tuple of acted-on modes. The ordering defines the index
        ordering used with matrices ``U`` and ``V``.
    U, V:
        Complex matrices of shape ``(k,k)``, ``k=len(modes)``.
    check_ccr:
        If True, validate CCR preservation using the Gram matrix.
    atol:
        Numerical tolerance for CCR checks and stabilization.

    Examples
    --------
    Single-mode squeezing on vacuum (canonical single-mode basis):

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.bogoliubov import Bogoliubov

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])

        core = GaussianCore.vacuum(B)

        r = 0.6
        phi = 0.1
        ch = np.cosh(r)
        sh = np.sinh(r)

        U = np.array([[ch]], dtype=complex)
        V = np.array([[np.exp(1j * phi) * sh]], dtype=complex)

        op = Bogoliubov(modes=(m,), U=U, V=V, check_ccr=True)
        core2 = op.apply(core)

        print("N_out:", core2.N[0, 0].real)
        print("M_out:", core2.M[0, 0])

    Two-mode squeezing between two modes:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.bogoliubov import Bogoliubov

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.vacuum(B)

        r = 0.4
        ch = np.cosh(r)
        sh = np.sinh(r)

        U = np.array([[ch, 0.0], [0.0, ch]], dtype=complex)
        V = np.array([[0.0, sh], [sh, 0.0]], dtype=complex)

        op = Bogoliubov(modes=(m1, m2), U=U, V=V, check_ccr=True)
        core2 = op.apply(core)

        print("N diag:", np.diag(core2.N).real)
        print("M offdiag:", core2.M[0, 1])

    """

    modes: tuple[ModeOpProto, ...]
    U: np.ndarray
    V: np.ndarray
    check_ccr: bool = True
    atol: float = 1e-12

    def apply(self, core: GaussianCore) -> GaussianCore:
        idx = self._idx(core)
        return apply_bogoliubov_subset(
            core,
            idx=idx,
            U=self.U,
            V=self.V,
            check_ccr=self.check_ccr,
            atol=self.atol,
        )


@dataclass(frozen=True)
class SingleModeSqueezer(GaussianSubsetMap):
    r"""Canonical single-mode squeezing map.

    Convention
    ----------
    Implements the Heisenberg-picture transform

    .. math::

        a' = \cosh(r)\, a + e^{i\phi}\sinh(r)\, a^\dagger.

    Parameters
    ----------
    mode:
        Mode to squeeze.
    r:
        Squeezing parameter.
    phi:
        Squeezing phase.
    check_ccr:
        If True, validate CCR preservation using the Gram matrix.
    atol:
        Numerical tolerance.

    Examples
    --------
    Squeeze vacuum and compare moments:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.bogoliubov import SingleModeSqueezer

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])

        core = GaussianCore.vacuum(B)

        r = 0.6
        phi = 0.1
        core2 = SingleModeSqueezer(mode=m, r=r, phi=phi, check_ccr=True).apply(core)

        sh = np.sinh(r)
        ch = np.cosh(r)

        print("N_out:", core2.N[0, 0].real, "expected:", sh * sh)
        print("M_out:", core2.M[0, 0], "expected:", np.exp(1j * phi) * sh * ch)

    """

    mode: ModeOpProto
    r: float
    phi: float = 0.0
    check_ccr: bool = False
    atol: float = 1e-12

    @property
    def modes(self) -> tuple[ModeOpProto, ...]:
        return (self.mode,)

    @property
    def U(self) -> np.ndarray:
        ch = np.cosh(float(self.r))
        return np.array([[ch]], dtype=complex)

    @property
    def V(self) -> np.ndarray:
        sh = np.sinh(float(self.r))
        ph = float(self.phi)
        return np.array([[np.exp(1j * ph) * sh]], dtype=complex)

    def apply(self, core: GaussianCore) -> GaussianCore:
        return Bogoliubov(
            modes=self.modes,
            U=self.U,
            V=self.V,
            check_ccr=self.check_ccr,
            atol=self.atol,
        ).apply(core)


@dataclass(frozen=True)
class TwoModeSqueezer(GaussianSubsetMap):
    r"""Canonical two-mode squeezing map.

    Convention
    ----------
    Implements

    .. math::

        a_1' = \cosh(r)\, a_1 + e^{i\phi}\sinh(r)\, a_2^\dagger, \\
        a_2' = \cosh(r)\, a_2 + e^{i\phi}\sinh(r)\, a_1^\dagger.

    This is the common entangling (non-degenerate) parametric amplifier
    convention.

    Parameters
    ----------
    mode1, mode2:
        The two modes to entangle.
    r:
        Squeezing parameter.
    phi:
        Phase factor on the creation-mixing term.
    check_ccr:
        If True, validate CCR preservation.
    atol:
        Numerical tolerance.

    Examples
    --------
    Apply two-mode squeezing to vacuum:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.bogoliubov import TwoModeSqueezer

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.vacuum(B)

        r = 0.4
        phi = 0.0
        core2 = TwoModeSqueezer(mode1=m1, mode2=m2, r=r, phi=phi, check_ccr=True).apply(core)

        sh = np.sinh(r)
        ch = np.cosh(r)
        print("N diag:", np.diag(core2.N).real, "expected:", sh * sh)
        print("M12:", core2.M[0, 1], "expected:", np.exp(1j * phi) * sh * ch)

    """

    mode1: ModeOpProto
    mode2: ModeOpProto
    r: float
    phi: float = 0.0
    check_ccr: bool = False
    atol: float = 1e-12

    @property
    def modes(self) -> tuple[ModeOpProto, ...]:
        return (self.mode1, self.mode2)

    @property
    def U(self) -> np.ndarray:
        ch = np.cosh(float(self.r))
        return np.array([[ch, 0.0], [0.0, ch]], dtype=complex)

    @property
    def V(self) -> np.ndarray:
        sh = np.sinh(float(self.r))
        ph = float(self.phi)
        e = np.exp(1j * ph) * sh
        return np.array([[0.0, e], [e, 0.0]], dtype=complex)

    def apply(self, core: GaussianCore) -> GaussianCore:
        return Bogoliubov(
            modes=self.modes,
            U=self.U,
            V=self.V,
            check_ccr=self.check_ccr,
            atol=self.atol,
        ).apply(core)


@dataclass(frozen=True)
class TwoModeAmplifierUnitary(GaussianSubsetMap):
    r"""Two-mode Bogoliubov unitary for a phase-insensitive amplifier.

    This is the canonical "signal + idler" unitary that realizes a quantum-limited
    phase-insensitive amplifier when the idler starts in vacuum.

    Convention
    ----------
    With gain parameter ``r`` (sometimes expressed via power gain ``g``),
    implement

    .. math::

        a_s' = \cosh(r)\, a_s + e^{i\phi}\sinh(r)\, a_i^\dagger, \\
        a_i' = \cosh(r)\, a_i + e^{i\phi}\sinh(r)\, a_s^\dagger.

    If the idler is vacuum and later discarded, the signal undergoes amplification
    with gain

    .. math::

        g = \cosh^2(r).

    Parameters
    ----------
    signal, idler:
        Modes playing the role of signal and idler.
    r:
        Gain parameter (g = cosh(r)^2).
    phi:
        Phase on the creation-mixing term.
    check_ccr:
        If True, validate CCR preservation.
    atol:
        Numerical tolerance.

    Examples
    --------
    Amplify a coherent signal by coupling to vacuum idler:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.bogoliubov import TwoModeAmplifierUnitary

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        s = ModeOp(env=env, label=ModeLabel(PathLabel("S"), PolarizationLabel.H()))
        i = ModeOp(env=env, label=ModeLabel(PathLabel("I"), PolarizationLabel.H()))
        B = ModeBasis.build([s, i])

        # coherent on signal, vacuum on idler
        alpha = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
        core = GaussianCore.coherent(B, alpha)

        r = 0.7
        phi = 0.0
        core2 = TwoModeAmplifierUnitary(signal=s, idler=i, r=r, phi=phi, check_ccr=True).apply(core)

        g = np.cosh(r) ** 2
        print("signal mean in:", core.alpha[0])
        print("signal mean out:", core2.alpha[0])
        print("expected scaling ~ cosh(r):", np.cosh(r) * core.alpha[0])
        print("power gain g:", g)

    """

    signal: ModeOpProto
    idler: ModeOpProto
    r: float
    phi: float = 0.0
    check_ccr: bool = False
    atol: float = 1e-12

    @property
    def modes(self) -> tuple[ModeOpProto, ...]:
        return (self.signal, self.idler)

    @property
    def U(self) -> np.ndarray:
        ch = np.cosh(float(self.r))
        return np.array([[ch, 0.0], [0.0, ch]], dtype=complex)

    @property
    def V(self) -> np.ndarray:
        sh = np.sinh(float(self.r))
        ph = float(self.phi)
        e = np.exp(1j * ph) * sh
        return np.array([[0.0, e], [e, 0.0]], dtype=complex)

    def apply(self, core: GaussianCore) -> GaussianCore:
        return Bogoliubov(
            modes=self.modes,
            U=self.U,
            V=self.V,
            check_ccr=self.check_ccr,
            atol=self.atol,
        ).apply(core)
