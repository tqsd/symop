from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from symop_proto.core.protocols import ModeOpProto
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.maps.base import GaussianSubsetMap
from symop_proto.gaussian.ops.passive import apply_passive_unitary_subset


@dataclass(frozen=True)
class PassiveUnitary(GaussianSubsetMap):
    r"""Passive linear map on chosen mode subset.

    This map represent a passive (photon-number preserving) linear
    optical transformation acting on a selected subspace of the
    global mode basis.

    Mathematical model
    ------------------

    Let :math:`a` be the column vector of annihilation operators for
    the full basis in ``core.basis`` and let ``S`` be the subset of
    indices corresponding to ``modes``. A passive device acts in the
    Heisenberg picture as

    .. math::

        a_S \mapsto U a_S

    where :math:`U \in \mathbb{C}^{k\times k}` is unitary and modes
    outside :math:`S` are unchanged.

    In the Schrödinger picture, the Gaussian moments transform as

    .. math::

        \alpha' = U_{\mathrm{full}} \alpha,\quad
        N'=U_{\mathrm{full}} N U_{\mathrm{full}}^\dagger,\quad
        M' = U_{\mathrm{full}} M U_{\mathrm{full}}^\dagger,

    where :math:`U_{mathrm{full}}` is identity on untouched modes and
    equals :math:`U` on the selected subspace.

    These formulas update not only the local blocks :math:`N_{S,S}` and
    :math:`M_{S,S}`, but also all cross-corelation blocks. This makes
    the update correct when the global state is entangled and only
    a subsystem is acted upon.

    Parameters
    ----------
    modes:
        Ordered tuple of modes defining the acted-on subspace. The ordering
        ``modes`` defines the ordering of indices used with ``U``
    U:
        Unitary mixing matrix of shape ``(k,k)`` where ``k=len(modes)``.
    check_unitary:
        If True, validate that ``U`` is unitary before applying it.
    atol:
        Absolute tolerance for unitary checks and for numerical invariant
        stabilization inside the kernel.

    Notes
    -----
    This class is a thin wrapper around
    :func:`symop_proto.gaussian.ops.apply_passive_unitary_subset`.


    Examples
    --------
    Rotate two modes (a simple real mixing):

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.passive import PassiveUnitary

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(
            PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(
            PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.coherent(B, np.array([1.0 + 0.0j, 0.0 + 0.0j]))

        theta = np.pi / 4.0
        U = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=complex)

        op = PassiveUnitary(modes=(m1, m2), U=U, check_unitary=True)
        core2 = op.apply(core)

        print("alpha_in:", core.alpha)
        print("alpha_out:", core2.alpha)
        print("expected:", U @ core.alpha)

    """

    modes: tuple[ModeOpProto, ...]
    U: np.ndarray
    check_unitary: bool = False
    atol: float = 1e-12

    def apply(self, core: GaussianCore) -> GaussianCore:
        idx = self._idx(core)
        return apply_passive_unitary_subset(
            core,
            idx=idx,
            U=self.U,
            check_unitary=self.check_unitary,
            atol=self.atol,
        )


@dataclass(frozen=True)
class PhaseShift(GaussianSubsetMap):
    r"""Single-mode phase shift.

    This map represents the passive transformation

    .. math::

        a \mapsto e^{i\phi} a,

    applied to one selected mode.

    Moment transformation
    ---------------------

    On the acted mode index :math:`i`, the moments transform as:

    .. math::

        \alpha_i' = e^{i\phi}\alpha_i,

    .. math::

        N_{ij}' = e^{-i\phi} N_{ij} \quad (i \text{ in creation index}), \qquad
        N_{ji}' = e^{i\phi} N_{ji},

    and the pairing correlations pick up phase factors on the corresponding
    indices, e.g.

    .. math::

        M_{ij}' = e^{i\phi} M_{ij} \quad \text{if only } i \text{ is acted on}.

    Parameters
    ----------
    mode:
        Mode to phase shift.
    phi:
        Phase shift angle in radians.
    check_unitary:
        If True, check the internally built ``U`` is unitary.
    atol:
        Numerical tolerance (unitary check + stabilization).

    Examples
    --------
    Apply a phase shift to one mode of a two-mode squeezed state:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.passive import PhaseShift

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(
            PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(
            PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        r = 0.7
        s2 = np.sinh(r) ** 2
        sc = np.sinh(r) * np.cosh(r)

        alpha0 = np.zeros((2,), dtype=complex)
        N = np.array([[s2, 0.0], [0.0, s2]], dtype=complex)
        M = np.array([[0.0, sc], [sc, 0.0]], dtype=complex)

        core = GaussianCore.from_moments(B, alpha=alpha0, N=N, M=M)

        phi = 0.3
        core2 = PhaseShift(mode=m1, phi=phi, check_unitary=True).apply(core)

        print("M12 in:", core.M[0, 1])
        print("M12 out:", core2.M[0, 1])
        print("expected:", np.exp(1j * phi) * core.M[0, 1])

    """

    mode: ModeOpProto
    phi: float
    check_unitary: bool = False
    atol: float = 1e-12

    @property
    def modes(self) -> tuple[ModeOpProto, ...]:
        return (self.mode,)

    @property
    def U(self) -> np.ndarray:
        return np.array([[np.exp(1j * float(self.phi))]], dtype=complex)

    def apply(self, core: GaussianCore) -> GaussianCore:
        idx = self._idx(core)
        return apply_passive_unitary_subset(
            core,
            idx=idx,
            U=self.U,
            check_unitary=self.check_unitary,
            atol=self.atol,
        )


@dataclass(frozen=True)
class BeamSplitter(GaussianSubsetMap):
    r"""Two-mode beam splitter.

    This map mixes two selected modes by a passive unitary transformation.

    Convention
    ----------

    The implemented unitary is

    .. math::

        \begin{pmatrix}
        a_1' \\
        a_2'
        \end{pmatrix}
        =
        \begin{pmatrix}
        \cos(\theta) & -e^{i\phi}\sin(\theta) \\
        e^{-i\phi}\sin(\theta) & \cos(\theta)
        \end{pmatrix}
        \begin{pmatrix}
        a_1 \\
        a_2
        \end{pmatrix}.

    This is unitary for all real :math:`\theta` and :math:`\phi`, and corresponds
    to a beam splitter with power transmissivity :math:`T = \cos^2(\theta)`.

    Parameters
    ----------
    mode1, mode2:
        The two modes to mix. The ordering defines the correspondence to
        :math:`(a_1, a_2)` in the matrix above.
    theta:
        Mixing angle. :math:`\theta=\pi/4` corresponds to a 50/50 beam splitter.
    phi:
        Relative phase convention between the ports.
    check_unitary:
        If True, validate the internal matrix is unitary.
    atol:
        Numerical tolerance.

    Examples
    --------
    50/50 beam splitter acting on a coherent input in mode 1:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.maps.passive import BeamSplitter

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([m1, m2])

        core = GaussianCore.coherent(B, np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex))

        bs = BeamSplitter(mode1=m1, mode2=m2, theta=np.pi / 4.0, phi=0.0, check_unitary=True)
        core2 = bs.apply(core)

        print("alpha_out:", core2.alpha)
        print("expected:", bs.U @ core.alpha)

    """

    mode1: ModeOpProto
    mode2: ModeOpProto
    theta: float
    phi: float = 0.0
    check_unitary: bool = False
    atol: float = 1e-12

    @property
    def modes(self) -> tuple[ModeOpProto, ...]:
        return (self.mode1, self.mode2)

    @property
    def U(self) -> np.ndarray:
        th = float(self.theta)
        ph = float(self.phi)
        c = np.cos(th)
        s = np.sin(th)
        e = np.exp(1j * ph)
        return np.array([[c, -e * s], [np.conjugate(e) * s, c]], dtype=complex)

    def apply(self, core: GaussianCore) -> GaussianCore:
        idx = self._idx(core)
        return apply_passive_unitary_subset(
            core,
            idx=idx,
            U=self.U,
            check_unitary=self.check_unitary,
            atol=self.atol,
        )
