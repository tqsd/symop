from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from symop_proto.core.protocols import ModeOpProto
from symop_proto.gaussian.basis import ModeBasis


@dataclass(frozen=True)
class GaussianCore:
    r"""
    Gaussian state moments in a (possibly non-orthogonal) mode basis.

    A Gaussian state on a bosonic CCR algebra is fully characterized by its
    first and second moments. The mode operators are allowed to be
    non-orthogonal:

    .. math::

        [a_i, a_j^\dagger] = G_{ij},

    where :math:`G` is the gram matrix provided by
    :class:`symop_proto.gaussian.basis.ModeBasis`.
    This class stores moments with respect to the chose basis:

    .. math::

        \alpha_i &= \langle a_i \rangle, \\
        N_{ij}   &= \langle a_i^\dagger a_j \rangle \\
        M_{ij}   &= \langle a_i a_j \rangle.


    These quantities, together with :math:`G`, determine all higher-order
    moments via Wick's theorem. The non-orthogonal commuatator enters only
    through identities such as

    .. math::

        \langle a_i a_j^\dagger \rangle
        = G_{ij} + \langle a_j^\dagger a_i \rangle
        = G_{ij} + N_{ji}


    Examples
    --------

    Construct a simple orthogonal two-mode basis and initialize the vacuum:

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore

        # Build orthogonal modes (different spatial paths)
        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)

        m1 = ModeOp(env=env,
                    label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env,
                    label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))

        B = ModeBasis.build([m1, m2])

        core = GaussianCore.vacuum(B)

        print("alpha =", core.alpha)
        print("N =")
        print(core.N)
        print("M =")
        print(core.M)
        print("Quadrature covariance =")
        print(core.quadrature_covariance())

    A coherent displacement corresponds to nonzero first moments
    while second moments remain vacuum-like:
    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B2 = ModeBasis.build([m1, m2])

        alpha = np.array([1.0 + 0.3j, -0.4 + 0.2j], dtype=complex)

        core = GaussianCore.coherent(B2, alpha)

        print("alpha =", core.alpha)

        print("Centered moments =")
        N0, M0 = core.centered_moments()
        print("N0 =")
        print(N0)
        print("M0 =")
        print(M0)

        print("Quadrature covariance =")
        print(core.quadrature_covariance())

    Two-mode squeezing produces nonzero pairing correlations ``M``.
    For squeezing parameter :math:`r`, the ideal orthogonal case is

    .. math::

        \alpha = 0,

    .. math::

        N = \sinh^2(r)\, I,

    .. math::

        M = \sinh(r)\cosh(r)
            \begin{pmatrix}
                0 & 1 \\
                1 & 0
            \end{pmatrix}.

    This encodes entanglement between the two modes.

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore

        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B2 = ModeBasis.build([m1, m2])

        alpha = np.zeros((2,), dtype=complex)

        r = 0.7
        s2 = np.sinh(r) ** 2
        sc = np.sinh(r) * np.cosh(r)

        N = np.array([[s2, 0.0],
                    [0.0, s2]], dtype=complex)

        M = np.array([[0.0, sc],
                    [sc, 0.0]], dtype=complex)

        core = GaussianCore.from_moments(B2, alpha=alpha, N=N, M=M)

        print("N =")
        print(core.N)
        print("M =")
        print(core.M)
        print("Quadrature covariance =")
        print(core.quadrature_covariance())

    """

    basis: ModeBasis
    alpha: np.ndarray
    N: np.ndarray
    M: np.ndarray

    def __post_init__(self) -> None:
        n = self.basis.n
        if self.alpha.shape != (n,):
            raise ValueError(f"alpha must have shape({n},), got {self.alpha.shape}")
        if self.N.shape != (n, n):
            raise ValueError(f"N must have shape ({n}, {n}), got {self.N.shape}")
        if self.M.shape != (n, n):
            raise ValueError(f"M must have shape ({n},{n}), got {self.M.shape}")

        if not np.allclose(self.N, self.N.conj().T):
            raise ValueError("N must be Hermitian")

        if not np.allclose(self.M, self.M.T):
            raise ValueError("M must be symmetric")

    @staticmethod
    def vacuum(basis: ModeBasis) -> GaussianCore:
        r"""
        Constructs a vaccum Gaussian core in the given basis.

        The vacuum state is defined by

        .. math::

            \alpha_i = 0

        .. math::

            N_{ij} = 0

        .. math::

            M_{ij} = 0

        for all modes.

        The commutation structure is inherited from the
        :class:`symop_proto.gaussian.basis.ModeBasis` via its
        Gram matrix

        .. math::

            [a_i, a_j^\dagger] = G_{ij}

        Examples
        --------

        Construct vacuum in a simple orthogonal two-mode basis:

        .. jupyter-execute::

            import numpy as np

            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.labels.path_label import PathLabel
            from symop_proto.labels.polarization_label import PolarizationLabel
            from symop_proto.labels.mode_label import ModeLabel
            from symop_proto.core.operators import ModeOp
            from symop_proto.gaussian.basis import ModeBasis
            from symop_proto.gaussian.core import GaussianCore

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)

            m1 = ModeOp(env=env,
                        label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            m2 = ModeOp(env=env,
                        label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))

            B = ModeBasis.build([m1, m2])

            core = GaussianCore.vacuum(B)

            print("alpha =", core.alpha)
            print("Covariance =")
            print(core.quadrature_covariance())

        Returns
        -------
        GaussianCore
            Vacuum Gaussian state in the specified basis.
        """
        n = basis.n
        return GaussianCore(
            basis=basis,
            alpha=np.zeros((n,), dtype=complex),
            N=np.zeros((n, n), dtype=complex),
            M=np.zeros((n, n), dtype=complex),
        )

    @staticmethod
    def from_moments(
        basis: ModeBasis,
        *,
        alpha: Optional[np.ndarray] = None,
        N: Optional[np.ndarray] = None,
        M: Optional[np.ndarray] = None,
    ) -> GaussianCore:
        r"""
            Constructs a Gaussian core from supplied first and second moments.

            A Gaussian state is fully characterized by:

            .. math::

                \alpha_i = \langle a_i \rangle,

            .. math::

                N_{ij} = \langle a_i^\dagger a_j \rangle,

            .. math::

                M_{ij} = \langle a_i a_j \rangle.

            Any omitted moment is initialized to zero.

            Examples
            --------

            Construct a Gaussian state directly from supplied moments:

        .. jupyter-execute::

            import numpy as np

            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.labels.path_label import PathLabel
            from symop_proto.labels.polarization_label import PolarizationLabel
            from symop_proto.labels.mode_label import ModeLabel
            from symop_proto.core.operators import ModeOp
            from symop_proto.gaussian.basis import ModeBasis
            from symop_proto.gaussian.core import GaussianCore

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
            B2 = ModeBasis.build([m1, m2])

            alpha = np.zeros((2,), dtype=complex)

            r = 0.7
            s2 = np.sinh(r) ** 2
            sc = np.sinh(r) * np.cosh(r)

            N = np.array([[s2, 0.0],
                        [0.0, s2]], dtype=complex)

            M = np.array([[0.0, sc],
                        [sc, 0.0]], dtype=complex)

            core = GaussianCore.from_moments(B2, alpha=alpha, N=N, M=M)

            print("N =")
            print(core.N)
            print("M =")
            print(core.M)
            print("Quadrature covariance =")
            print(core.quadrature_covariance())

        Parameters
        ----------
        basis:
            Mode basis defining the operator set and Gram matrix.
        alpha:
            First moment vector of shape ``(n,)``
        N:
            Number correlator matrix of shape ``(n,n)``
        M:
            Pairing correlator matrix of shape ``(n,n)``

        Returns
        -------
        GaussianCore
            Gaussian state with supplied moments.

        Raises
        ------
        ValueError
            If shapes are inconistent or algebraic constraints
            (Hermiticity of N, symmetry of M) are violated.

        Notes
        -----
        No physicality checks (positivity/uncertainty relations)
        are enforced here. This constructor only enforces algebraic
        consistency of the moment tensors.
        """
        n = basis.n
        a = (
            np.zeros((n,), dtype=complex)
            if alpha is None
            else np.asarray(alpha, dtype=complex).reshape(n).copy()
        )
        NN = (
            np.zeros((n, n), dtype=complex)
            if N is None
            else np.asarray(N, dtype=complex).reshape(n, n).copy()
        )

        MM = (
            np.zeros((n, n), dtype=complex)
            if M is None
            else np.asarray(M, dtype=complex).reshape(n, n).copy()
        )
        return GaussianCore(basis=basis, alpha=a, N=NN, M=MM)

    @staticmethod
    def coherent(basis: ModeBasis, alpha: np.ndarray) -> "GaussianCore":
        r"""
        Constructs a coherent (displaced vacuum) Gaussian state.

        A coherent state is defined as displacement of vacuum:

        .. math::

            \lvert \alpha \rangle
            = D(\alpha)\lvert 0\rangle,

        where

        .. math::

            D(\alpha)
            = \exp\left(
            \sum_i \alpha_i a_i^\dagger
            -\alpha_i^* a_i
            \right).

        For a coherent states the first and second moments are:

        .. math::

            \alpha_i = \langle a_i \rangle,

        .. math::

            N_{ij}
            = \langle a_i^\dagger a_j\rangle
            = \alpha_i^*\alpha_j,

        .. math

            M_{ij}
            = \langle a_i a_j \rangle
            = \alpha_i \alpha_j

        so the quadrature covariance matrix equals that of vacuum.

        Examples
        --------

        Construct a coherent (displaced vacuum) state:

        .. jupyter-execute::

            import numpy as np

            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.labels.path_label import PathLabel
            from symop_proto.labels.polarization_label import PolarizationLabel
            from symop_proto.labels.mode_label import ModeLabel
            from symop_proto.core.operators import ModeOp
            from symop_proto.gaussian.basis import ModeBasis
            from symop_proto.gaussian.core import GaussianCore

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
            B2 = ModeBasis.build([m1, m2])

            alpha = np.array([1.0 + 0.3j, -0.4 + 0.2j], dtype=complex)

            core = GaussianCore.coherent(B2, alpha)

            print("alpha =", core.alpha)

            print("Centered moments =")
            N0, M0 = core.centered_moments()
            print("N0 =")
            print(N0)
            print("M0 =")
            print(M0)

            print("Quadrature covariance =")
            print(core.quadrature_covariance())


        Parameters
        ----------
        basis:
            Mode basis defining the operator set and Gram matrix.

        alpha:
            Complex displacement vector of shape ``(n,)``

        Returns
        -------
        GaussianCore
            Coherent Gaussian state in the specified basis.

        Notes
        -----
        This constructor guarantees internal consistency between first
        and second moments for a displaced vacuum.
        """
        a = np.asarray(alpha, dtype=complex).reshape(basis.n).copy()
        N = np.outer(np.conjugate(a), a)
        M = np.outer(a, a)
        return GaussianCore.from_moments(basis, alpha=a, N=N, M=M)

    def centered_moments(self) -> Tuple[np.ndarray, np.ndarray]:
        N0 = self.N - np.outer(np.conjugate(self.alpha), self.alpha)
        M0 = self.M - np.outer(self.alpha, self.alpha)
        return N0, M0

    def quadrature_mean(self) -> np.ndarray:
        r"""
        Return the mean vector in the quadrature representation

        Define canonical quadratures for each mode:

        .. math::
            x_i = \frac{a_i + a_i^\dagger}{\sqrt{2}}, \quad
            p_i = \frac{a_i - a_i^\dagger}{i\sqrt{2}}.

        The quadrature mean vector is defined as 

        .. math::

            d =
            \begin{pmatrix}
            \langle x_1 \rangle \\
            \vdots\\
            \langle x_n\rangle\\
            \langle p_1\rangle\\
            \vdots\\
            \langle p_n\rangle
            \end{pmatrix} \in \mathbb{R}^{2n}.

        Using the stored first moments

        .. math::

            \alpha_i = \langle a_i \rangle,

        this function computes:

        .. math::

            \langle x_i \rangle
            =\frac{\alpha_i + \alpha_i^*}{\sqrt{2}}
            =\sqrt{2}\mathrm{Re}(\alpha_i),

        .. math::

            \langle p_i \rangle
            = \frac{\alpha_i - \alpha_i^*}{i\sqrt{2}}
            = \sqrt{2}\mathrm{Im}(\alpha_i)


        Returns
        -------
        np.ndarray
            Real vector of shape ``(2n,)`` containing quadrature means
            ordered as ``(x_1, ...,x_n,p_1, ..., p_n )``

        Notes
        -----
        The quadratures are defined with respect to the current mode basis.
        If the basis is non-orthogonal, the canonical commutation structure
        is encoded in the Gram matrix :math:`G`, but the quadrature definitions
        above remain algebraically valid.
        """
        a = self.alpha
        x = (a + np.conjugate(a)) / np.sqrt(2.0)
        p = (a - np.conjugate(a)) / (1j * np.sqrt(2.0))

        return np.concatenate([x.real, p.real], axis=0)

    def quadrature_covariance(self) -> np.ndarray:
        r"""
        Return the symmetrized quadrature covariance matrix.

        Let

        .. math::

            R =
            (x_1, \dots, x_n, p_1, \dots, p_n)^T

        be the vector of quadrature operators defined by

        .. math::

            x_i = \frac{a_i + a_i^\dagger}{\sqrt{2}}, \qquad
            p_i = \frac{a_i - a_i^\dagger}{i\sqrt{2}}.

        The (symmetrized) covariance matrix is defined as

        .. math::

            V_{kl}
            = \frac{1}{2}
              \left\langle
                \{ \Delta R_k, \Delta R_l \}
              \right\rangle,

        where

        .. math::

            \Delta R_k = R_k - \langle R_k \rangle.

        In terms of centered ladder operators

        .. math::

            \delta a_i = a_i - \alpha_i,

        define centered two-point functions

        .. math::

            N_{0,ij}
            = \langle \delta a_i^\dagger \delta a_j \rangle,

        .. math::

            M_{0,ij}
            = \langle \delta a_i \delta a_j \rangle.

        The non-orthogonal commutation structure enters via

        .. math::

            \langle \delta a_i \delta a_j^\dagger \rangle
            = G_{ij} + N_{0,ji},

        where :math:`G` is the Gram matrix stored in
        :class:`ModeBasis`.

        Using these relations, all quadrature covariances are
        expressed in terms of

        .. math::

            A = M_0, \quad
            B = G + N_0^T, \quad
            C = N_0, \quad
            D = M_0^\dagger.

        The returned matrix has block structure

        .. math::

            V =
            \begin{pmatrix}
                V_{xx} & V_{xp} \\
                V_{px} & V_{pp}
            \end{pmatrix},

        where each block is an ``(n,n)`` matrix.

        Returns
        -------
        np.ndarray
            Real symmetric covariance matrix of shape ``(2n,2n)``.

        Notes
        -----
        * The covariance is symmetrized.
        * The matrix is real-valued for physical Gaussian states.
        * If the basis is canonical (Gram approximately identity),
          this reduces to the standard covariance matrix used in
          continuous-variable quantum optics.
        * Physicality (uncertainty relation) is not enforced here.
        """
        n = self.basis.n
        G = self.basis.gram
        N0, M0 = self.centered_moments()

        A = M0
        B = G + N0.T
        C = N0
        D = np.conjugate(M0.T)

        s = np.sqrt(2.0)
        inv2 = 0.5

        # Expectations <dx_i dx_j> (already centered)
        Vxx = inv2 * (A + B + C + D).real

        # <dx_i dp_j>
        # dp = (da - da^dag)/(i*sqrt(2))
        Vxp = inv2 * ((A - B + C - D) / (1j)).real

        # <dp_i dx_j> should be Vxp^T in symmetrized covariance, but compute explicitly
        Vpx = inv2 * ((A + B - C - D) / (1j)).real

        # <dp_i dp_j>
        Vpp = inv2 * (-(A - B - C + D)).real

        V = np.zeros((2 * n, 2 * n), dtype=float)
        V[0:n, 0:n] = Vxx
        V[0:n, n : 2 * n] = 0.5 * (Vxp + Vpx.T)
        V[n : 2 * n, 0:n] = 0.5 * (Vpx + Vxp.T)
        V[n : 2 * n, n : 2 * n] = Vpp
        return V

    def symplectic_form(self) -> np.ndarray:
        r"""
        Returns the commutation (symplectic) form for the quadratures

        With Gram matrix :math:`G` defined by

        .. math::

            [a_i, a_j^\dagger] = G_{ij}

        and quadratures

        .. math::

            x = \frac{a + a^\dagger}{\sqrt{2}},\quad
            p = \frac{a - a^\dagger}{i\sqrt{2}},

        the corresponding commutator matrix for 
        :math:`R=(x_1,\ldots, x_n, p_1, \ldots, p_n)^T` is

        .. math::

            [R_k, R_l] = i (\Omega_G)_{kl}

        where

        .. math::

            \Omega_G =
            \begin{pmatrix}
              0 & \mathrm{Re}(G)\\
              -\mathrm{Re}{G}^T & 0
            \end{pmatrix}

        in the most common orthogonal/canonical setting. In the general
        complex Gram case, test uses the full complex form

        .. math::

            \Omega_G =
            \begin{pmatrix}
            0&G\\
            -G^T&0
            \end{pmatrix}

        Returns
        -------
        np.ndarray
            Complex matrix of shape ``(2n, 2n)``
        """
        n = self.basis.n
        G = self.basis.gram

        Omega = np.zeros((2 * n, 2 * n), dtype=complex)
        Omega[0:n, n : 2 * n] = G
        Omega[n : 2 * n, 0:n] = -G.T
        return Omega

    def uncertainty_matrix(self) -> np.ndarray:
        r"""
        Return the Hermitian uncertainty matrix

        .. math::

            H = V + \frac{i}{2}\Omega_G,

        where :math:`V` is the symmetrized quadrature covariance and
        :math:`\Omega_G` encodes commutators.
        """
        V = np.asarray(self.quadrature_covariance(), dtype=float)
        Omega = self.symplectic_form()
        return V + 0.5j * Omega

    def is_physical(self, *, atol: float = 1e-12) -> bool:
        r"""
        Checks the Robertson-Schrödinger uncertainty relation.

        A gaussian state is physical iff

        .. math::

            V + \frac{i}{2} \Omega_G \succeq 0.

        Parameters
        ----------
        atol:
            Allowed numerical slack (eigenvalues may be slightly negative)

        Returns
        -------
        bool
            True if physical within tolerance
        """
        H = self.uncertainty_matrix()
        eigvals = np.linalg.eigvalsh(H)
        return bool(np.min(eigvals.real) >= -float(atol))

    def validate_physical(self, *, atol: float = 1e-12) -> None:
        if not self.is_physical(atol=atol):
            raise ValueError("Gaussian state violates uncertainty relation")

    def keep(self, modes: Tuple[ModeOpProto, ...]) -> GaussianCore:
        r"""
        Restricts this Gaussian core to a subset of modes (partial subsystem).

        Thi operation keeps only the selected modes and discards the rest,
        producing a reduced Gaussian core on the smaller basis.

        In terms of first and second moments, if ``k`` is the ordered index list
        corresponding to ``modes``, then the reduced moments are:

        .. math::

            \alpha' = \alpha_K, \quad
            N' = N_{K,K}, \quad
            M' = M_{K,K},

        and the reduced Gram matrix is

        .. math::

            G' = G_{K,K}.

        Parameters
        ----------
        modes:
            Ordered tuple of modes to keep. The order defines the order in the
            returned basis.

        Returns
        -------
        GaussianCore
            Reduced Gaussian core defined on the kept modes.

        Notes
        -----
        This it the Gaussian equivalent of tracing out the complement subsystem.
        It is exact for Gaussian states and preserves physicality if the original
        core is physical.

        Examples
        --------

        Keep only one mode of a two-mode coherent state:

        .. jupyter-execute::

            import numpy as np

            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.labels.path_label import PathLabel
            from symop_proto.labels.polarization_label import PolarizationLabel
            from symop_proto.labels.mode_label import ModeLabel
            from symop_proto.core.operators import ModeOp
            from symop_proto.gaussian.basis import ModeBasis
            from symop_proto.gaussian.core import GaussianCore

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))

            B = ModeBasis.build([m1, m2])
            core = GaussianCore.coherent(B, np.array([1.0 + 0.2j, -0.5 + 0.0j], dtype=complex))

            core1 = core.keep((m1,))
            print("kept alpha:", core1.alpha)
            print("kept gram:", core1.basis.gram)

        """

        idx = [self.basis.require_index_of(m) for m in modes]
        idx_arr = np.asarray(idx, dtype=int)

        alpha2 = self.alpha[idx_arr].copy()
        N2 = self.N[np.ix_(idx_arr, idx_arr)].copy()
        M2 = self.M[np.ix_(idx_arr, idx_arr)].copy()

        modes2 = tuple(modes)
        G2 = self.basis.gram[np.ix_(idx_arr, idx_arr)].copy()
        index_by_sig2 = {m.signature: i for i, m in enumerate(modes2)}
        basis2 = ModeBasis(modes=modes2, gram=G2, index_by_sig=index_by_sig2)

        return GaussianCore(basis=basis2, alpha=alpha2, N=N2, M=M2)

    def trace_out(self, modes: Tuple[ModeOpProto, ...]) -> GaussianCore:
        r"""
        Traces out (discard) a subset of modes.

        If ``D`` is the ordered index list corresponding to the discarded modes,
        and ``K`` is its complement, then this returns the reduced Gaussian core
        on ``K``:

        .. math::

            \alpha' = \alpha_K,\quad
            N' = N_{K,K},\quad
            M' = M_{K,K},\quad
            G' = G_{K,K}.

        Parameters
        ----------
        modes:
            Modes to discard (trace out). The order does not matter for the
            result, but must reference modes present in the basis.

        Returns
        -------
        GaussianCore
            Reduced Gaussian core on the remaining modes.

        Examples
        --------

        Discard one mode from a two-mode squeezed vacuum:

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
            core = TwoModeSqueezer(mode1=m1, mode2=m2, r=0.5, phi=0.0, check_ccr=True).apply(core)

            red = core.trace_out((m2,))
            print("reduced basis n:", red.basis.n)
            print("reduced N:", red.N)
        """
        drop = {self.basis.require_index_of(m) for m in modes}
        keep_idx: list[int] = [i for i in range(self.basis.n) if i not in drop]
        keep_modes: Tuple[ModeOpProto, ...] = tuple(
            self.basis.modes[i] for i in keep_idx
        )
        return self.keep(keep_modes)

    def extend_with_vacuum(
        self,
        modes: Tuple[ModeOpProto, ...],
        *,
        merge_approx: bool = False,
        env_kw: Optional[dict] = None,
        tol: float = 0.0,
    ) -> GaussianCore:
        r"""
        Extends this Gaussian core by appending new modes initialized in vacuum.

        This is a structural operation used to build Stinespring dilations for
        Gaussian channels (loss, thermal loss, amplification, etc.). It enlarges
        the basis by taking the union with ``modes`` and initializes all
        correlations involving *new* modes to vacuum:

        .. math::

            \alpha_{\text{new}} = 0,\quad
            N_{\text{new}} = 0,\quad
            M_{\text{new}} = 0,

        while leaving the existing block unchanged.

        If some of the provided modes are already present in the basis, they are
        ignored and the state is unchanged.

        Parameters
        ----------
        modes:
            Modes to add to the basis (vacuum-initialized if new).
        merge_approx, env_kw, tol:
            Passed through to :meth:`ModeBasis.union`. This allows approximate
            mode merging if desired.

        Returns
        -------
        GaussianCore
            A new GaussianCore defined on the union basis.

        Notes
        -----
        The union basis preserves the existing ordering and appends new unique
        modes at the end. Therefore the original moment tensors occupy the top-left
        blocks of the enlarged arrays.

        Examples
        --------

        Add an environment vacuum mode to a single-mode coherent state:

        .. jupyter-execute::

            import numpy as np

            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.labels.path_label import PathLabel
            from symop_proto.labels.polarization_label import PolarizationLabel
            from symop_proto.labels.mode_label import ModeLabel
            from symop_proto.core.operators import ModeOp
            from symop_proto.gaussian.basis import ModeBasis
            from symop_proto.gaussian.core import GaussianCore

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            s = ModeOp(env=env, label=ModeLabel(PathLabel("S"), PolarizationLabel.H()))
            e = ModeOp(env=env, label=ModeLabel(PathLabel("E"), PolarizationLabel.H()))

            B = ModeBasis.build([s])
            core = GaussianCore.coherent(B, np.array([1.0 + 0.0j], dtype=complex))

            core2 = core.extend_with_vacuum((e,))
            print("n old:", core.basis.n, "n new:", core2.basis.n)
            print("alpha:", core2.alpha)
            print("N block:", core2.N)
        """
        env_kw = env_kw or {}

        basis2 = self.basis.union(
            modes,
            merge_approx=merge_approx,
            env_kw=env_kw,
            tol=tol,
        )

        # If nothing changed, return self (immutability-friendly)
        if basis2.n == self.basis.n:
            return self

        n0 = self.basis.n
        n1 = basis2.n

        alpha2 = np.zeros((n1,), dtype=complex)
        N2 = np.zeros((n1, n1), dtype=complex)
        M2 = np.zeros((n1, n1), dtype=complex)

        # union preserves existing ordering and appends new unique modes
        alpha2[0:n0] = self.alpha
        N2[0:n0, 0:n0] = self.N
        M2[0:n0, 0:n0] = self.M

        return GaussianCore(basis=basis2, alpha=alpha2, N=N2, M=M2)
