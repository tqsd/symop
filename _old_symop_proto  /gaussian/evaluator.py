from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from symop_proto.core.protocols import (
    LadderOpProto,
    ModeOpProto,
    MonomialProto,
)
from symop_proto.gaussian.core import GaussianCore


@dataclass(frozen=True)
class GaussianEvaluator:
    r"""
    Evaluate Gaussian expectation values from a
    :class:`symop_proto.gaussian.core.GaussianCore`

    This evaluator computes expectation values of *normally ordered*
    monomials in ladder operators with respect to the Gaussian state
    described by the core.

    The core stores (possibly non-orthogonal) mode commutators via the Gram
    matrix:

    .. math::

        [a_i, a_j^\dagger] = G_{ij}.

    The stored moments are:

    .. math::

        \alpha_i = \langle a_i \rangle

    .. math::

        N_{ij}=\langle a_i^\dagger a_j \rangle,

    .. math::

        M_{ij}=\langle a_i a_j \rangle.

    The centered operators are defined as:

    .. math::

        \delta a_i = a_i - \alpha_i

    and centered second moments are:

    .. math::

        N_{0,ij} = \langle \delta a_i^\dagger \delta a_j \rangle,

    .. math::

        M_{0,ij} = \langle \delta a_i \delta a_j \rangle.

    The non-orthogonal commutation structure enters via:

    .. math::

        \langle \delta a_i \delta a_j^\dagger \rangle = G_{ij} + N_{0,ji}.


    Algorithm
    ---------
    For a product of ladder operators :math:`O_1 O_2 \cdots O_k`, we use
    Gaussian recursion (mean+pairings):

    .. math::

        \langle O_1 O_2 \cdots O_k \rangle
        =
        \mu_1 \langle O_2 \cdots O_k \rangle
        +
        \sum_{t=2}^k C_{1t} \langle O_2 \cdots \widehat{O_t} \cdots O_k
        \rangle,

    where :math:`\mu_1=\langle O_1 \rangle` is the mean contribution and
    :math:`C_{1t}=\langle \delta O_1 \delta O_t\rangle` is the centered
    contraction. The hat indicates omission of that factor.

    This avoids explicit subset enumeration and is effiscient for the small
    degrees of typical in regaussianification and observable extraction.

    Contract
    --------
    - The monomial is assumed to be in *normal order*
    - All modes appearing must be present in the evaluator's basis
    - This evaluator does not reorder operators

    Examples
    --------

    Example 1: Vacuum state sanity checks.
    ======================================
    Demonstrates that all first and normal-ordered second moments vanish,
    and that non-normal-ordered moments correctly pick up the Gram matrix
    contribution via the CCR:

    .. math::

        \langle a_i a_j^\dagger \rangle = G_{ij}

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.core.monomial import Monomial
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.evaluator import GaussianEvaluator

        # Build orthogonal modes (different spatial paths)
        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)

        m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))

        B = ModeBasis.build([m1, m2])

        core = GaussianCore.vacuum(B)
        ev = GaussianEvaluator(core)

        # Identity monomial
        mon_id = Monomial()
        print("<I> =", ev.expect_monomial(mon_id))

        # <a1> = 0
        mon_a1 = Monomial(creators=(), annihilators=(m1.ann,))
        print("<a1> =", ev.expect_monomial(mon_a1))

        # <a1^dag a1> = 0
        mon_n1 = Monomial(creators=(m1.create,), annihilators=(m1.ann,))
        print("<n1> =", ev.expect_monomial(mon_n1))

        # Not normal ordered: <a1 a1^dag> = G_11 + <a1^dag a1>
        print("G_11 =", B.gram[0, 0])
        print("<a1 a1^dag> =", B.gram[0, 0] + ev.expect_monomial(mon_n1))


    Example 2: Coherent (displaced vacuum) state.
    =============================================
    Demonstrates that the evaluator reproduces:

    .. math::
        \langle a_i \rangle = \alpha_i, \quad
        \langle a_i^\dagger a_j \rangle = \alpha_i^* \alpha_j, \quad
        \langle a_i a_j \rangle = \alpha_i \alpha_j

    i.e. Gaussian factorization with nonzero first moments.

    .. jupyter-execute::

        import numpy as np

        from symop_proto.core.monomial import Monomial
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.evaluator import GaussianEvaluator

        # Reuse B, m1, m2 from the previous example cell

        alpha = np.array([0.3 + 0.1j, -0.2 + 0.5j], dtype=complex)
        core_c = GaussianCore.coherent(B, alpha)
        ev_c = GaussianEvaluator(core_c)

        # <a_i> = alpha_i
        mon_a1 = Monomial(creators=(), annihilators=(m1.ann,))
        mon_a2 = Monomial(creators=(), annihilators=(m2.ann,))
        print("<a1> =", ev_c.expect_monomial(mon_a1), "expected", alpha[0])
        print("<a2> =", ev_c.expect_monomial(mon_a2), "expected", alpha[1])

        # <a_i^dag a_i> = |alpha_i|^2
        mon_n1 = Monomial(creators=(m1.create,), annihilators=(m1.ann,))
        mon_n2 = Monomial(creators=(m2.create,), annihilators=(m2.ann,))
        print("<n1> =", ev_c.expect_monomial(mon_n1), "expected", abs(alpha[0]) ** 2)
        print("<n2> =", ev_c.expect_monomial(mon_n2), "expected", abs(alpha[1]) ** 2)

        # Cross moment <a1^dag a2> = alpha1^* alpha2
        mon_adag1_a2 = Monomial(creators=(m1.create,), annihilators=(m2.ann,))
        print(
            "<a1^dag a2> =",
            ev_c.expect_monomial(mon_adag1_a2),
            "expected",
            np.conjugate(alpha[0]) * alpha[1],
        )

        # Pairing moment <a1 a2> = alpha1 alpha2 for a coherent state
        mon_a1a2 = Monomial(creators=(), annihilators=(m1.ann, m2.ann))
        print("<a1 a2> =", ev_c.expect_monomial(mon_a1a2), "expected", alpha[0] * alpha[1])

    Example 3: Two-mode squeezed Gaussian state.
    Demonstrates nonzero pairing correlations M_ij and verifies that
    the evaluator reproduces:

    .. math::

        \langle a_i a_j \rangle = M_{ij}, \quad
        \langle a_i^\dagger a_i \rangle = N_{ii}

    confirming correct handling of anomalous correlations.

    .. jupyter-execute::

        import numpy as np

        from symop_proto.core.monomial import Monomial
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.evaluator import GaussianEvaluator

        # Reuse B, m1, m2 from the first example cell

        # Two-mode squeezed vacuum ideal moments (orthogonal case)
        r = 0.7
        s2 = np.sinh(r) ** 2
        sc = np.sinh(r) * np.cosh(r)

        alpha0 = np.zeros((2,), dtype=complex)
        N = np.array([[s2, 0.0], [0.0, s2]], dtype=complex)
        M = np.array([[0.0, sc], [sc, 0.0]], dtype=complex)

        core_sq = GaussianCore.from_moments(B, alpha=alpha0, N=N, M=M)
        ev_sq = GaussianEvaluator(core_sq)

        # <a1 a2> = M_12
        mon_a1a2 = Monomial(creators=(), annihilators=(m1.ann, m2.ann))
        print("<a1 a2> =", ev_sq.expect_monomial(mon_a1a2), "expected", M[0, 1])

        # <a1^dag a1> = N_11
        mon_n1 = Monomial(creators=(m1.create,), annihilators=(m1.ann,))
        print("<n1> =", ev_sq.expect_monomial(mon_n1), "expected", N[0, 0])

        # <a2^dag a2> = N_22
        mon_n2 = Monomial(creators=(m2.create,), annihilators=(m2.ann,))
        print("<n2> =", ev_sq.expect_monomial(mon_n2), "expected", N[1, 1])

        # <a1 a1> = M_11 = 0 in this idealized model
        mon_a1a1 = Monomial(creators=(), annihilators=(m1.ann, m1.ann))
        print("<a1 a1> =", ev_sq.expect_monomial(mon_a1a1), "expected", M[0, 0])


    Example 4: Non-orthogonal mode basis.
    Demonstrates that the evaluator correctly incorporates the Gram matrix
    when modes overlap (non-canonical commutation relations),
    verifying identities such as:

    .. math::

        \langle a_i a_j^\dagger \rangle = G_{ij} + \langle a_j^\dagger a_i\rangle

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.core.monomial import Monomial
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.evaluator import GaussianEvaluator

        # Non-orthogonal basis: same path and polarization, different delays -> overlap != 0
        env1 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        env2 = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.2, phi0=0.0)

        m1 = ModeOp(env=env1, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        m2 = ModeOp(env=env2, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))

        B = ModeBasis.build([m1, m2])
        print("Gram matrix G =")
        print(B.gram)

        core = GaussianCore.vacuum(B)
        ev = GaussianEvaluator(core)

        # Use identity: <a1 a2^dag> = G_12 + <a2^dag a1>
        mon_adag2_a1 = Monomial(creators=(m2.create,), annihilators=(m1.ann,))
        rhs = B.gram[0, 1] + ev.expect_monomial(mon_adag2_a1)

        print("<a2^dag a1> (normal ordered) =", ev.expect_monomial(mon_adag2_a1))
        print("<a1 a2^dag> via identity =", rhs)
        print("Expected (vacuum) <a1 a2^dag> = G_12 =", B.gram[0, 1])

    Example 5: Fourth-order Gaussian moment (Wick theorem).
    Demonstrates that higher-order expectations factorize correctly,
    e.g. for a single-mode thermal-like Gaussian:

    .. math::

        \langle a^\dagger a^\dagger a a \rangle = 2  \bar{n}^2

    confirming proper recursive contraction logic.

    .. jupyter-execute::

        import numpy as np

        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.core.operators import ModeOp
        from symop_proto.core.monomial import Monomial
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.evaluator import GaussianEvaluator

        # A "thermal-like" single-mode Gaussian: alpha=0, M=0, N=nbar.
        # Wick implies: <a^dag a^dag a a> = 2 * nbar^2
        env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
        m = ModeOp(env=env, label=ModeLabel(PathLabel("T"), PolarizationLabel.H()))

        B1 = ModeBasis.build([m])

        nbar = 0.37
        alpha0 = np.zeros((1,), dtype=complex)
        N = np.array([[nbar]], dtype=complex)
        M = np.array([[0.0]], dtype=complex)

        core = GaussianCore.from_moments(B1, alpha=alpha0, N=N, M=M)
        ev = GaussianEvaluator(core)

        mon_n = Monomial(creators=(m.create,), annihilators=(m.ann,))
        mon_adag2_a2 = Monomial(creators=(m.create, m.create), annihilators=(m.ann, m.ann))

        print("<a^dag a> =", ev.expect_monomial(mon_n), "expected", nbar)
        print("<a^dag a^dag a a> =", ev.expect_monomial(mon_adag2_a2), "expected", 2.0 * nbar * nbar)
    """

    core: GaussianCore

    def _require_index_of_mode(self, mode: ModeOpProto) -> int:
        return self.core.basis.require_index_of(mode)

    def _mean_of_op(self, op: LadderOpProto) -> complex:
        r"""
        Mean value of a ladder operator

        .. math::

            \langle a_i \rangle = \alpha_i \quad
            \langle a_i^\dagger \rangle = \alpha_i^*
        """
        i = self._require_index_of_mode(op.mode)
        a = self.core.alpha[i]
        if op.is_annihilation:
            return a
        if op.is_creation:
            return np.conjugate(a)
        return 0.0 + 0.0j

    def _centered_contraction(self, op1: LadderOpProto, op2: LadderOpProto) -> complex:
        r"""
        Returns <delta(op1) delta(op2)> expressed in terms of centered moments.

        Mathematics
        -----------

        .. math::

            \langle a_i \delta a_j\rangle = M_{0,ij}

        .. math::

            \langle \delta a_i^\dagger \delta a_j \rangle = N_{0,ij}

        .. math::

            \langle \delta a_i^\dagger \delta a_j^\dagger \rangle = M_{0,ij}^*

        """

        i = self._require_index_of_mode(op1.mode)
        j = self._require_index_of_mode(op2.mode)

        N0, M0 = self.core.centered_moments()
        G = self.core.basis.gram

        if op1.is_annihilation and op2.is_annihilation:
            return M0[i, j]
        if op1.is_creation and op2.is_creation:
            return np.conjugate(M0[i, j])
        if op1.is_creation and op2.is_annihilation:
            return N0[i, j]
        if op1.is_annihilation and op2.is_creation:
            return G[i, j] + N0[j, i]
        return 0.0 + 0.0j

    def _centered_contraction_fast(
        self,
        op1: LadderOpProto,
        op2: LadderOpProto,
        *,
        N0: np.ndarray,
        M0: np.ndarray,
        G: np.ndarray,
    ) -> complex:
        i = self._require_index_of_mode(op1.mode)
        j = self._require_index_of_mode(op2.mode)

        if op1.is_annihilation and op2.is_annihilation:
            return M0[i, j]
        if op1.is_creation and op2.is_creation:
            return np.conjugate(M0[i, j])
        if op1.is_creation and op2.is_annihilation:
            return N0[i, j]
        if op1.is_annihilation and op2.is_creation:
            return G[i, j] + N0[j, i]
        return 0.0 + 0.0j

    def _ops_key(self, ops: Sequence[LadderOpProto]) -> Tuple[Tuple[Tuple, int], ...]:
        out: List[Tuple[Tuple, int]] = []
        for op in ops:
            typ = 1 if op.is_creation else 0
            out.append((op.mode.signature, typ))
        return tuple(out)

    def _expect_ops(self, ops: Sequence[LadderOpProto]) -> complex:
        r"""
        Evaluates a Gaussian Expectation value fo an explicid operator sequence

        This is the core engine behind :meth:`expect_monomial`. It takes an
        explicit sequence of ladder operators (annihilation/creation) and
        evaluates

        .. math::

            \langle O_1 O_2 \ldots O_k \rangle

        with respect to the Gaussian state described by ``self.core``.

        Gaussian state defining data
        ----------------------------
        The Gaussian core stores:
        - First moments: :math:`\alpha_i=\langle a_i\rangle`
        - Second moments: :math:`N_{ij}=\langle a_i^\dagger a_j \rangle`,
          :math:`\quad M_{ij}=\langle a_i a_j \rangle`
        - A (possibly non-orthogonal) mode basis with Gram matrix :math:`G`
          where :math:`[a_i, a_j^\dagger]=G_{ij}`

        For the recursion this method uses centered operators:

        .. math::

            \delta a_i = a_i - \alpha_i

        and centered second moments:

        .. math::

            N_{0,ij} = \langle \delta a_i^\dagger \delta a_j \rangle
            \quad
            M_{0,ij} = \langle \delta a_i \delta a_j \rangle

        The remaining centered contraction is determined by the CCR:

        .. math::

            \langle \delta a_i \delta a_j^\dagger \rangle = G_{ij} + N_{0,ji}

        Contraction table used
        ----------------------
        For two ladder operators `op1`, `op2`, we define the centered
        contraction:

        .. math::

            C( \mathrm{op1}, \mathrm{op2}) = \langle \delta(\mathrm{op1})
            \delta(\mathrm{op2})\rangle

        which is computed from `(N0, M0, G)` as follows (i,j are mode indices):

        - :math:`a_i` with :math:`a_j`:                  `M0[i, j]`
        - :math:`a_i^\dagger` with :math:`a_j^\dagger`:  `conj(M0[i, j])`
        - :math:`a_i^\dagger` with :math:`a_j`:          `N0[i, j]`
        - :math:`a_i` with :math:`a_j^\dagger`:          `G[i, j] + N0[j, i]`

        Recursion Formula (Gaussian Wick recursion)
        -------------------------------------------
        For any operator sequence `(0_1, ..., O_k)`, this method uses the
        identity:

        .. math::

            \langle O_1 O_2 \ldots O_k\rangle = \langle O_1 \rangle
            \langle O_2 \ldots O_k \rangle + \sum_{t=2\ldots k}
            \langle\delta O_1 \delta O_t\rangle
            \langle O_2 \ldots O_{t-1} O_{t+1} \ldots O_k\rangle

        where:
        - :math:`\langle O_1 \rangle` is the mean contribution (from alpha)
        - :math:`\langle \delta O_1 \delta O_t \rangle` is the centered
          contraction C(0_1, O_t)
        - The second expectration removes :math:`O_t` from the remaining
          product.

        This recursion is equivalent to Wick's theorem, but avoids explicit
        enumeration of all pair partitions. It is usually efficient for the
        moderate degrees encountered in regaussianification and observable
        extraction.

        Memoization (cache)
        -------------------
        The recursion revisits the same subsequences many times. To avoid an
        exponentaill blow-up, this method memoizes intermediate results in a
        dict.

        Notes
        -----
        - Phisicality of moments (uncertainty constraints) is not checked here.

        """
        N0, M0 = self.core.centered_moments()
        G = self.core.basis.gram

        cache: Dict[Tuple[Tuple[Tuple, int], ...], complex] = {}

        def rec(seq: Sequence[LadderOpProto]) -> complex:
            if len(seq) == 0:
                return 1.0 + 0.0j

            key = self._ops_key(seq)
            if key in cache:
                return cache[key]

            first = seq[0]
            rest = seq[1:]

            out = self._mean_of_op(first) * rec(rest)

            for t, op_t in enumerate(rest):
                c = self._centered_contraction_fast(first, op_t, N0=N0, M0=M0, G=G)

                rest2 = list(rest[:t]) + list(rest[t + 1 :])
                out += c * rec(rest2)
            cache[key] = out
            return out

        return rec(ops)

    def expect_monomial(self, monomial: MonomialProto) -> complex:
        r"""
        Evaluates the expectation value of a normally ordered monomial
        :class:`symop_proto.core.protocols.MonomialProto` is normally
        ordered by construction.

        Parameters
        ----------
        monomial:
            A normally ordered monomial

        Returns
        -------
        complex
            The expectation values :math:`\langle \mathrm{monomial}\rangle`

        Raises
        ------
        KeyError
            If the monomial references a mode not present in the basis.

        Examples
        --------

        Vacuum expectations in a two-mode orthogonal basis:

        .. jupyter-execute::

            import numpy as np

            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.labels.path_label import PathLabel
            from symop_proto.labels.polarization_label import PolarizationLabel
            from symop_proto.labels.mode_label import ModeLabel
            from symop_proto.core.operators import ModeOp
            from symop_proto.core.monomial import Monomial
            from symop_proto.gaussian.basis import ModeBasis
            from symop_proto.gaussian.core import GaussianCore
            from symop_proto.gaussian.evaluator import GaussianEvaluator

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)

            m1 = ModeOp(env=env,
                        label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            m2 = ModeOp(env=env,
                        label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))

            B = ModeBasis.build([m1, m2])
            core = GaussianCore.vacuum(B)
            ev = GaussianEvaluator(core)

            # <a_1> = 0
            mon_a1 = Monomial(creators=(), annihilators=(m1.ann,))
            print("<a1> =", ev.expect_monomial(mon_a1))

            # <a_1^dag a_1> = 0 in vacuum
            mon_n1 = Monomial(creators=(m1.create,), annihilators=(m1.ann,))
            print("<n1> =", ev.expect_monomial(mon_n1))

            # <a_1 a_1^dag> is not normal ordered; use identity:
            # <a a^dag> = G_11 + <a^dag a> = G_11 in vacuum.
            print("G_11 =", B.gram[0, 0])
            print("<a1 a1^dag> =", B.gram[0, 0] + ev.expect_monomial(mon_n1))

        Coherent state expectations:

        .. jupyter-execute::

            alpha = np.array([0.3 + 0.1j, -0.2 + 0.5j], dtype=complex)
            core_c = GaussianCore.coherent(B, alpha)
            ev_c = GaussianEvaluator(core_c)

            mon_a1 = Monomial(creators=(), annihilators=(m1.ann,))
            mon_n1 = Monomial(creators=(m1.create,), annihilators=(m1.ann,))

            print("<a1> =", ev_c.expect_monomial(mon_a1))
            print("<n1> =", ev_c.expect_monomial(mon_n1))
            print("Expected <a1> =", alpha[0])
            print("Expected <n1> =", abs(alpha[0])**2)

        Two-mode squeezed vacuum correlations:

        .. jupyter-execute::

            r = 0.7
            s2 = np.sinh(r)**2
            sc = np.sinh(r) * np.cosh(r)

            alpha0 = np.zeros((2,), dtype=complex)
            N = np.array([[s2, 0.0],
                        [0.0, s2]], dtype=complex)
            M = np.array([[0.0, sc],
                        [sc, 0.0]], dtype=complex)

            core_sq = GaussianCore.from_moments(B, alpha=alpha0, N=N, M=M)
            ev_sq = GaussianEvaluator(core_sq)

            # <a1 a2> = M_12 for alpha=0
            mon_a1a2 = Monomial(creators=(), annihilators=(m1.ann, m2.ann))
            print("<a1 a2> =", ev_sq.expect_monomial(mon_a1a2))
            print("Expected =", M[0, 1])

        """

        ops: List[LadderOpProto] = []
        ops.extend(list(monomial.creators))
        ops.extend(list(monomial.annihilators))
        return self._expect_ops(ops)

    def expect_ops(self, ops: Sequence[LadderOpProto]) -> complex:
        r"""
        Evaluates a Gaussian expectation value for an explicit ladder-operator word.

        This is a thin public wrapper around the internal recursion engine used by
        :meth:`expect_monomial`.

        Parameters
        ----------
        ops:
            Sequence of ladder operators (annihilation and/or creation) in the
            indended order.

        Returns
        -------
        complex
            The expectation value :math:`\langle O_1 O_2 \cdots O_3 \rangle`.

        Notes
        -----
        If the operator word is *normally ordered*, this equals the corresponding
        monomial expectation. If it is not normally ordered, the result includes
        Gram-matrix contributions through contractions of the form

        .. math::

            \langle \delta a_i \delta a_j^\dagger \rangle
            = G_{ij} + N_{0,ji}

        Examples
        --------
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

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            B = ModeBasis.build([m])

            core = GaussianCore.vacuum(B)
            ev = GaussianEvaluator(core)

            # Non-normal-ordered: <a a^dag> = G + <a^dag a> = G in vacuum.
            val = ev.expect_ops([m.ann, m.create])
            print("value:", val)
            print("expected:", B.gram[0, 0])
        """
        return self._expect_ops(list(ops))

    def expect_word(
        self,
        *,
        creators: Sequence[LadderOpProto] = (),
        annihilators: Sequence[LadderOpProto] = (),
    ) -> complex:
        r"""
        Evaluate the expectation value of a normally ordered word.

        This helper builds the operator list as

        .. math::

            (a^\dagger \cdots a^\dagger)(a \cdots a)

        and evaluates it as a Gaussian expectation. It matches the semantics of
        a :class:`symop_proto.core.monomial.Monomial` but aboids explicitly
        constructing the monomial when doing quick queries.

        Parameters
        ----------
        creators:
            Creation operators
        annihilators:
            Annihilation operators

        Returns
        -------
        complex
            :math:`\langle \prod_i a_{c_i}^\dagger \prod_j a_{a_j}\rangle`


        Examples
        --------
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
            from symop_proto.gaussian.maps.channel import Displacement

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            B = ModeBasis.build([m])

            core = GaussianCore.vacuum(B)
            core = Displacement(modes=(m,), beta=np.array([0.7 + 0.2j])).apply(core)

            ev = GaussianEvaluator(core)

            n = ev.expect_word(creators=(m.create,), annihilators=(m.ann,))
            print("<n> =", n, "expected:", abs(core.alpha[0])**2)
        """
        ops: List[LadderOpProto] = []
        ops.extend(list(creators))
        ops.extend(list(annihilators))
        return self._expect_ops(ops)

    def expect_monomials(self, monomials: Sequence[MonomialProto]) -> np.ndarray:
        r"""
        Evaluate multiple normally ordered monomials.

        Parameters
        ----------
        monomials:
            Sequence of normally ordered monomials.

        Returns
        -------
        np.ndarray
            Complex array of shape ``(len(monomials),)`` with
            :math:`\langle m_k \rangle` for each monomial.

        Examples
        --------
        .. jupyter-execute::

            import numpy as np

            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.labels.path_label import PathLabel
            from symop_proto.labels.polarization_label import PolarizationLabel
            from symop_proto.labels.mode_label import ModeLabel
            from symop_proto.core.operators import ModeOp
            from symop_proto.core.monomial import Monomial
            from symop_proto.gaussian.basis import ModeBasis
            from symop_proto.gaussian.core import GaussianCore
            from symop_proto.gaussian.evaluator import GaussianEvaluator
            from symop_proto.gaussian.maps.passive import BeamSplitter

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
            B = ModeBasis.build([m1, m2])

            core = GaussianCore.coherent(B, np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex))
            core = BeamSplitter(mode1=m1, mode2=m2, theta=np.pi/4.0, phi=0.0, check_unitary=True).apply(core)

            ev = GaussianEvaluator(core)

            mons = [
                Monomial(creators=(), annihilators=(m1.ann,)),
                Monomial(creators=(), annihilators=(m2.ann,)),
                Monomial(creators=(m1.create,), annihilators=(m1.ann,)),
                Monomial(creators=(m2.create,), annihilators=(m2.ann,)),
            ]
            vals = ev.expect_monomials(mons)
            print(vals)
        """
        out = np.empty((len(monomials),), dtype=complex)
        for i, m in enumerate(monomials):
            out[i] = self.expect_monomial(m)
        return out

    def mean(self, mode: ModeOpProto) -> complex:
        r"""
        Mean field amplitude for a mode.

        .. math::

            \langle a \rangle = \alpha.

        Parameters
        ----------
        mode:
            Mode whose mean is returned.

        Returns
        -------
        complex
            The first moment :math:`\alpha_i` for the requested mode.

        Examples
        --------
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
            from symop_proto.gaussian.maps.channel import Displacement

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            B = ModeBasis.build([m])

            core = GaussianCore.vacuum(B)
            beta = np.array([0.3 - 0.4j], dtype=complex)
            core = Displacement(modes=(m,), beta=beta).apply(core)

            ev = GaussianEvaluator(core)
            print(ev.mean(m), "expected:", beta[0])
        """
        i = self._require_index_of_mode(mode)
        return complex(self.core.alpha[i])

    def number(self, mode: ModeOpProto) -> float:
        r"""
        Mean photon number of a mode.

        .. math::

            \langle a^\dagger a \rangle = N_{ii}.

        Parameters
        ----------
        mode:
            Mode whose mean photon number is returned.

        Returns
        -------
        float
            The real part of :math:`N_{ii}`.

        Notes
        -----
        For physical states, :math:`N_{ii}` is real and non-negative.
        This helper returns ``float(real(N_ii))``.

        Examples
        --------
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
            from symop_proto.gaussian.maps.bogoliubov import SingleModeSqueezer

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            B = ModeBasis.build([m])

            core = GaussianCore.vacuum(B)
            r = 0.5
            core = SingleModeSqueezer(mode=m, r=r, phi=0.0, check_ccr=True).apply(core)

            ev = GaussianEvaluator(core)
            print("n:", ev.number(m), "expected:", float(np.sinh(r)**2))
        """
        i = self._require_index_of_mode(mode)
        return float(np.real(self.core.N[i, i]))

    def pairing(self, mode1: ModeOpProto, mode2: ModeOpProto) -> complex:
        r"""
        Pairing (anomalous) correlator between two modes.

        .. math::

            \langle a_i a_j \rangle = M_{ij}.

        Parameters
        ----------
        mode1, mode2:
            The two modes.

        Returns
        -------
        complex
            The pairing correlator :math:`M_{ij}`.

        Examples
        --------
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
            from symop_proto.gaussian.maps.bogoliubov import TwoModeSqueezer

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m1 = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            m2 = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
            B = ModeBasis.build([m1, m2])

            core = GaussianCore.vacuum(B)
            r = 0.4
            phi = 0.0
            core = TwoModeSqueezer(mode1=m1, mode2=m2, r=r, phi=phi, check_ccr=True).apply(core)

            ev = GaussianEvaluator(core)
            sh = np.sinh(r)
            ch = np.cosh(r)
            print("M12:", ev.pairing(m1, m2), "expected:", np.exp(1j * phi) * sh * ch)
        """
        i = self._require_index_of_mode(mode1)
        j = self._require_index_of_mode(mode2)
        return complex(self.core.M[i, j])

    def g2(self, mode: ModeOpProto, *, eps: float = 1e-14) -> float:
        r"""
        Second-order intensity correlation :math:`g^{(2)}(0)` for a single mode.

        Defined as

        .. math::

            g^{(2)}(0)
            =
            \frac{\langle a^\dagger a^\dagger a a \rangle}{\langle a^\dagger a \rangle^2}.

        Parameters
        ----------
        mode:
            Mode to evaluate.
        eps:
            Numerical cutoff for the denominator.

        Returns
        -------
        float
            The real value of :math:`g^{(2)}(0)`.

        Raises
        ------
        ValueError
            If :math:`\langle a^\dagger a \rangle` is too small.

        Examples
        --------
        A thermal-like Gaussian has :math:`g^{(2)}(0)=2`.

        .. jupyter-execute::

            import numpy as np

            from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
            from symop_proto.labels.path_label import PathLabel
            from symop_proto.labels.polarization_label import PolarizationLabel
            from symop_proto.labels.mode_label import ModeLabel
            from symop_proto.core.operators import ModeOp
            from symop_proto.core.monomial import Monomial
            from symop_proto.gaussian.basis import ModeBasis
            from symop_proto.gaussian.core import GaussianCore
            from symop_proto.gaussian.evaluator import GaussianEvaluator

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m = ModeOp(env=env, label=ModeLabel(PathLabel("T"), PolarizationLabel.H()))
            B = ModeBasis.build([m])

            nbar = 0.37
            core = GaussianCore.from_moments(
                B,
                alpha=np.zeros((1,), dtype=complex),
                N=np.array([[nbar]], dtype=complex),
                M=np.array([[0.0]], dtype=complex),
            )

            ev = GaussianEvaluator(core)
            print("g2:", ev.g2(m), "expected ~ 2.0")
        """
        n = self.expect_word(creators=(mode.create,), annihilators=(mode.ann,))
        den = float(np.real(n))
        if abs(den) <= float(eps):
            raise ValueError("g2 undefined for near-zero mean photon number")

        num = self.expect_word(
            creators=(mode.create, mode.create),
            annihilators=(mode.ann, mode.ann),
        )
        return float(np.real(num) / (den * den))

    def expect_quadrature_mean(self) -> np.ndarray:
        r"""
        Convenience wrapper returning the quadrature mean vector.

        See :meth:`symop_proto.gaussian.core.GaussianCore.quadrature_mean`.

        Returns
        -------
        np.ndarray
            Real vector :math:`d \in \mathbb{R}^{2n}`.

        Examples
        --------
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
            from symop_proto.gaussian.maps.channel import Displacement

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            B = ModeBasis.build([m])

            core = GaussianCore.vacuum(B)
            core = Displacement(modes=(m,), beta=np.array([0.2 + 0.3j], dtype=complex)).apply(core)

            ev = GaussianEvaluator(core)
            print(ev.expect_quadrature_mean())
        """
        return self.core.quadrature_mean()

    def expect_quadrature_covariance(self) -> np.ndarray:
        r"""
        Convenience wrapper returning the symmetrized quadrature covariance.

        See :meth:`symop_proto.gaussian.core.GaussianCore.quadrature_covariance`.

        Returns
        -------
        np.ndarray
            Real symmetric matrix :math:`V \in \mathbb{R}^{2n \times 2n}`.

        Examples
        --------
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
            from symop_proto.gaussian.maps.bogoliubov import SingleModeSqueezer

            env = GaussianEnvelope(omega0=0.0, sigma=0.5, tau=0.0, phi0=0.0)
            m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
            B = ModeBasis.build([m])

            core = GaussianCore.vacuum(B)
            core = SingleModeSqueezer(mode=m, r=0.6, phi=0.0, check_ccr=True).apply(core)

            ev = GaussianEvaluator(core)
            V = ev.expect_quadrature_covariance()
            print("V shape:", V.shape)
            print(V)
        """
        return self.core.quadrature_covariance()
