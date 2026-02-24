from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import uuid

import numpy as np

from symop_proto.devices.base import DeviceApplyOptions
from symop_proto.devices.io import DeviceIO, DeviceResult
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.devices._utils import require_nonempty, select_modes
from symop_proto.gaussian.devices.base import GaussianDevice
from symop_proto.gaussian.ops.channel_between_bases import (
    apply_ladder_affine_between_bases,
)
from symop_proto.gaussian.ops.gram_blocks import gram_block
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel


def _default_out_path(in_path: object) -> PathLabel:
    return PathLabel(f"{in_path}_amp_{uuid.uuid4().hex[:8]}")


@dataclass(frozen=True)
class PathPhaseInsensitiveAmplifier(GaussianDevice):
    r"""
    Phase-insensitive (quantum-limited) amplifier acting on modes on a selected path.

    Physical model
    --------------
    The quantum-limited phase-insensitive amplifier is described in the Heisenberg
    picture by

    .. math::

        b = \sqrt{g}\,a + \sqrt{g-1}\,e^\dagger,

    where :math:`g \ge 1` is the power gain and :math:`e` is an environment mode
    prepared in vacuum. The creation operator :math:`e^\dagger` is required to
    preserve canonical commutation relations (CCR).

    In a possibly non-orthogonal mode basis, let :math:`G=[a,a^\dagger]` be the
    Gram block on the selected subspace. The amplifier induces the Gaussian moment
    map (normally ordered):

    .. math::

        \alpha' = \sqrt{g}\,\alpha,

    .. math::

        N' = g N + (g-1)\,G,

    .. math::

        M' = g M.

    This is the minimum added noise compatible with CCR (vacuum environment).

    Output relabeling
    -----------------
    For correct CCR bookkeeping in your label model, the Gaussian map must be applied
    on the same physical path labels as the input (otherwise path-orthogonality could
    incorrectly force overlaps to zero). After applying the map, the output modes may
    be *re-labeled* to a distinct path label for wiring/bookkeeping. This relabeling
    must not change the Gram matrix; it is performed by BaseDevice.apply via
    ``io.mode_map``.

    Parameters
    ----------
    path:
        Input path selector used by select_modes.
    gain:
        Power gain :math:`g \ge 1`.
    pol:
        Optional polarization selector.
    out_path:
        Optional bookkeeping output path label. If None, a unique path is generated.
    allow_empty:
        If False, raise when no modes are selected.
    tol:
        Numerical tolerance for Gram blocks.

    Examples
    --------

    Example 1: Coherent state amplification
    =======================================

    .. jupyter-execute::

        import numpy as np

        from symop_proto.core.operators import ModeOp
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.devices.amplifiers import PathPhaseInsensitiveAmplifier
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel

        env = GaussianEnvelope(omega0=20.0, sigma=0.8, tau=0.0, phi0=0.0)
        m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])

        core_in = GaussianCore.coherent(B, np.array([1.0 + 0.2j], dtype=complex))

        g = 4.0
        dev = PathPhaseInsensitiveAmplifier(path=PathLabel("A"), gain=g)
        res = dev.apply(core_in)
        core_out = res.state

        print("alpha_in:", np.asarray(core_in.alpha, dtype=complex))
        print("alpha_out:", np.asarray(core_out.alpha, dtype=complex))
        print("expected scale:", np.sqrt(g))

        print("N_out:\n", np.asarray(core_out.N, dtype=complex))
        print("M_out:\n", np.asarray(core_out.M, dtype=complex))


    Example 2: Noise contribution in N
    =================================

    For a coherent input (with negligible excess noise), the amplifier adds

    .. math::

        N_{\mathrm{excess}} = N - \alpha^* \alpha^{\mathsf T} \approx (g-1)\,G.

    .. jupyter-execute::

        import numpy as np

        from symop_proto.core.operators import ModeOp
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.devices.amplifiers import PathPhaseInsensitiveAmplifier
        from symop_proto.gaussian.ops.gram_blocks import gram_block
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel

        def as_c(x):
            return np.asarray(x, dtype=complex)

        def excess_N(core):
            alpha = as_c(core.alpha).reshape(-1, 1)
            N = as_c(core.N)
            N_coh = alpha.conj() @ alpha.T
            ex = N - N_coh
            ex = 0.5 * (ex + ex.conj().T)
            return ex

        env = GaussianEnvelope(omega0=20.0, sigma=0.8, tau=0.0, phi0=0.0)
        m = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])

        core_in = GaussianCore.coherent(B, np.array([1.0 + 0.2j], dtype=complex))

        g = 4.0
        dev = PathPhaseInsensitiveAmplifier(path=PathLabel("A"), gain=g)
        core_out = dev.apply(core_in).state

        G = gram_block((m,), (m,), tol=0.0)

        ex_out = excess_N(core_out)

        print("Excess N (output):\n", ex_out)
        print("Trace(excess):", float(np.real_if_close(np.trace(ex_out))))
        print("Expected (g-1)G:\n", (g - 1.0) * G)
        print("Difference:\n", ex_out - (g - 1.0) * G)

    """

    path: object
    gain: float
    pol: Optional[object] = None
    out_path: Optional[PathLabel] = None
    allow_empty: bool = False
    tol: float = 0.0

    def __post_init__(self) -> None:
        self._init_base()
        g = float(self.gain)
        if not np.isfinite(g) or g < 1.0:
            raise ValueError(
                f"gain must be finite and >= 1, got {self.gain!r}"
            )

    def resolve_io(self, state: GaussianCore) -> DeviceIO:
        modes_in = select_modes(state, path=self.path, pol=self.pol)
        if not self.allow_empty:
            modes_in = require_nonempty(
                modes_in, what="PathPhaseInsensitiveAmplifier"
            )

        final_out_path = (
            self.out_path
            if self.out_path is not None
            else _default_out_path(self.path)
        )

        # Physical output modes: same as input (same path) for the CCR map
        modes_out = list(modes_in)

        # Bookkeeping relabeling: same env, same pol, new path
        mode_map = []
        for m in modes_out:
            lbl = getattr(m, "label", None)
            if lbl is None:
                raise ValueError("Amplifier expects modes with a .label")
            in_pol = getattr(lbl, "pol", None)

            book_lbl = ModeLabel(final_out_path, in_pol)
            m_book = type(m)(env=m.env, label=book_lbl)
            mode_map.append((m, m_book))

        return DeviceIO(
            input_modes=tuple(modes_in),
            output_modes=tuple(modes_out),
            env_modes=(),
            mode_map=tuple(mode_map),
            meta={
                "gain": float(self.gain),
                "tol": float(self.tol),
                "out_path": str(final_out_path),
            },
        )

    def do_apply(self, state: GaussianCore, io: DeviceIO) -> GaussianCore:
        modes = tuple(io.input_modes)
        idx_in = [state.basis.require_index_of(m) for m in modes]

        # No basis change needed; basis_out can be rebuilt from the same physical modes.
        # This keeps the code aligned with apply_ladder_affine_between_bases expectations.
        basis_out = state.basis

        G = gram_block(modes, modes, tol=self.tol)
        G = 0.5 * (G + G.conj().T)

        if not np.isfinite(G).all():
            bad = np.argwhere(~np.isfinite(G))
            i, j = int(bad[0, 0]), int(bad[0, 1])
            raise ValueError(
                f"Gram block G has non-finite at {(i, j)}: {G[i, j]!r}"
            )

        g = float(self.gain)
        s = float(np.sqrt(g))
        m = len(modes)

        X = np.zeros((2 * m, 2 * m), dtype=complex)
        X[0:m, 0:m] = s * np.eye(m, dtype=complex)
        X[m: 2 * m, m: 2 * m] = s * np.eye(m, dtype=complex)

        Y = np.zeros((2 * m, 2 * m), dtype=complex)
        Y[m: 2 * m, 0:m] = (g - 1.0) * G

        return apply_ladder_affine_between_bases(
            state,
            idx_in=idx_in,
            basis_out=basis_out,
            X=X,
            Y=Y,
            d0=None,
            check_finite=True,
        )

    def _apply_gaussian(
        self,
        state: GaussianCore,
        *,
        options: Optional[DeviceApplyOptions] = None,
    ) -> DeviceResult[GaussianCore]:
        io = self.resolve_io(state)
        out = self.do_apply(state, io)
        return DeviceResult(state=out, io=io)
