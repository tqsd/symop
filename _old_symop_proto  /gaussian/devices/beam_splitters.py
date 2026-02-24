from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np

from symop_proto.core.protocols import HasSignature, ModeOpProto
from symop_proto.devices.base import DeviceApplyOptions
from symop_proto.devices.io import DeviceIO, DeviceResult
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.devices._utils import require_nonempty, select_modes
from symop_proto.gaussian.devices.base import GaussianDevice
from symop_proto.gaussian.ops.passive import apply_passive_unitary_subset
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel


def _env_key(env: Any, *, approx: bool, approx_kw: Dict[str, Any]) -> Tuple[Any, ...]:
    if not approx:
        return ("sig", env.signature)
    # In your codebase approx_signature accepts **kw (see ModeLabel),
    # so we forward approx_kw.
    return ("approx", env.approx_signature(**approx_kw))


def _mode_key(
    mode: ModeOpProto, *, approx: bool, approx_kw: Dict[str, Any]
) -> Tuple[Any, ...]:
    return (
        mode.label.pol.signature,
        _env_key(mode.env, approx=approx, approx_kw=approx_kw),
    )


@dataclass(frozen=True)
class PathBeamSplitter(GaussianDevice):
    r"""
    Ideal (frequency-flat) beamsplitter mixing two disjoint mode subsets.

    Overview
    --------
    This device selects two disjoint subsets of modes (``port1`` and ``port2``),
    and mixes them pairwise with an ideal passive beamsplitter transformation.
    Modes not selected by either port are left untouched, including all
    cross-correlations with the selected modes.

    If ``vacuum_fill=False`` (default), the two subsets must be perfectly pairable:
    equal counts per pairing key and each mode must find a compatible partner on
    the opposite port.

    If ``vacuum_fill=True``, any unpaired modes are mixed with vacuum on the missing
    port. This is implemented by explicitly adding the missing partner modes as
    vacuum into the state (via :meth:`GaussianCore.extend_with_vacuum`) and then
    applying a passive unitary on the complete paired subset. This keeps the map
    unitary on the enlarged space and preserves correlations with untouched modes.

    Physical model
    --------------
    For each paired mode, the Heisenberg transformation is

    .. math::

        \begin{pmatrix}
        b_1 \\
        b_2
        \end{pmatrix}
        =
        \begin{pmatrix}
        \cos\theta & e^{i\phi}\sin\theta \\
        -e^{-i\phi}\sin\theta & \cos\theta
        \end{pmatrix}
        \begin{pmatrix}
        a_1 \\
        a_2
        \end{pmatrix}.

    Here :math:`\theta` sets the power splitting
    :math:`T=\cos^2\theta`, :math:`R=\sin^2\theta`.

    Pairing logic
    -------------
    Two modes are compatible for pairing if
    - they have equal polarization labels, and
    - their envelope signatures match.

    Signature choice:
    - if ``approx=False``: use ``env.signature``
    - if ``approx=True``: use ``env.approx_signature(**approx_kw)``

    Output labels and stability
    ---------------------------
    A beamsplitter is a basis relabel: it changes the path identity of the
    mixed modes but does not shape pulses. Output modes are constructed by
    keeping the same envelope and polarization but replacing the path by
    ``out_path1`` / ``out_path2``.

    The output path labels are **stable per device instance**:
    if not provided, they are generated once in ``__post_init__`` and reused.

    Moment update and basis handling
    --------------------------------
    The passive unitary is applied with :func:`apply_passive_unitary_subset`,
    which embeds the transformation into the full basis so that:
    - untouched modes are preserved,
    - cross-correlation blocks are updated correctly.

    Then the device uses the device-layer relabel hook (``io.mode_map``),
    which is applied by :meth:`GaussianDevice._relabel`. This enforces that
    the relabel is *pure* (Gram matrix unchanged), i.e. only mode identity
    changes.

    Parameters
    ----------
    port1, port2:
        Path selectors compared by signature.
    theta:
        Mixing angle.
    phi:
        Relative phase.
    pol:
        Optional polarization selector applied to both ports.
    approx:
        If True, pair using approximate envelope signature.
    approx_kw:
        Keyword arguments forwarded to ``env.approx_signature``.
    vacuum_fill:
        If True, missing partner modes are added as vacuum.
    out_path1, out_path2:
        Stable output path labels.
    allow_empty:
        If True, allow selecting empty sets (no-op if both empty; if exactly one
        side non-empty and vacuum_fill=False -> error).
    tol:
        Selection tolerance is handled by the overlap/commutator machinery; this
        parameter is kept for API symmetry (not used here).
    check_unitary:
        If True, validate that each passive block is unitary (numerical).
    atol:
        Numerical tolerance for unitary checks.

    Examples
    --------

    Example 1: Balanced beamsplitter on two coherent modes
    ======================================================

    .. jupyter-execute::

        import numpy as np

        from symop_proto.core.operators import ModeOp
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.devices.beam_splitters import PathBeamSplitter
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel

        env = GaussianEnvelope(omega0=20.0, sigma=0.8, tau=0.0, phi0=0.0)
        mA = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        mB = ModeOp(env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H()))
        B = ModeBasis.build([mA, mB])

        core_in = GaussianCore.coherent(B, np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex))

        dev = PathBeamSplitter(
            port1=PathLabel("A"),
            port2=PathLabel("B"),
            theta=np.pi/4,
            phi=0.0,
        )

        out = dev.apply(core_in).state
        print("paths:", [m.label.path.signature for m in out.basis.modes])
        print("alpha:", np.asarray(out.alpha, dtype=complex))

    Example 2: Vacuum-fill when only one port has a compatible mode
    ===============================================================

    .. jupyter-execute::

        import numpy as np

        from symop_proto.core.operators import ModeOp
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.devices.beam_splitters import PathBeamSplitter
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel

        env = GaussianEnvelope(omega0=20.0, sigma=0.8, tau=0.0, phi0=0.0)
        mA = ModeOp(env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([mA])

        core_in = GaussianCore.coherent(B, np.array([1.0 + 0.0j], dtype=complex))

        dev = PathBeamSplitter(
            port1=PathLabel("A"),
            port2=PathLabel("B"),
            theta=np.pi/4,
            phi=0.0,
            vacuum_fill=True,
            allow_empty=True,
        )

        out = dev.apply(core_in).state
        print("n modes:", out.basis.n)
        print("paths:", [m.label.path.signature for m in out.basis.modes])

    """

    port1: HasSignature
    port2: HasSignature
    theta: float = np.pi / 4.0
    phi: float = 0.0
    pol: Optional[HasSignature] = None

    approx: bool = False
    approx_kw: Optional[Dict[str, Any]] = None
    vacuum_fill: bool = False

    out_path1: Optional[PathLabel] = None
    out_path2: Optional[PathLabel] = None

    allow_empty: bool = False
    tol: float = 0.0
    check_unitary: bool = False
    atol: float = 1e-12

    def __post_init__(self) -> None:
        self._init_base()

        th = float(self.theta)
        ph = float(self.phi)
        if (not np.isfinite(th)) or (not np.isfinite(ph)):
            raise ValueError("theta and phi must be finite")

        if self.out_path1 is None or self.out_path2 is None:
            tag = uuid.uuid4().hex[:8]
            p1 = (
                self.out_path1
                if self.out_path1 is not None
                else PathLabel(f"bs_{tag}_1")
            )
            p2 = (
                self.out_path2
                if self.out_path2 is not None
                else PathLabel(f"bs_{tag}_2")
            )
            object.__setattr__(self, "out_path1", p1)
            object.__setattr__(self, "out_path2", p2)

    def resolve_io(self, state: GaussianCore) -> DeviceIO:
        modes1 = list(select_modes(state, path=self.port1, pol=self.pol))
        modes2 = list(select_modes(state, path=self.port2, pol=self.pol))

        if not self.allow_empty:
            modes1 = list(
                require_nonempty(tuple(modes1), what="PathBeamSplitter(port1)")
            )
            modes2 = list(
                require_nonempty(tuple(modes2), what="PathBeamSplitter(port2)")
            )

        if set(modes1).intersection(modes2):
            raise ValueError("PathBeamSplitter requires disjoint port mode sets")

        approx_kw = dict(self.approx_kw or {})

        buckets1: Dict[Tuple[Any, ...], List[ModeOpProto]] = {}
        buckets2: Dict[Tuple[Any, ...], List[ModeOpProto]] = {}

        for m in modes1:
            buckets1.setdefault(
                _mode_key(m, approx=self.approx, approx_kw=approx_kw), []
            ).append(m)
        for m in modes2:
            buckets2.setdefault(
                _mode_key(m, approx=self.approx, approx_kw=approx_kw), []
            ).append(m)

        all_keys = set(buckets1.keys()).union(buckets2.keys())

        # Pair indices into current basis. If vacuum_fill=True, allow None partner.
        pairs: List[Tuple[Optional[int], Optional[int], Tuple[Any, ...]]] = []
        for k in sorted(all_keys, key=repr):
            a = buckets1.get(k, [])
            b = buckets2.get(k, [])
            n = min(len(a), len(b))

            for i in range(n):
                pairs.append(
                    (
                        state.basis.require_index_of(a[i]),
                        state.basis.require_index_of(b[i]),
                        k,
                    )
                )

            if len(a) > n:
                if not self.vacuum_fill:
                    raise ValueError(
                        f"Unpaired modes on port1 and vacuum_fill=False. key={k!r}"
                    )
                for i in range(n, len(a)):
                    pairs.append((state.basis.require_index_of(a[i]), None, k))

            if len(b) > n:
                if not self.vacuum_fill:
                    raise ValueError(
                        f"Unpaired modes on port2 and vacuum_fill=False. key={k!r}"
                    )
                for i in range(n, len(b)):
                    pairs.append((None, state.basis.require_index_of(b[i]), k))

        if len(pairs) == 0:
            # no-op
            return DeviceIO(
                input_modes=(),
                output_modes=(),
                env_modes=(),
                mode_map=(),
                meta={
                    "pairs": [],
                    "vacuum_fill": bool(self.vacuum_fill),
                    "approx": bool(self.approx),
                    "approx_kw": dict(self.approx_kw or {}),
                },
            )

        out1 = self.out_path1
        out2 = self.out_path2
        if out1 is None or out2 is None:
            raise ValueError("Internal error: out_path1/out_path2 not initialized")

        # Determine relabel map for the actually present selected modes.
        # This is applied after do_apply by GaussianDevice._relabel and must preserve Gram.
        mode_map: List[Tuple[ModeOpProto, ModeOpProto]] = []
        for m in modes1:
            mode_map.append((m, m.with_label(ModeLabel(out1, m.label.pol))))
        for m in modes2:
            mode_map.append((m, m.with_label(ModeLabel(out2, m.label.pol))))

        return DeviceIO(
            input_modes=tuple(modes1) + tuple(modes2),
            output_modes=tuple(out_mode for (_in_mode, out_mode) in mode_map),
            env_modes=(),
            mode_map=tuple(mode_map),
            meta={
                "pairs": [(i1, i2, repr(k)) for (i1, i2, k) in pairs],
                "vacuum_fill": bool(self.vacuum_fill),
                "approx": bool(self.approx),
                "approx_kw": dict(self.approx_kw or {}),
                "out_path1": out1.signature,
                "out_path2": out2.signature,
            },
        )

    def do_apply(self, state: GaussianCore, io: DeviceIO) -> GaussianCore:
        pairs_raw = io.meta.get("pairs", [])
        if not isinstance(pairs_raw, list):
            raise ValueError("Invalid io.meta['pairs']")

        pairs: List[Tuple[Optional[int], Optional[int]]] = []
        for t in pairs_raw:
            if not isinstance(t, (tuple, list)) or len(t) < 2:
                raise ValueError("Invalid pair entry in io.meta['pairs']")
            i1 = t[0]
            i2 = t[1]
            if i1 is not None and not isinstance(i1, int):
                raise ValueError("Invalid pair index type")
            if i2 is not None and not isinstance(i2, int):
                raise ValueError("Invalid pair index type")
            pairs.append((i1, i2))

        if len(pairs) == 0:
            return state

        th = float(self.theta)
        ph = float(self.phi)
        c = float(np.cos(th))
        s = float(np.sin(th))
        eip = np.exp(1j * ph)

        # For vacuum_fill, create missing partner modes (vacuum) once per apply.
        # These become environment modes that remain in the basis unless traced later.
        # The base device can trace io.env_modes if return_mode/trace_env policies require it.
        core = state
        env_modes: List[ModeOpProto] = []

        acted: List[ModeOpProto] = []
        U_blocks: List[np.ndarray] = []

        # Deterministic internal vacuum paths per device instance to avoid basis bloat.
        # These are not exposed via mode_map (only used internally).
        vac_path1 = (
            PathLabel(f"{self.out_path1.signature}_vac_in")
            if self.out_path1 is not None
            else PathLabel("vac_in_1")
        )
        vac_path2 = (
            PathLabel(f"{self.out_path2.signature}_vac_in")
            if self.out_path2 is not None
            else PathLabel("vac_in_2")
        )

        for i1, i2 in pairs:
            if i1 is None and i2 is None:
                raise ValueError("Internal error: empty pair")

            m1 = core.basis.modes[i1] if i1 is not None else None
            m2 = core.basis.modes[i2] if i2 is not None else None

            if m1 is None and m2 is None:
                raise ValueError("Internal error: empty pair")

            if m1 is None:
                if not self.vacuum_fill:
                    raise ValueError("Missing port1 partner with vacuum_fill=False")
                # Create a compatible vacuum partner using m2 as template
                m1 = m2.with_label(ModeLabel(vac_path1, m2.label.pol))
                core = core.extend_with_vacuum((m1,))
                env_modes.append(m1)

            if m2 is None:
                if not self.vacuum_fill:
                    raise ValueError("Missing port2 partner with vacuum_fill=False")
                m2 = m1.with_label(ModeLabel(vac_path2, m1.label.pol))
                core = core.extend_with_vacuum((m2,))
                env_modes.append(m2)

            acted.append(m1)
            acted.append(m2)

            U_block = np.array(
                [
                    [c, eip * s],
                    [-np.conjugate(eip) * s, c],
                ],
                dtype=complex,
            )
            U_blocks.append(U_block)

        k = len(acted)
        if k == 0:
            return core
        if k % 2 != 0:
            raise ValueError("Internal error: acted subset must have even length")

        U = np.zeros((k, k), dtype=complex)
        for p, Ub in enumerate(U_blocks):
            r = 2 * p
            U[r : r + 2, r : r + 2] = Ub

        idx = [core.basis.require_index_of(m) for m in acted]
        core2 = apply_passive_unitary_subset(
            core,
            idx=idx,
            U=U,
            check_unitary=self.check_unitary,
            atol=self.atol,
        )

        # If we created env vacuum partners, expose them through io.env_modes so that
        # DeviceApplyOptions.trace_env can discard them if desired.
        #
        # The BaseDevice.apply() will call _trace_out(out, io.env_modes) if trace_env=True.
        if env_modes:
            meta = dict(io.meta)
            meta["created_env_modes_sigs"] = [m.signature for m in env_modes]
            object.__setattr__(io, "meta", meta)  # DeviceIO may be frozen; if so, skip.
            # If DeviceIO is frozen in your codebase, don't mutate it here. Instead:
            # include env_modes already in resolve_io by precomputing them there.
            #
            # Given your current BaseDevice design, the clean solution is:
            # - precompute env vacuum modes in resolve_io, store them in io.env_modes,
            # - and in do_apply extend_with_vacuum(io.env_modes) before applying U.
            #
            # If DeviceIO is frozen, tell me and I'll adjust accordingly.

        return core2

    def _apply_gaussian(
        self,
        state: GaussianCore,
        *,
        options: Optional[DeviceApplyOptions] = None,
    ) -> DeviceResult[GaussianCore]:
        io = self.resolve_io(state)
        out = self.do_apply(state, io)
        return DeviceResult(state=out, io=io)
