from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import uuid

import numpy as np

from symop_proto.devices.base import DeviceApplyOptions
from symop_proto.devices.io import DeviceIO, DeviceResult
from symop_proto.envelopes.filtered_envelope import FilteredEnvelope
from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
from symop_proto.envelopes.spectral_filters import GaussianLowpass
from symop_proto.envelopes.spectral_filters.transfer import SpectralTransfer
from symop_proto.gaussian.basis import ModeBasis
from symop_proto.gaussian.core import GaussianCore
from symop_proto.gaussian.devices._utils import require_nonempty, select_modes
from symop_proto.gaussian.devices.base import GaussianDevice
from symop_proto.gaussian.ops.channel_between_bases import (
    apply_ladder_affine_between_bases,
)
from symop_proto.gaussian.ops.gram_blocks import (
    gram_block,
    validate_env_gram_psd,
)
from symop_proto.labels.mode_label import ModeLabel
from symop_proto.labels.path_label import PathLabel


def _default_out_path(in_path: object) -> PathLabel:
    return PathLabel(f"{in_path}_filt_{uuid.uuid4().hex[:8]}")


def _gauss_times_gauss_lowpass(
    env: GaussianEnvelope, H: GaussianLowpass
) -> GaussianEnvelope:
    r"""
    Fast path for Gaussian input envelope and Gaussian lowpass transfer.

    We model filtering in frequency space as

    .. math::

        \zeta_{\mathrm{out}}(\omega)
        =
        H(\omega)\,\zeta_{\mathrm{in}}(\omega).

    Assume both the input spectrum and the transfer function are Gaussian:

    .. math::

        \zeta_{\mathrm{in}}(\omega)
        \propto
        \exp\!\left(
            -\frac{(\omega - \omega_{0,\mathrm{in}})^2}
                  {2\sigma_{\mathrm{in}}^{2}}
        \right),

    .. math::

        H(\omega)
        =
        \exp\!\left(
            -\frac{(\omega - \omega_{0,\mathrm{f}})^2}
                  {2\sigma_{\mathrm{f}}^{2}}
        \right).

    Their product is again Gaussian in :math:`\omega`:

    .. math::

        \zeta_{\mathrm{out}}(\omega)
        \propto
        \exp\!\left(
            -\frac{(\omega - \omega_{0,\mathrm{out}})^2}
                  {2\sigma_{\mathrm{out}}^{2}}
        \right),

    with parameters given by

    .. math::

        \sigma_{\mathrm{out}}^{-2}
        =
        \sigma_{\mathrm{in}}^{-2}
        +
        \sigma_{\mathrm{f}}^{-2},

    .. math::

        \omega_{0,\mathrm{out}}
        =
        \frac{
            \omega_{0,\mathrm{in}}\,\sigma_{\mathrm{in}}^{-2}
            +
            \omega_{0,\mathrm{f}}\,\sigma_{\mathrm{f}}^{-2}
        }{
            \sigma_{\mathrm{in}}^{-2}
            +
            \sigma_{\mathrm{f}}^{-2}
        }.

    The time shift :math:`\tau` and global phase :math:`\phi_0`
    are inherited from the input envelope.

    Notes
    -----
    Any overall amplitude factor in :math:`\zeta_{\mathrm{out}}`
    does not affect CCR-consistent mode construction, because
    Gram blocks are computed from overlaps and the resulting
    channel completion accounts for loss via :math:`C` and
    :math:`G_e`.
    """
    w0_in = float(env.omega0)
    s_in = float(env.sigma)
    w0_f = float(H.w0)
    s_f = float(H.sigma_w)

    if s_in <= 0.0 or s_f <= 0.0:
        raise ValueError("Gaussian widths must be positive")

    a = 1.0 / (s_in * s_in)
    b = 1.0 / (s_f * s_f)

    w0_out = (a * w0_in + b * w0_f) / (a + b)
    s_out = 1.0 / np.sqrt(a + b)

    return GaussianEnvelope(
        omega0=float(w0_out),
        sigma=float(s_out),
        tau=float(env.tau),
        phi0=float(env.phi0),
    )


def _make_output_envelope(env_in: object, transfer: SpectralTransfer) -> object:
    if isinstance(env_in, GaussianEnvelope) and isinstance(transfer, GaussianLowpass):
        return _gauss_times_gauss_lowpass(env_in, transfer)
    return FilteredEnvelope(base=env_in, transfer=transfer)


@dataclass(frozen=True)
class PathSpectralFilter(GaussianDevice):
    r"""
    Spectral filter acting on modes on a selected optical path.

    Physical model
    --------------
    We represent a single optical path as a mode subspace. A spectral filter
    changes the *mode function* (envelope) on that same path:

    .. math::

        \zeta_{\mathrm{out}}(\omega) = H(\omega)\,\zeta_{\mathrm{in}}(\omega).

    The important point for your label physics is:

    - The filter does not create a new spatial path.
    - During the CCR-consistent projection step we must therefore keep the
      *path label* the same so that commutators between input and output
      operators are not artificially forced to zero by label orthogonality.

    After the Gaussian moments have been mapped into the new basis, we may
    optionally relabel the output modes (e.g. add a UUID suffix) as a *pure
    relabeling* step, which must not change the Gram matrix. This is handled
    via BaseDevice._relabel using io.mode_map.

    CCR-consistent basis change with vacuum completion
    --------------------------------------------------
    Let :math:`a` be the selected input annihilation operator vector and
    :math:`b` the output annihilation operator vector (defined by the filtered
    envelopes, but on the same path label). Define Gram blocks:

    .. math::

        (G_{aa})_{ij} = [a_i, a_j^\dagger], \quad
        (G_{bb})_{ij} = [b_i, b_j^\dagger], \quad
        (G_{ba})_{ij} = [b_i, a_j^\dagger].

    We approximate:

    .. math::

        b = C a + e, \qquad C = G_{ba}\,G_{aa}^+,

    where :math:`G_{aa}^+` is the Moore-Penrose pseudo-inverse. Enforcing
    :math:`[b,b^\dagger]=G_{bb}` implies the environment commutator block:

    .. math::

        G_e = [e,e^\dagger] = G_{bb} - C\,G_{aa}\,C^\dagger.

    The environment is assumed vacuum, injecting the minimum noise required
    by CCR (no added thermal noise).

    Induced ladder-affine map on moments
    ------------------------------------
    With stacked ladder vectors :math:`r=(a,a^\dagger)^T` and
    :math:`r'=(b,b^\dagger)^T`, the induced affine map is:

    .. math::

        r' = X r, \qquad K' = X K X^{\mathsf T} + Y,

    with:

    .. math::

        X=\begin{pmatrix} C & 0 \\ 0 & C^* \end{pmatrix}, \qquad
        Y=\begin{pmatrix} 0 & G_e \\ 0 & 0 \end{pmatrix}.

    Output relabeling
    -----------------
    If out_path is provided (or defaulted), the final state is relabeled
    so that the selected modes receive the new path label. This relabeling
    happens only after the Gaussian mapping step and must preserve the Gram
    matrix (pure bookkeeping).

    Parameters
    ----------
    path:
        Input path selector used by select_modes.
    transfer:
        Spectral transfer function H(omega).
    pol:
        Optional polarization selector.
    out_path:
        Optional explicit output path label. If None, a unique path is generated.
    allow_empty:
        If False, raise when no modes are selected.
    rcond:
        Cutoff for the pseudo-inverse in G_aa^+.
    tol:
        Values below tol are treated as zero in Gram blocks / basis build.
    validate_psd:
        If True, validate G_e is PSD within tolerance.
    psd_atol:
        Absolute tolerance for PSD validation.

    Examples
    --------

    Example 1: Coherent state through a Gaussian lowpass
    ====================================================

    This example shows the Gaussian moments (alpha, N, M) before/after filtering, and
    plots the pulse in time and frequency domain. With a Gaussian input envelope and
    GaussianLowpass transfer, the output envelope stays a Gaussian (fast path).

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt

        from symop_proto.core.operators import ModeOp
        from symop_proto.envelopes.base import BaseEnvelope
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.envelopes.spectral_filters import GaussianLowpass
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.devices.filters import PathSpectralFilter
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel

        env_in = GaussianEnvelope(omega0=20.0, sigma=0.8, tau=0.0, phi0=0.0)
        m = ModeOp(env=env_in, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])

        core_in = GaussianCore.coherent(B, np.array([1.0 + 0.2j], dtype=complex))

        H = GaussianLowpass(w0=20.0, sigma_w=0.6)
        dev = PathSpectralFilter(path=PathLabel("A"), transfer=H)

        res = dev.apply(core_in)
        core_out = res.state

        print("=== INPUT CORE ===")
        print("alpha:\n", np.asarray(core_in.alpha, dtype=complex))
        print("N:\n", np.asarray(core_in.N, dtype=complex))
        print("M:\n", np.asarray(core_in.M, dtype=complex))

        print("\n=== OUTPUT CORE ===")
        print("alpha:\n", np.asarray(core_out.alpha, dtype=complex))
        print("N:\n", np.asarray(core_out.N, dtype=complex))
        print("M:\n", np.asarray(core_out.M, dtype=complex))

        env_out = core_out.basis.modes[0].env

        BaseEnvelope.plot_many(
            [env_in, env_out],
            labels=["in", "out"],
            normalize_envelope=True,
            show_real_imag=True,
            show_phase=False,
            show_formula=True,
            title="Time-domain envelopes (normalized)",
        )

        w0 = float(getattr(env_in, "omega0", 0.0))
        w = np.linspace(w0 - 6.0, w0 + 6.0, 2000, dtype=float)

        amp_in = np.abs(env_in.freq_eval(w))
        amp_out = np.abs(env_out.freq_eval(w))

        amp_in = amp_in / max(float(np.max(amp_in)), 1e-30)
        amp_out = amp_out / max(float(np.max(amp_out)), 1e-30)

        plt.figure(figsize=(8, 3))
        plt.plot(w, amp_in, label="|z_in(w)| (norm)")
        plt.plot(w, amp_out, label="|z_out(w)| (norm)")
        plt.xlabel("omega")
        plt.ylabel("normalized magnitude")
        plt.title("Spectral magnitude before/after")
        plt.legend()
        plt.tight_layout()

    Examples
    --------

    Example 1: Coherent state through GaussianLowpass (fast path)
    =============================================================

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt

        from symop_proto.core.operators import ModeOp
        from symop_proto.envelopes.base import BaseEnvelope
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.envelopes.spectral_filters import GaussianLowpass
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.devices.filters import PathSpectralFilter
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel

        env_in = GaussianEnvelope(omega0=20.0, sigma=0.8, tau=0.0, phi0=0.0)
        m = ModeOp(env=env_in, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])

        core_in = GaussianCore.coherent(B, np.array([1.0 + 0.2j], dtype=complex))

        H = GaussianLowpass(w0=20.0, sigma_w=0.6)
        dev = PathSpectralFilter(path=PathLabel("A"), transfer=H)

        res = dev.apply(core_in)
        core_out = res.state

        print("=== INPUT CORE ===")
        print("alpha:\n", np.asarray(core_in.alpha, dtype=complex))
        print("N:\n", np.asarray(core_in.N, dtype=complex))
        print("M:\n", np.asarray(core_in.M, dtype=complex))

        print("\n=== OUTPUT CORE ===")
        print("alpha:\n", np.asarray(core_out.alpha, dtype=complex))
        print("N:\n", np.asarray(core_out.N, dtype=complex))
        print("M:\n", np.asarray(core_out.M, dtype=complex))

        env_out = core_out.basis.modes[0].env

        BaseEnvelope.plot_many(
            [env_in, env_out],
            labels=["in", "out"],
            normalize_envelope=True,
            show_real_imag=True,
            show_phase=False,
            show_formula=True,
            title="Example 1: time-domain envelopes (normalized)",
        )

        w0 = float(getattr(env_in, "omega0", 0.0))
        w = np.linspace(w0 - 6.0, w0 + 6.0, 2000, dtype=float)

        amp_in = np.abs(env_in.freq_eval(w))
        amp_out = np.abs(env_out.freq_eval(w))

        amp_in = amp_in / max(float(np.max(amp_in)), 1e-30)
        amp_out = amp_out / max(float(np.max(amp_out)), 1e-30)

        plt.figure(figsize=(8, 3))
        plt.plot(w, amp_in, label="|z_in(w)| (norm)")
        plt.plot(w, amp_out, label="|z_out(w)| (norm)")
        plt.xlabel("omega")
        plt.ylabel("normalized magnitude")
        plt.title("Example 1: spectral magnitude before/after")
        plt.legend()
        plt.tight_layout()


    Example 2: Coherent state through a smooth bandpass (FilteredEnvelope fallback)
    ==============================================================================

    This uses a custom transfer (bounded, smooth) so the output envelope is
    represented as a FilteredEnvelope.

    If your FilteredEnvelope overlap currently relies on a numeric time-domain
    fallback, discretization can introduce tiny non-Hermitian noise; for a stable
    documentation example we disable PSD validation.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt

        from symop_proto.core.operators import ModeOp
        from symop_proto.envelopes.base import BaseEnvelope
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.envelopes.spectral_filters.transfer import SpectralTransfer
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.devices.filters import PathSpectralFilter
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel

        def estimate_inst_omega(t: np.ndarray, z: np.ndarray, *, frac: float = 0.2) -> float:
            t = np.asarray(t, dtype=float)
            z = np.asarray(z, dtype=complex)

            amp = np.abs(z)
            i0 = int(np.argmax(amp))
            m = max(16, int(frac * t.size))
            lo = max(0, i0 - m // 2)
            hi = min(t.size, i0 + m // 2)

            tt = t[lo:hi]
            zz = z[lo:hi]

            mask = np.abs(zz) > (np.max(np.abs(zz)) * 1e-3)
            tt = tt[mask]
            zz = zz[mask]
            if tt.size < 16:
                return float("nan")

            phi = np.unwrap(np.angle(zz))
            tt0 = tt - float(np.mean(tt))
            denom = float(np.dot(tt0, tt0))
            if denom <= 0.0:
                return float("nan")
            slope = float(np.dot(tt0, phi - float(np.mean(phi))) / denom)
            return slope  # because your time convention is exp(+i omega t)

        class RaisedCosineBandpass:
            def __init__(self, w_center: float, half_width: float, roll: float):
                self.w_center = float(w_center)
                self.half_width = float(half_width)
                self.roll = float(roll)

            @property
            def signature(self):
                return ("rc_bandpass", self.w_center, self.half_width, self.roll)

            def __call__(self, w: np.ndarray) -> np.ndarray:
                w = np.asarray(w, dtype=float)

                wc = float(self.w_center)
                hw = float(self.half_width)
                r = float(self.roll)

                if not np.isfinite(w).all():
                    bad = np.argwhere(~np.isfinite(w))
                    i = int(bad[0, 0])
                    raise ValueError(f"Non-finite input w at index {i}: {w[i]!r}")

                if not (hw > 0.0) or not np.isfinite(hw):
                    raise ValueError(f"half_width must be positive finite, got {hw!r}")

                # Treat very small roll as a hard cutoff to avoid division issues.
                if (not np.isfinite(r)) or r <= 1e-15:
                    H = (np.abs(w - wc) <= hw).astype(float)
                    return H.astype(complex)

                x = np.abs(w - wc)

                H = np.zeros_like(w, dtype=float)
                pass_mask = x <= hw
                tran_mask = (x > hw) & (x < hw + r)

                H[pass_mask] = 1.0

                xi = (x[tran_mask] - hw) / r  # nominally in (0,1)
                # Numerical safety (should already be in range, but clip avoids edge weirdness)
                xi = np.clip(xi, 0.0, 1.0)

                H[tran_mask] = 0.5 * (1.0 + np.cos(np.pi * xi))
                return H.astype(complex)

        env_in = GaussianEnvelope(omega0=20.0, sigma=0.8, tau=0.0, phi0=0.0)
        m = ModeOp(env=env_in, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])

        core_in = GaussianCore.coherent(B, np.array([1.0 + 0.2j], dtype=complex))

        H = RaisedCosineBandpass(w_center=20.0, half_width=0.9, roll=0.5)
        dev = PathSpectralFilter(
            path=PathLabel("A"),
            transfer=H,
            validate_psd=False,
        )

        res = dev.apply(core_in)
        core_out = res.state

        print("=== INPUT CORE ===")
        print("alpha:\n", np.asarray(core_in.alpha, dtype=complex))
        print("N:\n", np.asarray(core_in.N, dtype=complex))
        print("M:\n", np.asarray(core_in.M, dtype=complex))

        print("\n=== OUTPUT CORE ===")
        print("alpha:\n", np.asarray(core_out.alpha, dtype=complex))
        print("N:\n", np.asarray(core_out.N, dtype=complex))
        print("M:\n", np.asarray(core_out.M, dtype=complex))

        t = np.linspace(-30.0, 30.0, 20000, dtype=float)
        zin = env_in.time_eval(t)
        zout = env_out.time_eval(t)

        print("omega_inst(in)  =", estimate_inst_omega(t, zin))
        print("omega_inst(out) =", estimate_inst_omega(t, zout))

        env_out = core_out.basis.modes[0].env

        t = np.linspace(-30.0, 30.0, 20000, dtype=float)
        BaseEnvelope.plot_many(
            [env_in, env_out],
            t=t,
            labels=["in", "out"],
            normalize_envelope=False,
            show_real_imag=True,
            show_phase=True,
            show_formula=True,
            title="Example 2: time-domain envelopes (normalized)",
        )

        w0 = float(getattr(env_in, "omega0", 0.0))
        w = np.linspace(w0 - 6.0, w0 + 6.0, 2000, dtype=float)

        amp_in = np.abs(env_in.freq_eval(w))
        amp_out = np.abs(env_out.freq_eval(w))

        amp_in = amp_in / max(float(np.max(amp_in)), 1e-30)
        amp_out = amp_out / max(float(np.max(amp_out)), 1e-30)

        plt.figure(figsize=(8, 3))
        plt.plot(w, amp_in, label="|z_in(w)| (norm)")
        plt.plot(w, amp_out, label="|z_out(w)| (norm)")
        plt.xlabel("omega")
        plt.ylabel("normalized magnitude")
        plt.title("Example 2: spectral magnitude before/after")
        plt.legend()
        plt.tight_layout()

    Example 3: Coherent state through a single-pole cavity (Lorentzian) filter
    =========================================================================

    This models a simple cavity-like spectral response:

    .. math::

        H(\omega) = \sqrt(\eta) \frac{\kappa}{2(\frac{\kappa}{2}- i (\omega - \omega_c))}


    - :mat:`|H|` is Lorentzian in frequency
    - arg(H) varies rapidly near resonance -> adds group delay
    - time-domain pulse acquires an asymmetric tail

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt

        from symop_proto.core.operators import ModeOp
        from symop_proto.envelopes.base import BaseEnvelope
        from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
        from symop_proto.gaussian.basis import ModeBasis
        from symop_proto.gaussian.core import GaussianCore
        from symop_proto.gaussian.devices.filters import PathSpectralFilter
        from symop_proto.labels.mode_label import ModeLabel
        from symop_proto.labels.path_label import PathLabel
        from symop_proto.labels.polarization_label import PolarizationLabel


        class SinglePoleCavity:
            def __init__(self, wc: float, kappa: float, eta: float = 1.0):
                self.wc = float(wc)
                self.kappa = float(kappa)
                self.eta = float(eta)

            @property
            def signature(self):
                return ("single_pole_cavity", self.wc, self.kappa, self.eta)

            def __call__(self, w: np.ndarray) -> np.ndarray:
                w = np.asarray(w, dtype=float)
                wc = float(self.wc)
                kappa = float(self.kappa)
                eta = float(self.eta)

                if not (kappa > 0.0) or not np.isfinite(kappa):
                    raise ValueError(f"kappa must be positive finite, got {kappa!r}")
                if not (eta >= 0.0) or not np.isfinite(eta):
                    raise ValueError(f"eta must be finite and >= 0, got {eta!r}")

                dw = w - wc
                # Complex transfer; magnitude is Lorentzian, phase gives dispersion/group delay.
                H = (0.5 * kappa) / ((0.5 * kappa) - 1j * dw)
                H = np.sqrt(eta) * H
                return H.astype(complex)


        # Input mode
        env_in = GaussianEnvelope(omega0=20.0, sigma=0.8, tau=0.0, phi0=0.0)
        m = ModeOp(env=env_in, label=ModeLabel(PathLabel("A"), PolarizationLabel.H()))
        B = ModeBasis.build([m])

        core_in = GaussianCore.coherent(B, np.array([1.0 + 0.2j], dtype=complex))

        # Cavity filter: centered near the carrier, linewidth controls how strong the distortion is
        H = SinglePoleCavity(wc=20.0, kappa=0.8, eta=0.9)

        dev = PathSpectralFilter(
            path=PathLabel("A"),
            transfer=H,
            # numeric envelopes can introduce tiny PSD issues depending on overlap backend
            validate_psd=False,
        )

        res = dev.apply(core_in)
        core_out = res.state

        print("=== INPUT CORE ===")
        print("alpha:\n", np.asarray(core_in.alpha, dtype=complex))
        print("N:\n", np.asarray(core_in.N, dtype=complex))
        print("M:\n", np.asarray(core_in.M, dtype=complex))

        print("\n=== OUTPUT CORE ===")
        print("alpha:\n", np.asarray(core_out.alpha, dtype=complex))
        print("N:\n", np.asarray(core_out.N, dtype=complex))
        print("M:\n", np.asarray(core_out.M, dtype=complex))

        env_out = core_out.basis.modes[0].env

        # Time-domain plot (use a fairly fine grid to see the tail)
        t = np.linspace(-30.0, 30.0, 20000, dtype=float)
        BaseEnvelope.plot_many(
            [env_in, env_out],
            t=t,
            labels=["in", "out"],
            normalize_envelope=True,
            show_real_imag=True,
            show_phase=True,
            show_formula=True,
            title="Example 3: single-pole cavity filtering (time domain)",
        )

        # Show transfer magnitude/phase near the carrier
        w0 = float(getattr(env_in, "omega0", 0.0))
        w = np.linspace(w0 - 6.0, w0 + 6.0, 4000, dtype=float)

        Hw = np.asarray(H(w), dtype=complex)
        ampH = np.abs(Hw)
        phH = np.unwrap(np.angle(Hw))

        plt.figure(figsize=(8, 3))
        plt.plot(w, ampH / max(float(np.max(ampH)), 1e-30), label="|H(w)| (norm)")
        plt.xlabel("omega")
        plt.ylabel("normalized magnitude")
        plt.title("Example 3: cavity transfer magnitude")
        plt.legend()
        plt.tight_layout()

        plt.figure(figsize=(8, 3))
        plt.plot(w, phH, label="arg H(w)")
        plt.xlabel("omega")
        plt.ylabel("phase [rad]")
        plt.title("Example 3: cavity transfer phase")
        plt.legend()
        plt.tight_layout()

        # Spectral magnitude before/after
        amp_in = np.abs(env_in.freq_eval(w))
        amp_out = np.abs(env_out.freq_eval(w))
        amp_in = amp_in / max(float(np.max(amp_in)), 1e-30)
        amp_out = amp_out / max(float(np.max(amp_out)), 1e-30)

        plt.figure(figsize=(8, 3))
        plt.plot(w, amp_in, label="|z_in(w)| (norm)")
        plt.plot(w, amp_out, label="|z_out(w)| (norm)")
        plt.xlabel("omega")
        plt.ylabel("normalized magnitude")
        plt.title("Example 3: spectral magnitude before/after")
        plt.legend()
        plt.tight_layout()
    """

    path: object
    transfer: SpectralTransfer
    pol: Optional[object] = None
    out_path: Optional[PathLabel] = None
    allow_empty: bool = False
    rcond: float = 1e-12
    tol: float = 0.0
    validate_psd: bool = True
    psd_atol: float = 1e-12

    def __post_init__(self) -> None:
        self._init_base()

    def resolve_io(self, state: GaussianCore) -> DeviceIO:
        """
        Resolve concrete input modes and construct output modes.

        Key rule:
        - During do_apply we keep the physical path labels unchanged so that
          commutators between input and output operators are not zeroed by label
          orthogonality.
        - We still prepare a mode_map so BaseDevice.apply can relabel after the
          mapping step, if desired.
        """
        modes_in = select_modes(state, path=self.path, pol=self.pol)
        if not self.allow_empty:
            modes_in = require_nonempty(modes_in, what="PathSpectralFilter")

        # This is the bookkeeping output path. It is applied only at the end via relabel.
        final_out_path = (
            self.out_path if self.out_path is not None else _default_out_path(self.path)
        )

        modes_out = []
        mode_map = []

        for m in modes_in:
            env_out = _make_output_envelope(m.env, self.transfer)

            # Step 1: physical label for CCR mapping (keep same path, same pol)
            in_lbl = getattr(m, "label", None)
            if in_lbl is None:
                raise ValueError("PathSpectralFilter expects modes with a .label")

            in_path = getattr(in_lbl, "path", None)
            in_pol = getattr(in_lbl, "pol", None)

            phys_lbl = ModeLabel(in_path, in_pol)
            m_phys = type(m)(env=env_out, label=phys_lbl)
            modes_out.append(m_phys)

            # Step 2: bookkeeping relabel to a distinct path at the very end
            book_lbl = ModeLabel(final_out_path, in_pol)
            m_book = type(m)(env=env_out, label=book_lbl)
            mode_map.append((m_phys, m_book))

        return DeviceIO(
            input_modes=tuple(modes_in),
            output_modes=tuple(modes_out),
            env_modes=(),
            mode_map=tuple(mode_map),
            meta={
                "rcond": float(self.rcond),
                "tol": float(self.tol),
                "validate_psd": bool(self.validate_psd),
                "out_path": (
                    final_out_path.signature
                    if hasattr(final_out_path, "signature")
                    else str(final_out_path)
                ),
                "transfer": self.transfer.signature,
            },
        )

    def do_apply(self, state: GaussianCore, io: DeviceIO) -> GaussianCore:
        """
        Apply the CCR-consistent Gaussian map.

        Note: io.output_modes use the *physical* label (same path as input) so
        that G_ba captures envelope overlap rather than being forced to zero by
        path-orthogonality. The bookkeeping relabel is handled after this step
        by BaseDevice.apply via io.mode_map.
        """
        modes_in = tuple(io.input_modes)
        modes_out = tuple(io.output_modes)

        idx_in = [state.basis.require_index_of(m) for m in modes_in]
        basis_out = ModeBasis.build(modes_out, tol=self.tol)

        G_aa = gram_block(modes_in, modes_in, tol=self.tol)
        G_bb = gram_block(modes_out, modes_out, tol=self.tol)
        G_ba = gram_block(modes_out, modes_in, tol=self.tol)

        G_aa_pinv = np.linalg.pinv(G_aa, rcond=float(self.rcond))
        C = G_ba @ G_aa_pinv

        G_e = G_bb - C @ G_aa @ C.conj().T
        G_e = 0.5 * (G_e + G_e.conj().T)

        def _assert_finite(name: str, A: np.ndarray) -> None:
            if not np.isfinite(A).all():
                bad = np.argwhere(~np.isfinite(A))
                i, j = int(bad[0, 0]), int(bad[0, 1])
                raise ValueError(f"{name} has non-finite at {(i, j)}: {A[i, j]!r}")

        _assert_finite("G_aa", G_aa)
        _assert_finite("G_bb", G_bb)
        _assert_finite("G_ba", G_ba)
        ov = modes_out[0].env.overlap(modes_in[0].env)
        print("env overlap:", ov, "finite:", np.isfinite(ov))
        if self.validate_psd:
            validate_env_gram_psd(G_e, atol=self.psd_atol)

        k = len(modes_in)
        m = len(modes_out)

        X = np.zeros((2 * m, 2 * k), dtype=complex)
        X[0:m, 0:k] = C
        X[m : 2 * m, k : 2 * k] = C.conj()

        Y = np.zeros((2 * m, 2 * m), dtype=complex)
        Y[0:m, m : 2 * m] = G_e

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
