from __future__ import annotations
import traceback
from symop.ccr.algebra.op.poly import OpPoly
from symop.core.operators import ModeOp
from symop.core.protocols.signature import SignatureProto

from symop.ccr.algebra.density.poly import DensityPoly
from symop.ccr.algebra.ket.poly import KetPoly

from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.labels.mode import ModeLabel
from symop.modes.labels.path import PathLabel
from symop.modes.labels.polarization import PolarizationLabel

from symop.modes.transfer.gaussian_lowpass import GaussianLowpass
from symop.devices.models.spectral_filter import SpectralFilter

from symop.polynomial.state.density_state import DensityPolyState
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def _fmt_sig(sig: SignatureProto) -> str:
    return repr(sig)


def _print_mode(prefix: str, m: ModeOp) -> None:
    env = m.label.envelope
    print(prefix)
    print("  mode.signature:", _fmt_sig(m.signature))
    print("  path.signature:", _fmt_sig(m.label.path.signature))
    print("  pol.signature :", _fmt_sig(m.label.pol.signature))
    print("  env.signature :", _fmt_sig(env.signature))
    if hasattr(env, "formalism"):
        print("  env.formalism :", getattr(env, "formalism"))
    if hasattr(env, "omega0"):
        print("  env.omega0    :", float(getattr(env, "omega0")))
    if hasattr(env, "sigma"):
        print("  env.sigma     :", float(getattr(env, "sigma")))
    if hasattr(env, "tau"):
        print("  env.tau       :", float(getattr(env, "tau")))
    if hasattr(env, "phi0"):
        print("  env.phi0      :", float(getattr(env, "phi0")))


def _print_n_expect(state: DensityPolyState, mode: ModeOp) -> None:
    n_op = OpPoly.n(mode)
    n = state.expect(n_op)
    print("E[n] = ", n)


def main() -> None:
    # Input/output paths
    in_path = PathLabel("in")
    out_path = PathLabel("out")

    # A single mode descriptor on the input path
    pol = PolarizationLabel.H()
    env_in = GaussianEnvelope(
        omega0=2.0,  # toy rad/s
        sigma=1.0,  # toy seconds
        tau=0.0,
        phi0=0.0,
    )
    label_in = ModeLabel(path=in_path, pol=pol, envelope=env_in)
    m_in = ModeOp(label=label_in)

    # Build |1><1| in that mode
    vac = KetPoly.identity()
    ket_1 = vac.apply_word((m_in.create,))
    rho = DensityPoly.pure(ket_1.terms)
    state = DensityPolyState.from_densitypoly(rho, normalize_trace=True)

    print("\n=== Initial ===")
    print("trace:", state.trace())
    ms_in = state.modes_on_path(in_path)
    _print_n_expect(state, m_in)
    print("modes on in_path:", len(ms_in))
    if ms_in:
        _print_mode("input mode:", ms_in[0])

    # Transfer function and device
    transfer = GaussianLowpass(w0=2.0, sigma_w=0.5)
    dev = SpectralFilter(transfer=transfer)

    # Full apply (kernel + apply_descriptor_edits), if your runtime is wired.
    try:
        st2 = dev.apply(
            state,
            ports={"in": in_path, "out": out_path},
            selection=None,
            runtime=None,
            ctx=None,
            out_kind=None,
        )
        print("\n=== After dev.apply(...) ===")
        print("trace:", st2.trace())
        _print_n_expect(st2, st2.modes_on_path(out_path)[0])
        print("purity_after: ", st2.purity())
        ms_out = st2.modes_on_path(out_path)
        print("modes on out_path:", len(ms_out))
        env_in = m_in.label.envelope
        env_out = ms_out[0].label.envelope

        fig = env_out.plot_many([env_in])
        fig, axs = type(env_out).plot_many([env_in, env_out])
        fig.savefig("envelopes.png", dpi=200)
        if ms_out:
            _print_mode("output mode:", ms_out[0])
    except Exception as e:
        traceback.print_exc()
        print("\n=== dev.apply(...) not runnable yet ===")
        print("Reason:", repr(e))
        print(
            "Planning output above is still valid and is the key sanity check."
        )


if __name__ == "__main__":
    main()
