from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from tqdm import tqdm


# -----------------------------
# Helpers
# -----------------------------
def theta_from_eta(eta: float) -> float:
    """
    Convert power transmittance eta in [0,1] to beamsplitter angle theta.

    Convention:
        t = cos(theta)  ->  eta = |t|^2 = cos(theta)^2
    """
    eta_f = float(eta)
    if not np.isfinite(eta_f) or eta_f < 0.0 or eta_f > 1.0:
        raise ValueError(f"eta must be finite and in [0,1], got {eta!r}")
    return float(np.arccos(np.sqrt(eta_f)))


def _mode_path_name(mode: object) -> Optional[str]:
    lbl = getattr(mode, "label", None)
    if lbl is None:
        return None
    p = getattr(lbl, "path", None)
    if p is None:
        return None
    return getattr(p, "name", None)


def _find_mode_by_exact_path(core: object, path_name: str) -> object:
    for m in core.basis.modes:
        if _mode_path_name(m) == path_name:
            return m
    known = [_mode_path_name(m) for m in core.basis.modes]
    raise KeyError(f"No mode with path={path_name!r}. Known: {known!r}")


def _find_mode_via_mode_map_for_input_path(
    *,
    mode_map: Sequence[Tuple[object, object]],
    in_path_name: str,
) -> object:
    """
    Find the bookkeeping-mode (second element) corresponding to a physical-mode
    (first element) whose label.path.name == in_path_name.
    """
    for m_phys, m_book in mode_map:
        if _mode_path_name(m_phys) == in_path_name:
            return m_book
    phys_known = [_mode_path_name(p) for (p, _) in mode_map]
    raise KeyError(
        f"No mapped physical mode with path={in_path_name!r}. "
        f"Known physical paths: {phys_known!r}"
    )


def _outcome_moments_xp(
    core: object,
    mode: object,
    *,
    meas_var: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytic outcome distribution for measuring (x,p) of `mode`.

    From condition_on_quadratures:
        y ~ N(d_B, S),  S = V_BB + Vm
    where Vm = meas_var * I for xp measurement.

    Returns (mean, cov) with shapes (2,), (2,2).
    """
    from symop_proto.gaussian.ops.measurement import quadrature_indices

    d = np.asarray(core.quadrature_mean(), dtype=float).reshape(-1)
    V = np.asarray(core.quadrature_covariance(), dtype=float)

    q_idx = np.asarray(
        quadrature_indices(core, (mode,), which="xp"), dtype=int
    ).reshape(-1)
    if q_idx.size != 2:
        raise ValueError("xp measurement on a single mode must give 2 indices")

    dB = d[q_idx].copy()
    VBB = V[np.ix_(q_idx, q_idx)].copy()

    mv = float(meas_var)
    if mv < 0.0:
        raise ValueError("meas_var must be >= 0")
    Vm = mv * np.eye(2, dtype=float)

    S = VBB + Vm
    S = 0.5 * (S + S.T)
    return dB, S


# -----------------------------
# Demo configuration
# -----------------------------
@dataclass(frozen=True)
class DemoParams:
    squeeze_r: float = 0.6
    eta_mix: float = 0.5
    meas_var: float = 0.5
    omega0: float = 20.0
    sigma: float = 0.8


def _build_initial_core(params: DemoParams):
    from symop_proto.core.operators import ModeOp
    from symop_proto.envelopes.gaussian_envelope import GaussianEnvelope
    from symop_proto.gaussian.basis import ModeBasis
    from symop_proto.gaussian.core import GaussianCore
    from symop_proto.labels.mode_label import ModeLabel
    from symop_proto.labels.path_label import PathLabel
    from symop_proto.labels.polarization_label import PolarizationLabel

    env = GaussianEnvelope(
        omega0=float(params.omega0),
        sigma=float(params.sigma),
        tau=0.0,
        phi0=0.0,
    )

    mA = ModeOp(
        env=env, label=ModeLabel(PathLabel("A"), PolarizationLabel.H())
    )
    mB = ModeOp(
        env=env, label=ModeLabel(PathLabel("B"), PolarizationLabel.H())
    )
    mE = ModeOp(
        env=env, label=ModeLabel(PathLabel("E"), PolarizationLabel.H())
    )

    basis = ModeBasis.build([mA, mB, mE])
    core = GaussianCore.vacuum(basis)
    return core, mA, mB, mE


def _apply_two_mode_squeezing(core, mA, mB, r: float):
    from symop_proto.gaussian.maps.bogoliubov import TwoModeSqueezer

    op = TwoModeSqueezer(
        mode1=mA, mode2=mB, r=float(r), phi=0.0, check_ccr=True
    )
    return op.apply(core)


def _apply_jammer_displacement(core, mE, beta: complex):
    from symop_proto.gaussian.maps.channel import Displacement

    op = Displacement(modes=(mE,), beta=np.asarray([beta], dtype=complex))
    return op.apply(core)


def _mix_idler_with_jammer(core, *, eta: float):
    from symop_proto.gaussian.devices.beam_splitters import PathBeamSplitter
    from symop_proto.labels.path_label import PathLabel

    theta = theta_from_eta(float(eta))
    bs = PathBeamSplitter(
        port1=PathLabel("B"),
        port2=PathLabel("E"),
        theta=theta,
        phi=0.0,
        vacuum_fill=False,
        allow_empty=False,
        approx=False,
    )
    res = bs.apply(core)
    return res.state, res.io


def run_demo(params: DemoParams) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for this example")

    jammer_powers = np.linspace(0.0, 6.0, 31, dtype=float)

    # Florian-faithful quantities for coherent jamming:
    # - outcome covariance determinant should be flat vs power
    # - outcome mean norm^2 should grow vs power
    det_cov = np.zeros_like(jammer_powers, dtype=float)
    mean_norm2 = np.zeros_like(jammer_powers, dtype=float)

    for i, nbar in enumerate(tqdm(jammer_powers, desc="Jammer power")):
        core, mA0, mB0, mE0 = _build_initial_core(params)
        core = _apply_two_mode_squeezing(core, mA0, mB0, params.squeeze_r)

        beta = np.sqrt(float(nbar)) + 0.0j
        core = _apply_jammer_displacement(core, mE0, beta)

        core2, io_bs = _mix_idler_with_jammer(core, eta=params.eta_mix)

        mB_out = _find_mode_via_mode_map_for_input_path(
            mode_map=io_bs.mode_map, in_path_name="B"
        )

        mu, S = _outcome_moments_xp(core2, mB_out, meas_var=params.meas_var)

        mean_norm2[i] = float(mu[0] * mu[0] + mu[1] * mu[1])
        det_cov[i] = float(np.linalg.det(S))

    plt.figure(figsize=(7.6, 3.2))
    plt.plot(jammer_powers, det_cov, marker="o", markersize=3)
    plt.xlabel("jammer mean photons |beta|^2")
    plt.ylabel("det( Cov[outcome xp] )")
    plt.title(
        "Outcome covariance determinant vs jammer strength (coherent jammer)"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7.6, 3.2))
    plt.plot(jammer_powers, mean_norm2, marker="o", markersize=3)
    plt.xlabel("jammer mean photons |beta|^2")
    plt.ylabel("||E[outcome]||^2")
    plt.title("Outcome mean shift vs jammer strength (coherent jammer)")
    plt.tight_layout()
    plt.show()

    print("Done.")
    print(
        f"params: squeeze_r={params.squeeze_r}, eta_mix={
            params.eta_mix}, meas_var={params.meas_var}"
    )
    print(
        f"det_cov at nbar={float(jammer_powers[0]):.3g}: {
            float(det_cov[0]):.6g}"
    )
    print(
        f"det_cov at nbar={
            float(jammer_powers[-1]):.3g}: {float(det_cov[-1]):.6g}"
    )
    print(
        f"mean_norm2 at nbar={float(jammer_powers[0]):.3g}: {
            float(mean_norm2[0]):.6g}"
    )
    print(
        f"mean_norm2 at nbar={
            float(jammer_powers[-1]):.3g}: {float(mean_norm2[-1]):.6g}"
    )


if __name__ == "__main__":
    run_demo(
        DemoParams(
            squeeze_r=0.6,
            eta_mix=0.5,
            meas_var=0.5,
        )
    )
