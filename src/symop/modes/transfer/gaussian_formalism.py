r"""Closed-form Gaussian transfer expansions for gaussian_closed envelopes.

This module provides a small algebra for staying inside the
``"gaussian_closed"`` envelope formalism when applying certain frequency-domain
transfer functions.

Core idea
---------
Let :math:`\zeta(t)` be a normalized mode descriptor and :math:`Z(\omega)` its
spectrum under the package convention

.. math::

    Z(\omega) = \int_{-\infty}^{\infty} \zeta(t)\,e^{+i\omega t}\,dt,

so that the mode inner product can be evaluated in the frequency domain as

.. math::

    \langle \zeta_1, \zeta_2 \rangle
    = \frac{1}{2\pi}\int_{-\infty}^{\infty}
      \overline{Z_1(\omega)}\,Z_2(\omega)\,d\omega.

Filtering a mode by an amplitude transfer :math:`H(\omega)` produces an
*unnormalized* spectrum

.. math::

    Z_{\mathrm{raw}}(\omega) = H(\omega)\,Z_{\mathrm{in}}(\omega).

Because envelopes are treated as **mode descriptors**, we keep the envelope
normalized and return the power transmission separately:

.. math::

    \eta = \langle \zeta_{\mathrm{raw}}, \zeta_{\mathrm{raw}} \rangle
         = \frac{1}{2\pi}\int |H(\omega)|^2\,|Z_{\mathrm{in}}(\omega)|^2\,d\omega,
    \qquad
    \zeta_{\mathrm{out}} = \frac{\zeta_{\mathrm{raw}}}{\sqrt{\eta}}.

The quantum state is then damped by a loss channel with transmissivity
:math:`\eta`, while the envelope remains normalized.

Supported closed-form family
----------------------------
This module targets transfers representable as a constant plus a finite sum
of Gaussian low-pass "atoms":

.. math::

    H(\omega) = c_0 + \sum_{k=1}^{K} c_k
    \exp\left[-\frac{1}{2}\left(\frac{\omega-\omega_k}{\sigma_k}\right)^2\right].

Applying such a transfer to a single :class:`~symop.modes.envelopes.gaussian.GaussianEnvelope`
produces a :class:`~symop.modes.envelopes.gaussian_mixture.GaussianMixtureEnvelope`
in closed form (a finite linear combination of normalized Gaussian components).

Pruning
-------
Expansions can create mixtures with many components (especially when applying
to an already-mixture input). We optionally prune components whose raw
contribution is negligible using a conservative proxy:

.. math::

    p_i \approx |w_i|^2,

since each component is individually normalized. Pruning is applied to the
*returned envelope representation* only. The returned :math:`\eta` is computed
from the full unpruned raw mixture by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from symop.modes.envelopes.gaussian import GaussianEnvelope
from symop.modes.envelopes.gaussian_mixture import GaussianMixtureEnvelope
from symop.modes.protocols.envelope import SupportsGaussianClosedOverlap


def _clip_eta(x: float) -> float:
    r"""Clip a scalar to :math:`[0,1]` with non-finite mapped to 0.

    Parameters
    ----------
    x:
        Candidate transmissivity.

    Returns
    -------
    float
        Value clipped to :math:`[0, 1]`.

    Notes
    -----
    In exact arithmetic, the transmissivity

    .. math::

        \eta = \frac{1}{2\pi}\int |H(\omega)|^2 |Z(\omega)|^2 d\omega

    should lie in :math:`[0,1]` for passive amplitude transfers with
    :math:`|H|\le 1`. In floating-point arithmetic and for "semantic" filters
    (e.g. complements), tiny violations can occur; clipping keeps the
    higher-level semantics stable.

    """
    if not math.isfinite(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def prune_mixture_terms(
    comps: list[GaussianEnvelope],
    weights: list[complex],
    *,
    rel_weight2_min: float = 1e-12,
    abs_weight2_min: float = 0.0,
    max_terms: int | None = 128,
    keep_at_least: int = 1,
) -> tuple[tuple[GaussianEnvelope, ...], np.ndarray]:
    r"""Prune mixture terms by a conservative diagonal power proxy.

    Given a raw mixture

    .. math::

        \zeta_{\mathrm{raw}}(t) = \sum_{i=1}^{N} w_i\,g_i(t),

    with normalized components :math:`\langle g_i, g_i\rangle = 1`, the diagonal
    contribution to :math:`\|\zeta_{\mathrm{raw}}\|^2` is approximately

    .. math::

        p_i \approx |w_i|^2.

    This function discards terms whose :math:`p_i` is far below the maximum.

    Parameters
    ----------
    comps:
        List of Gaussian components :math:`g_i`.
    weights:
        List of complex weights :math:`w_i` (raw, unnormalized).
    rel_weight2_min:
        Relative threshold on :math:`|w_i|^2` compared to :math:`\max_j |w_j|^2`.
    abs_weight2_min:
        Absolute threshold on :math:`|w_i|^2`.
    max_terms:
        If not None, cap the number of kept terms by keeping the largest
        :math:`|w_i|^2` terms after thresholding.
    keep_at_least:
        Ensure at least this many terms remain (keeps the largest terms if needed).

    Returns
    -------
    (components, weights):
        Pruned components (tuple) and corresponding weights (1D complex array).

    Notes
    -----
    - This is a heuristic: it ignores cross terms
      :math:`\overline{w_i}w_j\langle g_i, g_j\rangle`. Use conservative
      thresholds.
    - Pruning should be viewed as a representation-compression step. For safety,
      compute :math:`\eta` from the full unpruned raw mixture first, then prune
      the returned envelope representation.

    """
    if len(comps) != len(weights):
        raise ValueError("comps and weights length mismatch")
    n = len(comps)
    if n == 0:
        return tuple(), np.asarray([], dtype=complex)

    w = np.asarray(weights, dtype=complex)
    p = np.abs(w) ** 2

    p_max = float(np.max(p))
    if not math.isfinite(p_max) or p_max <= 0.0:
        # Keep minimal structure if everything is degenerate.
        k = max(1, min(int(keep_at_least), n))
        idx = np.arange(k, dtype=int)
        return tuple(comps[i] for i in idx), w[idx]

    thr = max(float(abs_weight2_min), float(rel_weight2_min) * p_max)
    idx_keep = np.flatnonzero(p >= thr)

    if idx_keep.size < int(keep_at_least):
        idx_keep = np.argsort(p)[::-1][: int(keep_at_least)]

    if max_terms is not None and idx_keep.size > int(max_terms):
        idx_sorted = idx_keep[np.argsort(p[idx_keep])[::-1]]
        idx_keep = idx_sorted[: int(max_terms)]

    idx_keep = np.asarray(idx_keep, dtype=int)
    return tuple(comps[i] for i in idx_keep), w[idx_keep]


def _gauss_atom_times_gauss_env(
    env: GaussianEnvelope,
    *,
    w0: float,
    sigma_w: float,
) -> tuple[GaussianEnvelope, float]:
    r"""Apply a Gaussian low-pass *atom* to a single Gaussian envelope (closed form).

    The atom transfer is

    .. math::

        H(\omega) = \exp\left[-\frac{1}{2}\left(\frac{\omega-w_0}{\sigma_\omega}\right)^2\right].

    For the :class:`~symop.modes.envelopes.gaussian.GaussianEnvelope` spectral shape
    used in this package

    .. math::

        Z(\omega)\propto \exp\left[-\sigma_t^2(\omega-\omega_0)^2\right]e^{-i\omega\tau}e^{i\phi_0},

    the product :math:`H(\omega)Z(\omega)` remains Gaussian in :math:`\omega`.
    Therefore there exists a normalized output Gaussian envelope :math:`g_{\mathrm{out}}`
    and a scalar :math:`\eta\in[0,1]` such that

    .. math::

        H(\omega)Z(\omega) = \sqrt{\eta}\,Z_{\mathrm{out}}(\omega),

    where :math:`Z_{\mathrm{out}}` is the spectrum of :math:`g_{\mathrm{out}}`
    and :math:`\langle g_{\mathrm{out}}, g_{\mathrm{out}}\rangle=1`.

    Parameters
    ----------
    env:
        Input normalized Gaussian envelope.
    w0:
        Atom center frequency.
    sigma_w:
        Atom width parameter.

    Returns
    -------
    (env_out, eta):
        - env_out: normalized output Gaussian envelope shape.
        - eta: power transmissivity (squared norm of the raw filtered mode).

    Notes
    -----
    - The returned :math:`\eta` matches the definition

      .. math::

          \eta = \frac{1}{2\pi}\int |H(\omega)|^2 |Z(\omega)|^2 d\omega.

    - This function assumes the transfer is real amplitude (no phase).
      If you later support dispersive Gaussian transfers (quadratic phase),
      you will want a chirped-Gaussian envelope class.

    """
    s = float(env.sigma)
    if not (s > 0.0) or not math.isfinite(s):
        raise ValueError("sigma must be positive finite")

    sw = float(sigma_w)
    if not (sw > 0.0) or not math.isfinite(sw):
        raise ValueError("sigma_w must be positive finite")

    a = s * s
    b = 1.0 / (2.0 * sw * sw)

    sigma2_new = a + b
    sigma_new = math.sqrt(sigma2_new)
    omega_new = (a * float(env.omega0) + b * float(w0)) / sigma2_new

    # eta uses |H|^2 = exp(- (w-w0)^2 / sw^2 )
    c = 1.0 / (sw * sw)
    A = 2.0 * a
    denom = A + c
    pref = math.sqrt(A / denom)
    det = float(env.omega0) - float(w0)
    expo = -((A * c) / denom) * (det * det)
    eta = _clip_eta(pref * math.exp(expo))

    out = GaussianEnvelope(
        omega0=float(omega_new),
        sigma=float(sigma_new),
        tau=float(env.tau),
        phi0=float(env.phi0),
    )
    return out, eta


@dataclass(frozen=True)
class GaussianAtom:
    r"""One Gaussian low-pass atom term in an expansion.

    The atom is

    .. math::

        T(\omega) = c\,
        \exp\left[-\frac{1}{2}\left(\frac{\omega-w_0}{\sigma_\omega}\right)^2\right].

    Attributes
    ----------
    coeff:
        Complex coefficient :math:`c`.
    w0:
        Atom center frequency :math:`w_0`.
    sigma_w:
        Atom width :math:`\sigma_\omega`.

    """

    coeff: complex
    w0: float
    sigma_w: float


@dataclass(frozen=True)
class GaussianTransferExpansion:
    r"""Transfer function as a constant plus a sum of Gaussian low-pass atoms.

    We represent a transfer function as

    .. math::

        H(\omega) = c_0 + \sum_{k=1}^{K} c_k
        \exp\left[-\frac{1}{2}\left(\frac{\omega-\omega_k}{\sigma_k}\right)^2\right].

    Applying :math:`H` to a Gaussian envelope spectrum remains within a finite
    span of Gaussian spectra, so the output envelope can be represented as a
    :class:`~symop.modes.envelopes.gaussian_mixture.GaussianMixtureEnvelope`.

    The returned scalar :math:`\eta` is the squared norm of the *raw* output mode
    before renormalization, suitable for applying loss to the quantum state.

    Attributes
    ----------
    c0:
        Constant term :math:`c_0`.
    atoms:
        Gaussian atom terms :math:`(c_k,\omega_k,\sigma_k)`.

    Pruning controls
    ---------------
    This class supports optional pruning when building output mixtures:

    - `rel_weight2_min` and `abs_weight2_min` define pruning thresholds.
    - `max_terms` caps the mixture size.
    - `keep_at_least` enforces a minimum number of terms.

    Pruning affects only the **returned envelope representation** by default.
    The transmissivity :math:`\eta` is computed from the full unpruned raw mixture
    unless you set `eta_from_pruned=True` in the apply methods.

    """

    c0: complex
    atoms: tuple[GaussianAtom, ...]

    def apply_to_gaussian_env(
        self,
        env: GaussianEnvelope,
        *,
        rel_weight2_min: float = 1e-12,
        abs_weight2_min: float = 0.0,
        max_terms: int | None = 128,
        keep_at_least: int = 1,
        eta_from_pruned: bool = False,
    ) -> tuple[GaussianMixtureEnvelope, float]:
        r"""Apply this expansion to a single GaussianEnvelope (closed form).

        Parameters
        ----------
        env:
            Input Gaussian envelope (normalized).
        rel_weight2_min, abs_weight2_min, max_terms, keep_at_least:
            Pruning controls passed to :func:`prune_mixture_terms`.
        eta_from_pruned:
            If True, compute :math:`\eta` using the pruned mixture only.
            If False (default), compute :math:`\eta` from the full unpruned raw
            mixture, then prune only the returned envelope representation.

        Returns
        -------
        (env_out, eta):
            - env_out: normalized Gaussian mixture envelope representing the
              filtered mode.
            - eta: transmissivity (raw mode squared norm).

        Notes
        -----
        The raw (unnormalized) output is

        .. math::

            Z_{\mathrm{raw}}(\omega) = H(\omega)\,Z_{\mathrm{in}}(\omega).

        We then return a normalized envelope representing

        .. math::

            Z_{\mathrm{out}}(\omega) = \frac{1}{\sqrt{\eta}} Z_{\mathrm{raw}}(\omega),

        and the scalar

        .. math::

            \eta = \langle \zeta_{\mathrm{raw}}, \zeta_{\mathrm{raw}} \rangle.

        """
        comps: list[GaussianEnvelope] = []
        weights: list[complex] = []

        if abs(self.c0) != 0.0:
            comps.append(env)
            weights.append(complex(self.c0))

        for atom in self.atoms:
            g_out, eta_k = _gauss_atom_times_gauss_env(
                env, w0=float(atom.w0), sigma_w=float(atom.sigma_w)
            )
            comps.append(g_out)
            weights.append(complex(atom.coeff) * math.sqrt(float(eta_k)))

        if len(comps) == 0:
            out_env = GaussianMixtureEnvelope(
                components=(env,),
                weights=np.asarray([1.0 + 0.0j], dtype=complex),
            )
            return out_env, 0.0

        comps_t_full = tuple(comps)
        w_full = np.asarray(weights, dtype=complex)

        eta_full = float(GaussianMixtureEnvelope._norm2_closed(w_full, comps_t_full))
        eta_full = _clip_eta(eta_full)

        comps_t, w_pruned = prune_mixture_terms(
            comps,
            weights,
            rel_weight2_min=rel_weight2_min,
            abs_weight2_min=abs_weight2_min,
            max_terms=max_terms,
            keep_at_least=keep_at_least,
        )

        if len(comps_t) == 0:
            out_env = GaussianMixtureEnvelope(
                components=(env,),
                weights=np.asarray([1.0 + 0.0j], dtype=complex),
            )
            return out_env, 0.0

        if eta_from_pruned:
            eta_p = float(GaussianMixtureEnvelope._norm2_closed(w_pruned, comps_t))
            eta = _clip_eta(eta_p)
        else:
            eta = eta_full

        out_env = GaussianMixtureEnvelope(components=comps_t, weights=w_pruned)
        return out_env, eta

    def apply_to_gaussian(
        self,
        env: SupportsGaussianClosedOverlap,
        *,
        rel_weight2_min: float = 1e-12,
        abs_weight2_min: float = 0.0,
        max_terms: int | None = 128,
        keep_at_least: int = 1,
        eta_from_pruned: bool = False,
    ) -> tuple[SupportsGaussianClosedOverlap, float]:
        r"""Apply this expansion to a gaussian_closed envelope.

        Parameters
        ----------
        env:
            Input envelope in the gaussian_closed family (single Gaussian or mixture).
        rel_weight2_min, abs_weight2_min, max_terms, keep_at_least:
            Pruning controls.
        eta_from_pruned:
            If True, compute :math:`\eta` using the pruned mixture only.

        Returns
        -------
        (env_out, eta):
            Output gaussian_closed envelope (typically a mixture) and transmissivity.

        """
        if isinstance(env, GaussianEnvelope):
            out, eta = self.apply_to_gaussian_env(
                env,
                rel_weight2_min=rel_weight2_min,
                abs_weight2_min=abs_weight2_min,
                max_terms=max_terms,
                keep_at_least=keep_at_least,
                eta_from_pruned=eta_from_pruned,
            )
            c_out = _canonicalize_closed_env(out)
            return c_out, eta

        if isinstance(env, GaussianMixtureEnvelope):
            out, eta = _apply_expansion_to_mixture(
                env,
                self,
                rel_weight2_min=rel_weight2_min,
                abs_weight2_min=abs_weight2_min,
                max_terms=max_terms,
                keep_at_least=keep_at_least,
                eta_from_pruned=eta_from_pruned,
            )
            c_out = _canonicalize_closed_env(out)
            return c_out, eta

        raise TypeError(
            "GaussianTransferExpansion supports GaussianEnvelope or GaussianMixtureEnvelope"
        )


def _apply_expansion_to_mixture(
    env: GaussianMixtureEnvelope,
    exp: GaussianTransferExpansion,
    *,
    rel_weight2_min: float = 1e-12,
    abs_weight2_min: float = 0.0,
    max_terms: int | None = 128,
    keep_at_least: int = 1,
    eta_from_pruned: bool = False,
) -> tuple[GaussianMixtureEnvelope, float]:
    r"""Apply an expansion :math:`H(\omega)` to a Gaussian mixture input.

    If the input mode is

    .. math::

        \zeta_{\mathrm{in}}(t) = \sum_i c_i g_i(t),

    then the raw output is

    .. math::

        \zeta_{\mathrm{raw}}(t) = \sum_i c_i \,(H g_i)(t),

    and each :math:`H g_i` is represented in closed form as a (small) Gaussian
    mixture. The final output is therefore a Gaussian mixture whose number of
    components scales with the input mixture size and the number of atoms.

    Parameters
    ----------
    env:
        Input Gaussian mixture envelope (normalized).
    exp:
        Transfer expansion to apply.
    rel_weight2_min, abs_weight2_min, max_terms, keep_at_least:
        Pruning controls.
    eta_from_pruned:
        If True, compute :math:`\eta` using the pruned mixture only.

    Returns
    -------
    (env_out, eta):
        Output Gaussian mixture envelope and transmissivity :math:`\eta`.

    """
    out_comps: list[GaussianEnvelope] = []
    out_weights: list[complex] = []

    for c_i, g_i in zip(env.weights, env.components, strict=True):
        if abs(exp.c0) != 0.0:
            out_comps.append(g_i)
            out_weights.append(complex(exp.c0) * complex(c_i))

        for atom in exp.atoms:
            g_out, eta_k = _gauss_atom_times_gauss_env(
                g_i, w0=float(atom.w0), sigma_w=float(atom.sigma_w)
            )
            out_comps.append(g_out)
            out_weights.append(
                complex(atom.coeff) * complex(c_i) * math.sqrt(float(eta_k))
            )

    if len(out_comps) == 0:
        out_env = GaussianMixtureEnvelope(
            components=(env.components[0],),
            weights=np.asarray([1.0 + 0.0j], dtype=complex),
        )
        return out_env, 0.0

    comps_full = tuple(out_comps)
    w_full = np.asarray(out_weights, dtype=complex)

    eta_full = float(GaussianMixtureEnvelope._norm2_closed(w_full, comps_full))
    eta_full = _clip_eta(eta_full)

    comps_t, w_pruned = prune_mixture_terms(
        out_comps,
        out_weights,
        rel_weight2_min=rel_weight2_min,
        abs_weight2_min=abs_weight2_min,
        max_terms=max_terms,
        keep_at_least=keep_at_least,
    )

    if len(comps_t) == 0:
        out_env = GaussianMixtureEnvelope(
            components=(env.components[0],),
            weights=np.asarray([1.0 + 0.0j], dtype=complex),
        )
        return out_env, 0.0

    if eta_from_pruned:
        eta_p = float(GaussianMixtureEnvelope._norm2_closed(w_pruned, comps_t))
        eta = _clip_eta(eta_p)
    else:
        eta = eta_full

    out_env = GaussianMixtureEnvelope(components=comps_t, weights=w_pruned)
    return out_env, eta


def _canonicalize_closed_env(
    env: SupportsGaussianClosedOverlap,
) -> SupportsGaussianClosedOverlap:
    r"""Canonicalize a gaussian_closed envelope representation.

    This helper simplifies the returned representation after closed-form
    transfer application.

    If the input is a :class:`GaussianMixtureEnvelope` with exactly one
    component, the mixture is collapsed back to a single
    :class:`GaussianEnvelope` by absorbing the unit-modulus mixture weight
    into the component global phase :math:`\phi_0`.

    Parameters
    ----------
    env:
        Envelope in the ``gaussian_closed`` family.

    Returns
    -------
    SupportsGaussianClosedOverlap
        Canonicalized envelope representation.

        - If ``env`` is not a single-term mixture, it is returned unchanged.
        - If ``env`` is a one-term mixture, a single
          :class:`GaussianEnvelope` is returned.

    Notes
    -----
    Let a normalized one-term mixture be

    .. math::

        \zeta(t) = w\,g(t),

    where :math:`|w| = 1` after normalization. Since this factor is a pure
    global phase, it can be absorbed into the Gaussian phase parameter:

    .. math::

        w = e^{i\Delta\phi},
        \qquad
        g(t) \mapsto g'(t) \text{ with } \phi_0' = \phi_0 + \Delta\phi.

    This preserves the represented mode while yielding a simpler canonical
    form.

    """
    # Collapse a 1-term GaussianMixtureEnvelope into a GaussianEnvelope
    # by absorbing the (unit-modulus) mixture weight into phi0.
    if not isinstance(env, GaussianMixtureEnvelope):
        return env

    if len(env.components) != 1:
        return env

    g = env.components[0]
    w = complex(env.weights[0])  # normalized by __post_init__
    # guard: if weight is (almost) zero, phase is undefined; just drop it
    if abs(w) == 0.0:
        return g

    dphi = float(np.angle(w))
    # absorb global phase into phi0
    return GaussianEnvelope(
        omega0=float(g.omega0),
        sigma=float(g.sigma),
        tau=float(g.tau),
        phi0=float(g.phi0 + dphi),
    )
