"""Mach–Zehnder interferometer utilities.

Constructs the 2x2 unitary matrix of a Mach–Zehnder interferometer (MZI)
from beamsplitters and phase shifters under the package Heisenberg
operator convention.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from .beamsplitter import beamsplitter_u
from .conventions import require_unitary_optional
from .phase import phase_u


def mzi_u(
    *,
    theta1: float,
    theta2: float,
    phi_internal: float,
    phi_in0: float = 0.0,
    phi_in1: float = 0.0,
    phi_out0: float = 0.0,
    phi_out1: float = 0.0,
    check_unitary: bool = False,
    atol: float = 1e-10,
) -> NDArray[np.complex128]:
    r"""Return a 2x2 Mach-Zehnder interferometer (MZI) unitary.

    This composes two couplers with input/output phase shifters and one internal
    differential phase between the arms.

    Construction
    ------------
    Define two 2x2 couplers U1, U2 (directional-coupler family via beamsplitter_u),
    and diagonal phase matrices:

    .. math::

        P_\mathrm{in}  = \mathrm{diag}(e^{i\phi_{in,0}}, e^{i\phi_{in,1}}), \\
        P_\mathrm{int} = \mathrm{diag}(e^{i\phi_{internal}}, 1), \\
        P_\mathrm{out} = \mathrm{diag}(e^{i\phi_{out,0}}, e^{i\phi_{out,1}}).

    Then:

    .. math::

        U = P_\mathrm{out} \, U_2 \, P_\mathrm{int} \, U_1 \, P_\mathrm{in}.

    Parameters
    ----------
    theta1, theta2:
        Coupler angles that define splitting ratios.
        We use t=cos(theta), r=sin(theta) with the beamsplitter_u convention.
    phi_internal:
        Internal phase on arm 0 relative to arm 1.
    phi_in0, phi_in1:
        Input port phases.
    phi_out0, phi_out1:
        Output port phases.
    check_unitary:
        If True, validate unitarity.
    atol:
        Tolerance for optional unitary check.

    Returns
    -------
    numpy.ndarray
        Complex matrix of shape (2, 2).

    Notes
    -----
    This is a “photonic circuit” convenience constructor. If you prefer a
    different internal phase placement (arm 1 instead of arm 0), swap the
    diagonal of P_int.

    """
    # Couplers (directional-coupler style, no extra phases).
    t1 = float(np.cos(theta1))
    r1 = float(np.sin(theta1))
    t2 = float(np.cos(theta2))
    r2 = float(np.sin(theta2))

    U1 = beamsplitter_u(t=t1, r=r1, phi_t=0.0, phi_r=0.0)
    U2 = beamsplitter_u(t=t2, r=r2, phi_t=0.0, phi_r=0.0)

    Pin = np.diag([phase_u(phi=phi_in0)[0, 0], phase_u(phi=phi_in1)[0, 0]]).astype(
        np.complex128
    )
    Pint = np.diag([phase_u(phi=phi_internal)[0, 0], 1.0 + 0.0j]).astype(np.complex128)
    Pout = np.diag([phase_u(phi=phi_out0)[0, 0], phase_u(phi=phi_out1)[0, 0]]).astype(
        np.complex128
    )

    U = Pout @ U2 @ Pint @ U1 @ Pin
    require_unitary_optional(U, check_unitary=check_unitary, atol=atol)
    return cast(NDArray[np.complex128], U)
