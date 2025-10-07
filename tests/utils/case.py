from __future__ import annotations
import unittest
from typing import Any, Optional

import numpy as np
import numpy.testing as npt


class ExtendedTestCase(unittest.TestCase):
    """
    unittest.TestCase with complex and NumPy helpers.
    Works under unittest or pytest.
    """

    default_places = 12
    default_rtol = 1e-12
    default_atol = 1e-12

    # ---------- scalars ----------

    def assertComplexAlmostEqual(
        self,
        a: complex,
        b: complex,
        *,
        places: Optional[int] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        msg: Optional[str] = None,
    ) -> None:
        """
        Compare complex numbers. Use either `places` (decimal places) or (rtol, atol).
        """
        if rtol is not None or atol is not None:
            r = self.default_rtol if rtol is None else rtol
            t = self.default_atol if atol is None else atol
            diff = abs(a - b)
            bound = max(t, r * max(abs(a), abs(b)))
            if not diff <= bound:
                self.fail(
                    self._formatMessage(
                        msg,
                        f"|a-b|={diff:.3e} > max(atol={t:.3e}, rtol={
                            r:.3e}*max(|a|,|b|)) "
                        f"with a={a}, b={b}",
                    )
                )
            return

        p = self.default_places if places is None else places
        self.assertAlmostEqual(a.real, b.real, places=p, msg=msg)
        self.assertAlmostEqual(a.imag, b.imag, places=p, msg=msg)

    # ---------- arrays ----------

    def assertArrayAllClose(
        self,
        actual: Any,
        desired: Any,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        equal_nan: bool = False,
        err_msg: str | None = None,
    ) -> None:
        """
        Vectorized closeness check (complex supported).
        """
        r = self.default_rtol if rtol is None else rtol
        t = self.default_atol if atol is None else atol
        try:
            npt.assert_allclose(
                actual,
                desired,
                rtol=r,
                atol=t,
                equal_nan=equal_nan,
                err_msg=err_msg,
            )
        except AssertionError as e:
            self.fail(str(e))

    def assertArrayEqual(
        self,
        actual: Any,
        desired: Any,
        *,
        equal_nan: bool = False,
        err_msg: str | None = None,
    ) -> None:
        """
        Exact equality (shape + values). For floats/complex, prefer assertArrayAllClose.
        """
        a = np.asarray(actual)
        b = np.asarray(desired)
        if a.shape != b.shape:
            self.fail(
                self._formatMessage(
                    err_msg, f"shape mismatch: {a.shape} != {b.shape}"
                )
            )
        if equal_nan:
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            if not np.all(same):
                idx = np.argwhere(~same)
                self.fail(
                    self._formatMessage(
                        err_msg,
                        f"arrays differ at indices (first 5): {
                            idx[:5].tolist()}",
                    )
                )
        else:
            try:
                npt.assert_array_equal(a, b, err_msg=err_msg)
            except AssertionError as e:
                self.fail(str(e))

    # ---------- linear algebra niceties ----------

    def assertHermitian(
        self,
        A: np.ndarray,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        err_msg: str | None = None,
    ) -> None:
        A = np.asarray(A)
        self.assertTrue(
            A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square"
        )
        self.assertArrayAllClose(
            A, A.conj().T, rtol=rtol, atol=atol, err_msg=err_msg
        )

    def assertUnitary(
        self,
        U: np.ndarray,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        err_msg: str | None = None,
    ) -> None:
        U = np.asarray(U)
        self.assertTrue(
            U.ndim == 2 and U.shape[0] == U.shape[1], "U must be square"
        )
        n = U.shape[0]
        I = np.eye(n, dtype=U.dtype)
        self.assertArrayAllClose(
            U.conj().T @ U, I, rtol=rtol, atol=atol, err_msg=err_msg
        )
        self.assertArrayAllClose(
            U @ U.conj().T, I, rtol=rtol, atol=atol, err_msg=err_msg
        )
