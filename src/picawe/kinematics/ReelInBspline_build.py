import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# -------------------------------
# Spline Building
# -------------------------------
class ReelInBspline_build():

    # something0 or somethingf means the start or end 0 for start and f for final
    # p - point eg. p0 start point
    # v - velocity
    # crs - course
    # idx - index
    # cyc - cycle
    # ri - reel-in
    # ro - reel-out
    # sph - spherical
    # cart - cartesian

    def evaluate_bspline(self, C, p, U, u, return_basis=False, return_derivative=False):
        """
        Evaluate B-spline at u (scalar or array).
        If return_basis=True, returns basis matrix Nmat.
        If return_derivative=True, also returns derivative basis matrix dNmat.
        """
        u = np.atleast_1d(u)
        n_ctrl = C.shape[0]
        Nmat = np.zeros((len(u), n_ctrl))
        dNmat = np.zeros((len(u), n_ctrl))

        def N(i, k, u_val):
            if k == 0:
                # Special case: include right endpoint for last knot span
                if (U[i] <= u_val < U[i+1]) or (
                    u_val == U[-1] and U[i] <= u_val <= U[i+1]
                ):
                    return 1.0
                else:
                    return 0.0

            left = 0.0
            right = 0.0
            left = 0.0
            right = 0.0

            if U[i+k] > U[i]:
                left = (u_val - U[i])/(U[i+k]-U[i]) * N(i, k-1, u_val)
            if U[i+k+1] > U[i+1]:
                right = (U[i+k+1]-u_val)/(U[i+k+1]-U[i+1]) * N(i+1, k-1, u_val)
            return left + right

        def dN(i, k, u_val):
            if k == 0:
                return 0.0
            left = 0.0
            right = 0.0
            if U[i+k] > U[i]:
                left = k/(U[i+k]-U[i]) * N(i, k-1, u_val)
            if U[i+k+1] > U[i+1]:
                right = k/(U[i+k+1]-U[i+1]) * N(i+1, k-1, u_val)
            return left - right

        for ui, u_val in enumerate(u):
            for i in range(n_ctrl):
                Nmat[ui, i] = N(i, p, u_val)
                if return_derivative:
                    dNmat[ui, i] = dN(i, p, u_val)

        S = Nmat @ C
        dS = dNmat @ C if return_derivative else None

        return S, Nmat, dNmat, dS

    # -------------------------------
    # Spline evaluation
    # -------------------------------
    def eval_cartesian_spline(self, u):
        Su, _, _, _ = self.evaluate_bspline(self.C_cart, self.p, self.U_cart, u)
        return Su

    def eval_spherical_spline(self, u):
        Su, _, _, _ = self.evaluate_bspline(self.C_sph, self.p, self.U_sph, u)
        return Su

if __name__ == "__main__":
    pass