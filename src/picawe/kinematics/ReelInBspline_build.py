import numpy as np
import casadi as ca

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
    # ri - reel-ina
    # ro - reel-out
    # sph - spherical
    # cart - cartesian


    # -------------------------------
    # Return CasADi function N_func(u)
    # -------------------------------
    def Nvec_symbolic(self):
        u_sym = ca.MX.sym("u")
        n_ctrl = self.n_ctrl

        def N(i, k, u):
            if k == 0:
                return ca.if_else(ca.logic_and(self.U[i] <= u, u <= self.U[i+1]), 1.0, 0.0)
            left = ca.if_else(self.U[i+k] > self.U[i],
                              (u - self.U[i]) / (self.U[i+k]-self.U[i]) * N(i, k-1, u),
                              0)
            right = ca.if_else(self.U[i+k+1] > self.U[i+1],
                               (self.U[i+k+1]-u)/(self.U[i+k+1]-self.U[i+1]) * N(i+1, k-1, u),
                               0)
            return left + right

        Nvec_sym = ca.vertcat(*[N(i, self.p, u_sym) for i in range(n_ctrl)]).T
        N_func = ca.Function("N_func", [u_sym], [Nvec_sym], ["u"], ["Nvec"])
        return N_func

    # -------------------------------
    # Build symbolic spline function
    # -------------------------------
    def build_bspline_symbolic(self, return_derivative=True):
        C_sym = ca.MX.sym("C", self.n_ctrl, self.dim)
        u_sym = ca.MX.sym("u")

        N_func = self.Nvec_symbolic()
        S_sym = ca.mtimes(N_func(u_sym), C_sym)
        dS_sym = ca.jacobian(S_sym, u_sym) if return_derivative else None

        spline_func = ca.Function("spline_func", [C_sym, u_sym], [S_sym, dS_sym], ["C","u"], ["S","dS"])
        return spline_func

    # -------------------------------
    # Evaluate spline numerically
    # -------------------------------
    def eval_spline(self, spline_func=None, C_val=None):
        if C_val is None:
            C_val = self.C
        
        if spline_func is None:
            spline_func = self.build_bspline_symbolic()

        S_list, dS_list = [], []
        for ui in self.u_vals:
            res = spline_func(C=C_val, u=ui)
            S_list.append(np.array(res["S"]).flatten())
            dS_list.append(np.array(res["dS"]).flatten())

        S_eval = np.vstack(S_list)
        dS_eval = np.vstack(dS_list) if dS_list else None
        return S_eval, dS_eval

    # -------------------------------
    # Build full N matrix numerically
    # -------------------------------
    def build_Nmat(self):
        N_func = self.Nvec_symbolic()
        Nmat = np.vstack([np.array(N_func(ui)).flatten() for ui in self.u_vals])
        return Nmat

    # def build_bspline(self, C, p, U, u, return_derivative=True):
    #     """
    #     Build a B-spline symbolic expression.

    #     Parameters
    #     ----------
    #     C : MX/SX symbolic or DM numeric
    #         Control points (n_ctrl x dim)
    #     p : int
    #         Degree of the spline
    #     U : array-like or MX symbolic
    #         Knot vector (length n_ctrl + p + 1)
    #     u : MX/SX symbolic
    #         Parameter value
    #     return_derivative : bool
    #         Whether to compute dS/du

    #     Returns
    #     -------
    #     S : MX/SX
    #         Spline position at u
    #     dS : MX/SX or None
    #         Spline derivative
    #     Nvec : MX/SX
    #         Basis vector
    #     """
    #     n_ctrl = C.shape[0]

    #     def N(i, k, u_val):
    #         if k == 0:
    #             return ca.if_else(
    #                 ca.logic_and(U[i] <= u_val, u_val <= U[i+1]),
    #                 1.0,
    #                 0.0
    #             )
    #         # Recursion
    #         left = ca.if_else(U[i+k] > U[i],
    #                           (u_val - U[i]) / (U[i+k]-U[i]) * N(i, k-1, u_val),
    #                           0)
    #         right = ca.if_else(U[i+k+1] > U[i+1],
    #                            (U[i+k+1]-u_val)/(U[i+k+1]-U[i+1]) * N(i+1, k-1, u_val),
    #                            0)
    #         return left + right

    #     Nvec = ca.vertcat(*[N(i, p, u) for i in range(n_ctrl)]).T
    #     S = ca.mtimes(Nvec, C)

    #     dS = ca.jacobian(S, u) if return_derivative else None
    #     return S, dS, Nvec

    # # -------------------------------
    # # Vectorized basis matrix
    # # -------------------------------
    # def build_Nmat(self, U, p, u_vals):
    #     """
    #     Build the B-spline basis matrix Nmat for all u_vals (vectorized).

    #     Parameters
    #     ----------
    #     U : array-like (numeric)
    #         Knot vector
    #     p : int
    #         Degree
    #     u_vals : array-like (numeric)
    #         Parameter values

    #     Returns
    #     -------
    #     Nmat : np.ndarray
    #         Shape (len(u_vals), n_ctrl)
    #     """
    #     n_ctrl = len(U) - p - 1

    #     # Symbolic variables
    #     u_sym = ca.MX.sym("u")
    #     C_dummy = ca.MX.sym("C", n_ctrl, 1)  # dummy control points
    #     _, _, Nvec_sym = self.build_bspline(C_dummy, p, U, u_sym, return_derivative=False)

    #     # CasADi function to evaluate Nvec
    #     N_func = ca.Function("N_func", [u_sym], [Nvec_sym])

    #     # Vectorized evaluation
    #     Nmat = np.array([N_func(ui).full().flatten() for ui in u_vals])
    #     return Nmat  # shape (len(u_vals), n_ctrl)

    # # -------------------------------
    # # Vectorized spline evaluation
    # # -------------------------------
    # def eval_spline(self, spline_func, C_val, u_vals):
    #     """
    #     Evaluate a CasADi spline function for multiple u values.

    #     Parameters
    #     ----------
    #     spline_func : casadi.Function
    #         Function [C, u] -> [S, dS]
    #     C_val : np.ndarray
    #         Control points
    #     u_vals : array-like
    #         Parameter values

    #     Returns
    #     -------
    #     S_eval : np.ndarray
    #         Evaluated spline positions
    #     dS_eval : np.ndarray
    #         Evaluated derivatives
    #     """
    #     u_vals = np.atleast_1d(u_vals)
    #     S_list, dS_list = [], []

    #     for ui in u_vals:
    #         res = spline_func(C=C_val, u=ui)
    #         S_list.append(res["S"].full().flatten())
    #         dS_list.append(res["dS"].full().flatten())

    #     S_eval = np.vstack(S_list)
    #     dS_eval = np.vstack(dS_list)
    #     return S_eval, dS_eval