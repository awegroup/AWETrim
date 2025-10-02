import numpy as np
import casadi as ca

# -------------------------------
# Spline Building
# -------------------------------
class Bspline_build():

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
    # Return CasADi function N_func(u, U) that evaluates the B-spline basis vector
    # -------------------------------
    def Nvec_symbolic(self):
        """
        Returns a CasADi function N_func(u, U) that evaluates
        the B-spline basis vector for given parameter u and knot vector U.
        """
        u_sym = ca.MX.sym("u")
        U_sym = ca.MX.sym("U", self.n_ctrl + self.p + 1)  # full knot vector
        n_ctrl = self.n_ctrl
        p = self.p

        def N(i, k, u):
            if k == 0:
                # use CasADi logical operations for symbolic MX
                return ca.if_else(ca.logic_and(U_sym[i] <= u, u <= U_sym[i+1]), 1.0, 0.0)
            left = ca.if_else(U_sym[i+k] > U_sym[i],
                            (u - U_sym[i]) / (U_sym[i+k]-U_sym[i]) * N(i, k-1, u),
                            0)
            right = ca.if_else(U_sym[i+k+1] > U_sym[i+1],
                            (U_sym[i+k+1]-u)/(U_sym[i+k+1]-U_sym[i+1]) * N(i+1, k-1, u),
                            0)
            return left + right

        Nvec_sym = ca.vertcat(*[N(i, p, u_sym) for i in range(n_ctrl)]).T
        N_func = ca.Function("N_func", [u_sym, U_sym], [Nvec_sym], ["u", "U"], ["Nvec"])
        return N_func

    # -------------------------------
    # Build symbolic spline function S(u) = N(u, U)*C
    # -------------------------------
    def build_bspline_symbolic(self, return_derivative=True):
        C_sym = ca.MX.sym("C", self.n_ctrl, self.dim)
        u_sym = ca.MX.sym("u") 
        U_sym = ca.MX.sym("U", self.n_ctrl + self.p + 1)  # always symbolic

        # Use the new Nvec_symbolic that depends on u and U
        N_func = self.Nvec_symbolic()  # N_func(u, U)
        S_sym = ca.mtimes(N_func(u_sym, U_sym), C_sym)
        dS_sym = ca.jacobian(S_sym, u_sym) if return_derivative else None

        spline_func = ca.Function("spline_func",
                                [C_sym, u_sym, U_sym],
                                [S_sym, dS_sym],
                                ["C","u","U"],
                                ["S","dS"])
        return spline_func

    # -------------------------------
    # Evaluate spline numerically for multiple u values
    # -------------------------------
    def eval_spline(self, spline_func=None, C_val=None, U_val=None):
        """
        Evaluate the CasADi spline function for multiple u values.

        Parameters
        ----------
        spline_func : casadi.Function, optional
            Function [C, u, U] -> [S, dS]
        C_val : np.ndarray, optional
            Control points (n_ctrl x dim)
        U_val : np.ndarray, optional
            Knot vector (length n_ctrl + p + 1)

        Returns
        -------
        S_eval : np.ndarray
            Evaluated spline positions
        dS_eval : np.ndarray
            Evaluated derivatives
        """
        if C_val is None:
            C_val = self.C
        if U_val is None:
            U_val = self.U
        if spline_func is None:
            spline_func = self.build_bspline_symbolic()

        S_list, dS_list = [], []
        for ui in self.u_vals:
            res = spline_func(C=C_val, u=ui, U=U_val)  # pass U_val now
            S_list.append(np.array(res["S"]).flatten())
            dS_list.append(np.array(res["dS"]).flatten())

        S_eval = np.vstack(S_list)
        dS_eval = np.vstack(dS_list) if dS_list else None
        return S_eval, dS_eval

    # -------------------------------
    # Build full N matrix numerically
    # -------------------------------
    # def build_Nmat(self):
    #     N_func = self.Nvec_symbolic()
    #     Nmat = np.vstack([np.array(N_func(ui)).flatten() for ui in self.u_vals])
    #     return Nmat