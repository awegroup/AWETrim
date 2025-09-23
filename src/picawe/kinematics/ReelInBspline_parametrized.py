import numpy as np
import casadi as ca
import picawe.kinematics.ReelInBspline_fitting as ribfit

# -------------------------------
# Spline Building
# -------------------------------
class ReelInBspline():

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

    def __init__(self, 
                 p=3, 
                 n_ctrl=8, 
                 r0=300, 
                 rf=None, 
                 crs0=(11/6)*np.pi, 
                 crsf=np.pi/2, 
                 phi0=0, 
                 phif=0, 
                 beta0=0, 
                 betaf=0, 
                 C_interior=None, 
                 u_vals=None, 
                 U=None,
                 mode="spherical"):
        
        # dimension
        self.dim = 2 if mode == "spherical" else 3

        # spline order and control points
        self.p = p
        self.n_ctrl = n_ctrl

        # start/end scalars (numeric, not symbolic)
        self.r0 = r0
        self.rf = rf if rf is not None else 0.0
        self.crs0 = crs0
        self.crsf = crsf
        self.phi0 = phi0
        self.phif = phif
        self.beta0 = beta0
        self.betaf = betaf

        # knots
        self.n_knots = self.n_ctrl + self.p + 1
        self.n_interior_knots = self.n_knots - 2*(self.p+1)
        if self.n_interior_knots < 0:
            raise ValueError("Too few control points for spline order")
        
        if U is None:
            self.U_interior = np.linspace(0.15, 0.85, self.n_interior_knots+2)[1:-1]
            self.U = np.concatenate(([0]*(self.p+1), self.U_interior, [1]*(self.p+1)))
        else:
            self.U = U

        # interior control points
        if C_interior is None:
            C_interior = np.ones((self.n_ctrl-2, self.dim))
        self.C_interior = C_interior

        # full control points matrix (numeric)
        self.C = np.vstack([
            np.array([self.phi0, self.beta0]),
            self.C_interior,
            np.array([self.phif, self.betaf])
        ])

        # u values for evaluation
        self.u_vals = np.linspace(0, 1, 100) if u_vals is None else u_vals

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


    

if __name__ == "__main__":

    fitted = ribfit.ReelInBspline_fitting(
    file_path_full = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv",
    file_path_cycle = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv",
    cyc_idx=0,
    p=3,
    n_ctrl=8,
    c_penalty=1.0,
    v_penalty=0.0,
    eps_knot=1e-3
    )
    
    mode = "spherical"  # "spherical" or "cartesian"

    Spline = ReelInBspline(
        p=fitted.p,
        n_ctrl=fitted.n_ctrl,
        crs0=fitted.ri_crs0,
        crsf=fitted.ri_crsf,
        phi0=fitted.ri_p0_sph[0],
        phif=fitted.ri_pf_sph[0],
        beta0=fitted.ri_p0_sph[1],
        betaf=fitted.ri_pf_sph[1],
        C_interior=fitted.C_sph[1:-1] if mode=="spherical" else fitted.C_cart[1:-1],
        u_vals=fitted.u_vals,
        U=fitted.U_sph if mode=="spherical" else fitted.U_cart
    )

    S_fit, dS_fit = Spline.eval_spline()

