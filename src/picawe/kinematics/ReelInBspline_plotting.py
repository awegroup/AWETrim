import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .ReelInBspline_fitting import ReelInBspline_fitting as ribfit
from .ReelInBspline_build import ReelInBspline_build as ribbuild

# -------------------------------
# Plotting
# -------------------------------
class ReelInBspline_plotting(ribfit, ribbuild):

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

    def plot_spline_fit_cart(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.x_cyc, self.y_cyc, self.z_cyc, label="Trajectory", alpha=0.6)
        S_fit_cart = np.vstack([self.eval_cartesian_spline(u) for u in self.u_vals])
        ax.plot(S_fit_cart[:,0], S_fit_cart[:,1], S_fit_cart[:,2], "r--", label="B-spline fit")
        
        # Plot control points
        if self.C_cart is not None:
            ax.scatter(self.C_cart[:,0], self.C_cart[:,1], self.C_cart[:,2],
                    color="black", s=30, label="Control points")
        
        # Plot ri start/end
        if self.ri_p0_cart is not None and self.ri_pf_cart is not None:
            ax.scatter(*self.ri_p0_cart, color="green", s=30, label="ri start")
            ax.scatter(*self.ri_pf_cart, color="red", s=30, label="ri end")
        
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(); ax.set_box_aspect([1,1,1])
        plt.show()

    def plot_spline_fit_sph(self):
        if self.C_sph is None or self.U_sph is None or self.u_vals is None:
            raise ValueError("Run fit_spherical_spline() before plotting spherical fit.")
        if self.az_ri is None or self.el_ri is None:
            raise ValueError("Run get_ri_ro_boundaries() before plotting spherical fit.")

        # Evaluate spherical spline
        S_fit_sph = np.vstack([
            self.evaluate_bspline(self.C_sph, self.p, self.U_sph, u) for u in self.u_vals
        ])

        # Create subplots (single fig only!)
        fig, (ax_az, ax_el) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

        # Azimuth plot
        ax_az.plot(self.u_vals, self.az_ri, label="Azimuth (data)", color="C0")
        ax_az.plot(self.u_vals, S_fit_sph[:, 0], "--", label="Azimuth (spline)", color="C1")
        
        # Elevation plot
        ax_el.plot(self.u_vals, self.el_ri, label="Elevation (data)", color="C0")
        ax_el.plot(self.u_vals, S_fit_sph[:, 1], "--", label="Elevation (spline)", color="C1")

        ax_az.scatter(
            np.linspace(self.u_vals[0], self.u_vals[-1], len(self.C_sph)),
            self.C_sph[:, 0], color="black", s=30, label="Control points"
            )
        
        ax_el.scatter(
            np.linspace(self.u_vals[0], self.u_vals[-1], len(self.C_sph)),
            self.C_sph[:, 1], color="black", s=30, label="Control points"
            )

        # Add ri endpoints
        if self.ri_p0_sph is not None and self.ri_pf_sph is not None:
            ax_az.scatter(self.u_vals[0], self.ri_p0_sph[0], color="green", s=30, label="ri start")
            ax_az.scatter(self.u_vals[-1], self.ri_pf_sph[0], color="red", s=30, label="ri end")

            ax_el.scatter(self.u_vals[0], self.ri_p0_sph[1], color="green", s=30, label="ri start")
            ax_el.scatter(self.u_vals[-1], self.ri_pf_sph[1], color="red", s=30, label="ri end")

        ax_az.set_xlabel("u")
        ax_az.set_ylabel("Azimuth [rad]")
        ax_az.grid(True, alpha=0.3)
        ax_az.legend()
        
        ax_el.set_xlabel("u")
        ax_el.set_ylabel("Elevation [rad]")
        ax_el.grid(True, alpha=0.3)
        ax_el.legend()

        fig.tight_layout()
        plt.show()