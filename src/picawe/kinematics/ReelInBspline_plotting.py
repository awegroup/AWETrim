import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scripts.src.picawe.kinematics import ReelInBspline

# -------------------------------
# Plotting
# -------------------------------

class ReelInBspline_plotting(ReelInBspline):
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
        
        # Plot RI start/end
        if self.ri_start_point is not None and self.ri_end_point is not None:
            ax.scatter(*self.ri_start_point, color="green", s=30, label="RI Start")
            ax.scatter(*self.ri_end_point, color="red", s=30, label="RI End")
        
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(); ax.set_box_aspect([1,1,1])
        plt.show()

    def plot_spline_fit_sph(self):
        if self.C_sph is None or self.U_sph is None or self.u_vals is None:
            raise ValueError("Run fit_spherical_spline() before plotting spherical fit.")
        if self.az_RI is None or self.el_RI is None:
            raise ValueError("Run get_RI_RO_boundaries() before plotting spherical fit.")

        # Evaluate spherical spline
        S_fit_sph = np.vstack([
            self.evaluate_bspline(self.C_sph, self.p, self.U_sph, u) for u in self.u_vals
        ])

        # Create subplots (single fig only!)
        fig, (ax_az, ax_el) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

        # Azimuth plot
        ax_az.plot(self.u_vals, self.az_RI, label="Azimuth (data)", color="C0")
        ax_az.plot(self.u_vals, S_fit_sph[:, 0], "--", label="Azimuth (spline)", color="C1")
        
        # Add control points
        if self.C_sph is not None:
            ax_az.scatter(
                np.linspace(self.u_vals[0], self.u_vals[-1], len(self.C_sph)),
                self.C_sph[:, 0], color="black", s=30, label="Control points"
            )

        # Add RI endpoints
        if self.ri_start_point is not None and self.ri_end_point is not None:
            ax_az.scatter(self.u_vals[0], self.ri_start_point_sph[0], color="green", s=30, label="RI Start")
            ax_az.scatter(self.u_vals[-1], self.ri_end_point_sph[0], color="red", s=30, label="RI End")

        ax_az.set_xlabel("u")
        ax_az.set_ylabel("Azimuth [rad]")
        ax_az.grid(True, alpha=0.3)
        ax_az.legend()

        # Elevation plot
        ax_el.plot(self.u_vals, self.el_RI, label="Elevation (data)", color="C0")
        ax_el.plot(self.u_vals, S_fit_sph[:, 1], "--", label="Elevation (spline)", color="C1")
        
        # Add control points
        if self.C_sph is not None:
            ax_el.scatter(
                np.linspace(self.u_vals[0], self.u_vals[-1], len(self.C_sph)),
                self.C_sph[:, 1], color="black", s=30, label="Control points"
            )

        # Add RI endpoints
        if self.ri_start_point is not None and self.ri_end_point is not None:
            ax_el.scatter(self.u_vals[0], self.ri_start_point_sph[1], color="green", s=30, label="RI Start")
            ax_el.scatter(self.u_vals[-1], self.ri_end_point_sph[1], color="red", s=30, label="RI End")

        ax_el.set_xlabel("u")
        ax_el.set_ylabel("Elevation [rad]")
        ax_el.grid(True, alpha=0.3)
        ax_el.legend()

        fig.tight_layout()
        plt.show()