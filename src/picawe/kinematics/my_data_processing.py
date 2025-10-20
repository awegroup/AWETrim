import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataProcessing:
    """
    Load and process kite cycle data:
      - read CSVs and preprocess time
      - convert spherical -> cartesian
      - extract one cycle (cyc_)
      - detect Reel-In (RI_) and Reel-Out (RO_) segments
      - detect transitions RI->RO (RI_RO_) and RO->RI (RO_RI_)
      - plotting helpers (3D & 2D Lissajous)
    Naming convention: cyc_*, RI_*, RO_*, RI_RO_*, RO_RI_*.
    """

    def __init__(self, file_path_full, file_path_cycle, file_path_waypoints, cyc_idx=0):
        # --- Load CSVs ---
        self.wp_df = pd.read_csv(file_path_waypoints)
        self.full_df = pd.read_csv(file_path_full)
        self.cycle_df = pd.read_csv(file_path_cycle)
        self.cyc_idx = cyc_idx

        # --- Time preprocessing (rounded to 0.1s like original) ---
        self.wp_df["time_s"] = np.round(self._to_seconds(self.wp_df["time_string"]), 1)
        self.full_df["time_s"] = np.round(self._to_seconds(self.full_df["time_of_day"]), 1)
        self.cycle_df["start_time_s"] = np.round(self._to_seconds(self.cycle_df["start_time_cycle_LT"]), 1)

        # --- Full arrays ---
        self.time_waypoints = self.wp_df["time_s"].to_numpy()
        self.time_full = self.full_df["time_s"].to_numpy()
        self.time_cycles = self.cycle_df["start_time_s"].to_numpy()


        # primary spherical signals
        self.az_full = self.full_df["kite_azimuth"].astype(float).to_numpy()
        self.el_full = self.full_df["kite_elevation"].astype(float).to_numpy()
        self.r_full = self.full_df["kite_distance"].astype(float).to_numpy()
        self.phase_full = self.full_df["flight_phase"].astype(str).to_numpy()
        self.crs_full = self.full_df["kite_course"].astype(float).to_numpy()
        self.depower_full = self.full_df["kite_actual_depower"].astype(float).to_numpy()
        self.wp_names = self.wp_df["waypoint_name"].astype(str).to_numpy()

        # Cartesian & derivatives for full dataset
        self.x_full, self.y_full, self.z_full = self._sph2cart(self.az_full, self.el_full, self.r_full)
        self.dx_full, self.dy_full, self.dz_full = (
            np.gradient(self.x_full),
            np.gradient(self.y_full),
            np.gradient(self.z_full),
        )

        # --- Cycle selection (cyc_) ---
        start_indices = [i for i, t in enumerate(self.time_full) if t in self.time_cycles]
        if self.cyc_idx >= len(start_indices) - 1:
            raise IndexError("cyc_idx out of range")
        self.cyc_idx0 = start_indices[self.cyc_idx]
        # if next exists, end at next-1, else end at end-of-file (original logic used -1 fallback)
        self.cyc_idxf = (
            start_indices[self.cyc_idx + 1] - 1
            if self.cyc_idx + 1 < len(start_indices)
            else len(self.time_full) - 1
        )

        self._extract_cycle_slice()

        # --- Detect RI and define RO (reel-out is before RI start) ---
        self._detect_RI_segment()
        self._extract_RO_segment()

        # --- Find Lissajous bounds on RO data ---
        self._find_lissajous_bounds()

        # --- Find transitions and compute normalized distances (u) ---
        self._find_RI_RO_transition()
        self._find_RO_RI_transition()

        self.Lissajous_r0 = self.r_cyc[self.Lissajous_idx0]
        self.Lissajous_r1 = self.r_cyc[self.Lissajous_idxf]
        self.Lissajous_Duration = self.time_cyc[self.Lissajous_idxf] - self.time_cyc[self.Lissajous_idx0]

    # -------------------------
    # Utilities
    # -------------------------
    def _to_seconds(self, series):
        """Convert HH:MM:SS.sss strings to seconds."""
        return np.array(
            [float(h) * 3600 + float(m) * 60 + float(s) for h, m, s in (str(t).split(":") for t in series)]
        )

    def _sph2cart(self, az, el, r):
        """Spherical -> Cartesian conversion (vectorized)."""
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    def _compute_u(self, x, y, z):
        """Return normalized cumulative distance along (x,y,z)."""        
        cart = np.vstack([x, y, z]).T
        dist = np.cumsum(np.linalg.norm(np.diff(cart, axis=0), axis=1))
        dist = np.insert(dist, 0, 0.0)
        return dist / dist[-1]

    def _plot3d_generic(self, x, y, z, seg=None, seg_label=None, seg_color="orange", start=None, end=None, title=None):
        """Small helper to plot a main path and an optional highlighted segment."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x, y, z, label="Cycle Path")
        if seg is not None:
            ax.plot(*seg, color=seg_color, linestyle="--", label=seg_label)
        if start is not None:
            ax.scatter(*start, color="g", label="Start")
        if end is not None:
            ax.scatter(*end, color="r", label="End")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        if title:
            ax.set_title(title)
        ax.legend()
        plt.show()

    # -------------------------
    # Cycle slice extraction
    # -------------------------
    def _extract_cycle_slice(self):
        """Populate cyc_ prefixed arrays for the selected cycle."""
        s, f = self.cyc_idx0, self.cyc_idxf + 1
        self.time_cyc = self.time_full[s:f]
        self.az_cyc = self.az_full[s:f]
        self.el_cyc = self.el_full[s:f]
        self.r_cyc = self.r_full[s:f]
        self.phase_cyc = self.phase_full[s:f]
        self.crs_cyc = self.crs_full[s:f]
        self.x_cyc = self.x_full[s:f]
        self.y_cyc = self.y_full[s:f]
        self.z_cyc = self.z_full[s:f]
        self.dx_cyc = self.dx_full[s:f]
        self.dy_cyc = self.dy_full[s:f]
        self.dz_cyc = self.dz_full[s:f]
        self.depower_cyc = self.depower_full[s:f]
        self.u_vals_cyc = self._compute_u(self.x_cyc, self.y_cyc, self.z_cyc)

    # -------------------------
    # Reel-In (RI_) extraction
    # -------------------------
    def _detect_RI_segment(self):
        """
        Find RI start index (first phase tag in set) relative to cycle slice.
        Use same tags as original: "pp-ri", "pp-rori", "pp-riro".
        """
        self.RI_idx0 = next(
            (i for i, tag in enumerate(self.phase_cyc) if tag.lower() in ["pp-ri"]),
            None,
        )
        self.RI_idxf = next(
            (i for i, tag in enumerate(self.phase_cyc) if tag.lower() in ["pp-riro"]),
            None,
        )
        if self.RI_idx0 is None or self.RI_idxf is None:
            raise ValueError("Reel-In not found in this cycle")
        
        self._assign_transition("RI", self.RI_idx0, self.RI_idxf)

    # -------------------------
    # Reel-Out (RO_) extraction
    # -------------------------
    def _extract_RO_segment(self):
        """RO is cycle portion before RI start (indices relative to cycle)."""

        self.RO_idxf = next(
            (i for i, tag in enumerate(self.phase_cyc) if tag.lower() in ["pp-rori"]),
            None,
        )

        s, f = 0, self.RO_idxf
        self.RO_az = self.az_cyc[s:f]
        self.RO_el = self.el_cyc[s:f]
        self.RO_r = self.r_cyc[s:f]
        self.RO_x = self.x_cyc[s:f]
        self.RO_y = self.y_cyc[s:f]
        self.RO_z = self.z_cyc[s:f]
        # derivatives on RO signals (used in Lissajous detection / transition heuristics)
        self.daz_RO = np.gradient(self.RO_az)
        self.del_RO = np.gradient(self.RO_el)
        self.RO_r0 = self.RO_r[0]

    # -------------------------
    # Lissajous detection on RO
    # -------------------------
    def _find_lissajous_bounds(self):
        """
        Find start and end indices for a consistent Lissajous loop on RO data.
        Heuristic: small azimuth near zero, positive derivatives, limited elevation.
        """
        self.Lissajous_p0 = []
        self.Lissajous_pf = []
        start_found = False
        self.Lissajous_idx0 = None
        self.Lissajous_idxf = None

        for i in range(self.RO_idxf+1):
            cond = (
                self.daz_RO[i] > 0
                and self.del_RO[i] > 0
                and -0.01 <= self.RO_az[i] <= 0.01
                and self.RO_el[i] <= 0.5
            )
            if cond and not start_found:
                self.Lissajous_idx0 = i
                self.Lissajous_p0.append((self.RO_az[i], self.RO_el[i]))
                start_found = True
            elif cond and start_found and i > self.Lissajous_idx0 + 10:
                self.Lissajous_idxf = i
                self.Lissajous_pf.append((self.RO_az[i], self.RO_el[i]))
                break

        if not start_found or self.Lissajous_idxf is None:
            raise ValueError("No valid Lissajous pattern found in Reel-Out data")

        # store truncated Lissajous signals (azimuth / elevation)
        self.Lissajous_az = self.RO_az[self.Lissajous_idx0 : self.Lissajous_idxf + 1]
        self.Lissajous_el = self.RO_el[self.Lissajous_idx0 : self.Lissajous_idxf + 1]
        self.Lissajous_r = self.RO_r[self.Lissajous_idx0 : self.Lissajous_idxf + 1]

    # -------------------------
    # Transitions detection & assignment helpers
    # -------------------------
    def _assign_transition(self, prefix, i0, i1):
        """Helper to set transition attributes for [prefix] between cycle indices i0..i1 (inclusive)."""
        az_slice = self.az_cyc[i0 : i1 + 1]
        el_slice = self.el_cyc[i0 : i1 + 1]
        x_slice = self.x_cyc[i0 : i1 + 1]
        y_slice = self.y_cyc[i0 : i1 + 1]
        z_slice = self.z_cyc[i0 : i1 + 1]
        dx_slice = self.dx_cyc[i0 : i1 + 1]
        dy_slice = self.dy_cyc[i0 : i1 + 1]
        dz_slice = self.dz_cyc[i0 : i1 + 1]
        r_slice = self.r_cyc[i0 : i1 + 1]

        setattr(self, f"{prefix}_az", az_slice)
        setattr(self, f"{prefix}_el", el_slice)
        setattr(self, f"{prefix}_x", x_slice)
        setattr(self, f"{prefix}_y", y_slice)
        setattr(self, f"{prefix}_z", z_slice)
        setattr(self, f"{prefix}_dx", dx_slice)
        setattr(self, f"{prefix}_dy", dy_slice)
        setattr(self, f"{prefix}_dz", dz_slice)
        setattr(self, f"{prefix}_r", r_slice)
        setattr(self, f"{prefix}_p0_sph", np.array([az_slice[0], el_slice[0]]))
        setattr(self, f"{prefix}_pf_sph", np.array([az_slice[-1], el_slice[-1]]))
        setattr(self, f"{prefix}_p0_cart", np.array([x_slice[0], y_slice[0], z_slice[0]]))
        setattr(self, f"{prefix}_pf_cart", np.array([x_slice[-1], y_slice[-1], z_slice[-1]]))
        setattr(self, f"{prefix}_r0", self.r_cyc[i0])
        setattr(self, f"{prefix}_r1", self.r_cyc[i1])
        setattr(self, f"{prefix}_crs0", self.crs_cyc[i0])
        setattr(self, f"{prefix}_crsf", self.crs_cyc[i1])
        setattr(self, f"{prefix}_v0", np.array([dx_slice[0], dy_slice[0], dz_slice[0]]))
        setattr(self, f"{prefix}_vf", np.array([dx_slice[-1], dy_slice[-1], dz_slice[-1]]))
        setattr(self, f"{prefix}_u_vals", self._compute_u(x_slice, y_slice, z_slice))

    def _combine_slices_for_RI_RO(self, slice1, slice2, i0, i1, prefix = "RI_RO"):
        # Combine the slices for the RI_RO transition
        az_combined = np.concatenate((getattr(self, f"{slice1}_az"), getattr(self, f"{slice2}_az")))
        el_combined = np.concatenate((getattr(self, f"{slice1}_el"), getattr(self, f"{slice2}_el")))
        x_combined = np.concatenate((getattr(self, f"{slice1}_x"), getattr(self, f"{slice2}_x")))
        y_combined = np.concatenate((getattr(self, f"{slice1}_y"), getattr(self, f"{slice2}_y")))
        z_combined = np.concatenate((getattr(self, f"{slice1}_z"), getattr(self, f"{slice2}_z")))
        dx_combined = np.concatenate((getattr(self, f"{slice1}_dx"), getattr(self, f"{slice2}_dx")))
        dy_combined = np.concatenate((getattr(self, f"{slice1}_dy"), getattr(self, f"{slice2}_dy")))
        dz_combined = np.concatenate((getattr(self, f"{slice1}_dz"), getattr(self, f"{slice2}_dz")))
        r_combined = np.concatenate((getattr(self, f"{slice1}_r"), getattr(self, f"{slice2}_r")))

        setattr(self, f"{prefix}_az", az_combined)
        setattr(self, f"{prefix}_el", el_combined)
        setattr(self, f"{prefix}_x", x_combined)
        setattr(self, f"{prefix}_y", y_combined)
        setattr(self, f"{prefix}_z", z_combined)
        setattr(self, f"{prefix}_dx", dx_combined)
        setattr(self, f"{prefix}_dy", dy_combined)
        setattr(self, f"{prefix}_dz", dz_combined)
        setattr(self, f"{prefix}_r", r_combined)
        setattr(self, f"{prefix}_p0_sph", np.array([az_combined[0], el_combined[0]]))
        setattr(self, f"{prefix}_pf_sph", np.array([az_combined[-1], el_combined[-1]]))
        setattr(self, f"{prefix}_p0_cart", np.array([x_combined[0], y_combined[0], z_combined[0]]))
        setattr(self, f"{prefix}_pf_cart", np.array([x_combined[-1], y_combined[-1], z_combined[-1]]))
        setattr(self, f"{prefix}_r0", self.r_cyc[i0])
        setattr(self, f"{prefix}_r1", self.r_cyc[i1])
        setattr(self, f"{prefix}_crs0", self.crs_cyc[i0])
        setattr(self, f"{prefix}_crsf", self.crs_cyc[i1])
        setattr(self, f"{prefix}_v0", np.array([dx_combined[0], dy_combined[0], dz_combined[0]]))
        setattr(self, f"{prefix}_vf", np.array([dx_combined[-1], dy_combined[-1], dz_combined[-1]]))
        setattr(self, f"{prefix}_u_vals", self._compute_u(x_combined, y_combined, z_combined))



    def _find_RI_RO_transition(self):
        """
        Find end index of RI->RO transition (search before Lissajous_idx0).
        Original heuristic: az_cyc[i] < 0 and del_RO[i] < 0 and daz_RO[i] < 0
        """
        self.RI_RO_idx0 = next(
            (i for i, tag in enumerate(self.phase_cyc) if tag.lower() in ["pp-riro"]),
            None,
        )
        self.RI_RO_idxf = None
        for i in range(self.Lissajous_idx0):
            if self.az_cyc[i] < 0 and self.del_RO[i] < 0 and self.daz_RO[i] < 0:
                self.RI_RO_idxf = i
                break
        if self.RI_RO_idxf is None:
            raise ValueError("No valid end point found for the RI->RO transition in the reel-out data.")
        self._assign_transition("RI_RO_1", self.RI_RO_idx0, len(self.time_cyc)-1)
        self._assign_transition("RI_RO_2", 0, self.RI_RO_idxf)

        self._combine_slices_for_RI_RO("RI_RO_1", "RI_RO_2", self.RI_RO_idx0, self.RI_RO_idxf, prefix="RI_RO")


    def _find_RO_RI_transition(self):
        """
        Find start index of RO->RI transition (search between Lissajous_idxf and RI_idx0).
        Heuristic: az_cyc[i] > 0.1 and del_RO[i] > 0 and daz_RO[i] < 0 and el_cyc[i] < 0.25
        """
        self.RO_RI_idx0 = None
        for i in range(self.Lissajous_idxf, self.RI_idx0):
            if self.az_cyc[i] > 0.1 and self.del_RO[i] > 0 and self.daz_RO[i] < 0 and self.el_cyc[i] < 0.25:
                self.RO_RI_idx0 = i
                break
        if self.RO_RI_idx0 is None:
            raise ValueError("No valid start point found for the RO->RI transition in the reel-out data.")
        self.RO_RI_idxf = self.RI_idx0
        self._assign_transition("RO_RI", self.RO_RI_idx0, self.RO_RI_idxf)

    # -------------------------
    # Plotting helpers (call plt.show() automatically)
    # -------------------------
    def plot_cycle_3D(self):
        """Plot the whole cycle in 3D and highlight RI endpoints."""
        self._plot3d_generic(
            self.x_cyc,
            self.y_cyc,
            self.z_cyc,
            title="Full Cycle Path",
        )

    def plot_RI_3D(self):
        """Plot the cycle and highlight the Reel-In segment."""
        self._plot3d_generic(
            self.x_cyc,
            self.y_cyc,
            self.z_cyc,
            seg=(self.RI_x, self.RI_y, self.RI_z),
            seg_label="Reel-In",
            start=self.RI_p0_cart,
            end=self.RI_pf_cart,
            title="Reel-In Phase",
        )

    def plot_RI_RO_3D(self):
        """Plot RI->RO transition segment in the context of the cycle."""
        self._plot3d_generic(
            self.x_cyc,
            self.y_cyc,
            self.z_cyc,
            seg=(self.RI_RO_x, self.RI_RO_y, self.RI_RO_z),
            seg_label="RI->RO",
            start=self.RI_RO_p0_cart,
            end=self.RI_RO_pf_cart,
            title="Reel-In → Reel-Out Transition",
        )

    def plot_RO_RI_3D(self):
        """Plot RO->RI transition segment in the context of the cycle."""
        self._plot3d_generic(
            self.x_cyc,
            self.y_cyc,
            self.z_cyc,
            seg=(self.RO_RI_x, self.RO_RI_y, self.RO_RI_z),
            seg_label="RO->RI",
            start=self.RO_RI_p0_cart,
            end=self.RO_RI_pf_cart,
            title="Reel-Out → Reel-In Transition",
        )

    def plot_Lissajous_path2D(self):
        """
        2D Lissajous plot (azimuth vs elevation) on the RO Lissajous segment.
        Mirrors your original plotting style and labels.
        """
        plt.figure()
        plt.plot(self.Lissajous_az, self.Lissajous_el, label="Lissajous (az vs el)")
        # use stored endpoints lists (they contain tuples)
        p0 = self.Lissajous_p0[0] if self.Lissajous_p0 else (self.Lissajous_az[0], self.Lissajous_el[0])
        pf = self.Lissajous_pf[0] if self.Lissajous_pf else (self.Lissajous_az[-1], self.Lissajous_el[-1])
        plt.scatter(p0[0], p0[1], color="green", label="Lissajous Start Point")
        plt.scatter(pf[0], pf[1], color="red", label="Lissajous End Point")
        plt.xlabel("Azimuth (rad)")
        plt.ylabel("Elevation (rad)")
        plt.title("Kite Path During Reel-Out Phase (Lissajous)")
        plt.legend()
        plt.show()

    def plot_Lissajous_path3D(self):
        """
        3D plot of the Reel-Out path with Lissajous start/end and whole RO start/end markers.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.RO_x, self.RO_y, self.RO_z, label="Reel-Out Path")
        ax.scatter(
            self.RO_x[self.Lissajous_idx0],
            self.RO_y[self.Lissajous_idx0],
            self.RO_z[self.Lissajous_idx0],
            color="green",
            label="Lissajous Start Point",
        )
        ax.scatter(
            self.RO_x[self.Lissajous_idxf],
            self.RO_y[self.Lissajous_idxf],
            self.RO_z[self.Lissajous_idxf],
            color="red",
            label="Lissajous End Point",
        )
        ax.scatter(self.RO_x[0], self.RO_y[0], self.RO_z[0], color="blue", label="RO Start Point")
        ax.scatter(self.RO_x[-1], self.RO_y[-1], self.RO_z[-1], color="orange", label="RO End Point")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Kite Path During Reel-Out Phase")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    # File paths
    base_path = "./processed_data/fitting"
    waypoint_path = f"{base_path}/2025-09-25_11-48-58_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"

    dp = DataProcessing(full_path, cycle_path, waypoint_path, cyc_idx=0)
    dp.plot_cycle_3D()
    dp.plot_RI_3D()
    dp.plot_RI_RO_3D()
    dp.plot_RO_RI_3D()
    dp.plot_Lissajous_path2D()
    dp.plot_Lissajous_path3D()
