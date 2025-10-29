from unittest import skip
import numpy as np
import pandas as pd
import re
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

    def __init__(self, file_path_full, file_path_cycle, file_path_waypoints, cyc_idx=0, run_plots_DP=True):
        self.run_plots_DP = run_plots_DP
        
        # --- Load CSVs ---
        self.file_path_cycle = file_path_cycle

        self.wp_df = pd.read_csv(file_path_waypoints)

        if file_path_cycle.endswith(".txt"):

            with open(file_path_full, "r") as f:
                total_rows = sum(1 for _ in f) - 1  # minus 1 for header

            start = int(0.7 * total_rows)
            end = int(0.9 * total_rows)

            nrows = end - start  # number of rows to read
            skip = range(1, start + 1)  # skip first 'start' rows (keep header)

            self.full_df = pd.read_csv(file_path_full, skiprows=skip, nrows=nrows, sep='\s+')
        
        else:
            self.full_df = pd.read_csv(file_path_full, sep='\s+')

        self.cyc_idx = cyc_idx

        # --- Time preprocessing (rounded to 0.1s like original) ---
        self.wp_df["time_s"] = np.round(self._to_seconds(self.wp_df["time_string"]), 1)
        self.full_df["time_s"] = np.round(
            self._to_seconds(self.full_df["time_of_day"]), 1
        )

        # --- Full arrays ---
        self.time_waypoints = self.wp_df["time_s"].to_numpy()
        self.time_full = self.full_df["time_s"].to_numpy()

        if file_path_cycle.endswith(".txt"):
            cycle_pattern = re.compile(r"Cycle\s+\d+:\s+Start:\s+(\d{2}:\d{2}:\d{2})")

            cycle_times = []

            with open(file_path_cycle, "r") as f:
                for line in f:
                    # Look for cycle start times
                    cycle_match = cycle_pattern.search(line)
                    if cycle_match:
                        cycle_times.append(cycle_match.group(1))

            cycle_times = np.round(self._to_seconds(cycle_times), 1)

            self.time_cycles = np.array(cycle_times)

        else:
            self.cycle_df = pd.read_csv(file_path_cycle)
            self.cycle_df["start_time_s"] = np.round(
                self._to_seconds(self.cycle_df["start_time_cycle_LT"]), 1
            )
            self.time_cycles = self.cycle_df["start_time_s"].to_numpy()

        # primary spherical signals
        self.az_full = self.full_df["kite_azimuth"].astype(float).to_numpy()
        self.el_full = self.full_df["kite_elevation"].astype(float).to_numpy()
        self.r_full = self.full_df["kite_distance"].astype(float).to_numpy()
        self.phase_full = self.full_df["flight_phase"].astype(str).to_numpy()
        self.crs_full = self.full_df["kite_course"].astype(float).to_numpy()
        self.depower_full = self.full_df["kite_actual_depower"].astype(float).to_numpy()

        self.tension_tether_ground_full = (
            self.full_df["ground_tether_force"].astype(float).to_numpy()
        )  # Kg
        self.CL = self.full_df["lift_coeff"].astype(float).to_numpy()
        self.CD = self.full_df["drag_coeff"].astype(float).to_numpy()
        self.Mech_Power = self.full_df["ground_mech_power"].astype(float).to_numpy()
        if file_path_cycle.endswith(".txt"):
            self.Vtan = np.sqrt((
                self.r_full * np.gradient(self.az_full, self.time_full))**2
                + (self.r_full * np.gradient(self.el_full, self.time_full)
            )**2)
        else:
            self.Vtan = (
                self.full_df["kite_tangential_velocity_mps"].astype(float).to_numpy()
            )  # m/s
        self.Vr = (
            self.full_df["ground_tether_reelout_speed"].astype(float).to_numpy()
        )  # m/s

        self.wp_names = self.wp_df["waypoint_name"].astype(str).to_numpy()

        # Cartesian & derivatives for full dataset
        self.x_full, self.y_full, self.z_full = self._sph2cart(
            self.az_full, self.el_full, self.r_full
        )
        self.dx_full, self.dy_full, self.dz_full = (
            np.gradient(self.x_full),
            np.gradient(self.y_full),
            np.gradient(self.z_full),
        )

        # --- Cycle selection (cyc_) ---
        start_indices = [
            i for i, t in enumerate(self.time_full) if t in self.time_cycles
        ]
        if self.cyc_idx >= len(start_indices[1:-1]) - 1:
            raise IndexError("cyc_idx out of range")
        self.cyc_idx0 = start_indices[1:-1][self.cyc_idx]

        self.cyc_idxf = (
            start_indices[1:-1][self.cyc_idx + 1] - 1
            if self.cyc_idx + 1 < len(start_indices[1:-1])
            else len(self.time_full) - 1
        )

        self._extract_cycle_slice()

        self._extract_csv_RO_segment()

        self._find_lissajous_shape_bounds()

        self._find_RI_RO_transition()

        self._find_RO_RI_transition()

        self._RI_Spline_segment()

        self._RO_segment()

        # -------Plotting-----------
        if self.run_plots_DP:
            self.plot_cycle_3D()
            self.plot_RI_Spline_3D()
            self.plot_Lissajous_path2D()
            self.plot_Lissajous_path3D()

    # -------------------------
    # Utilities
    # -------------------------
    def _to_seconds(self, series):
        """Convert HH:MM:SS.sss strings to seconds."""
        return np.array(
            [
                float(h) * 3600 + float(m) * 60 + float(s)
                for h, m, s in (str(t).split(":") for t in series)
            ]
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

    def _plot3d_generic(
        self,
        x,
        y,
        z,
        seg=None,
        seg_label=None,
        seg_color="orange",
        start=None,
        end=None,
        title=None,
    ):
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
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
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
        self.tension_tether_ground_cyc = self.tension_tether_ground_full[s:f]
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
        self.CL_cyc = self.CL[s:f]
        self.CD_cyc = self.CD[s:f]
        self.Vtan_cyc = self.Vtan[s:f]
        self.Vr_cyc = self.Vr[s:f]
        self.Mech_Power_cyc = self.Mech_Power[s:f]

    # -------------------------
    # CSV Reel-Out extraction
    # -------------------------
    def _extract_csv_RO_segment(self):
        """RO is cycle portion before RI start (indices relative to cycle)."""

        self.csv_RO_idxf = next(
            (i for i, tag in enumerate(self.phase_cyc) if tag.lower() in ["pp-rori"]),
            None,
        )

        self.csv_RO_idx0 = 0

        self._assign_transition("csv_RO", self.csv_RO_idx0, self.csv_RO_idxf)

        self.csv_RO_daz = np.gradient(self.csv_RO_az)
        self.csv_RO_del = np.gradient(self.csv_RO_el)

    # ----------------------------------
    # Lissajous loop detection on CSV RO
    # ----------------------------------
    def _find_lissajous_shape_bounds(self):
        """
        Find start and end indices for a single consistent Lissajous loop on CSV RO data.
        Heuristic: small azimuth near zero, positive derivatives, limited elevation.
        """
        self.L_shape_p0 = []
        self.L_shape_pf = []
        start_found = False
        end_found = False
        self.L_shape_idx0 = None
        self.L_shape_idxf = None

        for i in range(len(self.csv_RO_daz)):
            cond_julia = (
                self.csv_RO_daz[i] > 0
                and self.csv_RO_del[i] > 0
                and -0.01 <= self.csv_RO_az[i] <= 0.01
                and self.csv_RO_el[i] <= 0.5
            )
            cond_experimental = (
                self.csv_RO_daz[i] < 0
                and self.csv_RO_del[i] > 0
                and -0.01 <= self.csv_RO_az[i] <= 0.03
                and self.csv_RO_el[i] <= 0.6
            )
            if self.file_path_cycle.endswith(".txt"):
                cond = cond_experimental
            else:
                cond = cond_julia

            if cond and not start_found:
                self.L_shape_idx0 = i
                self.L_shape_p0.append((self.csv_RO_az[i], self.csv_RO_el[i]))
                start_found = True
            elif cond and start_found and i > self.L_shape_idx0 + 10:
                self.L_shape_idxf = i
                self.L_shape_pf.append((self.csv_RO_az[i], self.csv_RO_el[i]))
                end_found = True
                break

        # if self.run_plots_DP:
        #     plt.figure()
        #     plt.plot(self.csv_RO_az, self.csv_RO_el)
        #     plt.scatter(self.csv_RO_az[0], self.csv_RO_el[0], color="green", label="Start Point")
        #     plt.scatter(self.csv_RO_az[-1], self.csv_RO_el[-1], color="red", label="End Point")
        #     plt.scatter(
        #         self.csv_RO_az[self.L_shape_idx0],
        #     self.csv_RO_el[self.L_shape_idx0],
        #     color="orange",
        #     label="Lissajous Start Point",
        #     )
        #     plt.scatter(
        #         self.csv_RO_az[self.L_shape_idxf],
        #         self.csv_RO_el[self.L_shape_idxf],
        #         color="purple",
        #         label="Lissajous End Point",
        #     )
        #     plt.legend()
        #     plt.show()

        if not start_found or not end_found:
            raise ValueError("No valid Lissajous pattern found in Reel-Out data")

        # store truncated Lissajous signals (azimuth / elevation)
        self.L_shape_az = self.csv_RO_az[self.L_shape_idx0 : self.L_shape_idxf + 1]
        self.L_shape_el = self.csv_RO_el[self.L_shape_idx0 : self.L_shape_idxf + 1]
        self.L_shape_r = self.csv_RO_r[self.L_shape_idx0 : self.L_shape_idxf + 1]
        self.L_shape_duration = (
            self.csv_RO_time[self.L_shape_idxf] - self.csv_RO_time[self.L_shape_idx0]
        )

    def _find_RI_RO_transition(self):
        """
        Find end index of RI->RO transition (search before L_shape_idx0).
        Original heuristic: az_cyc[i] < 0 and csv_RO_del[i] < 0 and csv_RO_daz[i] < 0
        """
        self.RI_RO_idxf = None
        for i in range(self.L_shape_idx0):
            cond_julia = (
                self.csv_RO_az[i] < 0
                and self.csv_RO_del[i] < 0
                and self.csv_RO_daz[i] < 0
            )
            cond_experimental = (
                self.csv_RO_az[i] < 0.32
                and self.csv_RO_del[i] < 0
                and self.csv_RO_daz[i] < 0
            )

            if self.file_path_cycle.endswith(".txt"):
                cond = cond_experimental
            else:
                cond = cond_julia

            if cond:
                self.RI_RO_idxf = i
                break
            
        # if self.run_plots_DP:
        #     plt.figure()
        #     plt.plot(self.csv_RO_az, self.csv_RO_el)
        #     plt.scatter(self.csv_RO_az[0], self.csv_RO_el[0], color="green", label="Start Point")
        #     plt.scatter(self.csv_RO_az[-1], self.csv_RO_el[-1], color="red", label="End Point")
        #     plt.scatter(
        #         self.csv_RO_az[self.L_shape_idx0],
        #         self.csv_RO_el[self.L_shape_idx0],
        #         color="orange",
        #         label="Lissajous Start Point",
        #     )
        #     plt.scatter(
        #         self.csv_RO_az[self.L_shape_idxf],
        #         self.csv_RO_el[self.L_shape_idxf],
        #         color="purple",
        #         label="Lissajous End Point",
        #     )
        #     plt.scatter(
        #         self.csv_RO_az[self.RI_RO_idxf],
        #         self.csv_RO_el[self.RI_RO_idxf],
        #         color="blue",
        #         label="RI->RO Transition End Point",
        #     )
        #     plt.legend()
        #     plt.show()

        if self.RI_RO_idxf is None:
            raise ValueError(
                "No valid end point found for the RI->RO transition in the reel-out data."
            )

    def _find_RO_RI_transition(self):
        """
        Find start index of RO->RI transition (search between L_shape_idxf and RI_idx0).
        Heuristic: az_cyc[i] > 0.1 and csv_RO_del[i] > 0 and csv_RO_daz[i] < 0 and el_cyc[i] < 0.25
        """
        self.RO_RI_idx0 = None
        for i in range(self.L_shape_idxf - 100, len(self.csv_RO_del) - 1):
            cond_julia = (
                self.csv_RO_az[i] > 0.1
                and self.csv_RO_del[i] > 0
                and self.csv_RO_daz[i] < 0
                and self.csv_RO_el[i] < 0.25
            )
            cond_experimental = (
                self.csv_RO_az[i] < 0.35
                and self.csv_RO_del[i] < 0
                and self.csv_RO_daz[i] < 0
                and self.csv_RO_el[i] < 0.4
            )

            if self.file_path_cycle.endswith(".txt"):
                cond = cond_experimental
            else:
                cond = cond_julia

            if cond:
                self.RO_RI_idx0 = i
                break
            
        if self.run_plots_DP:
            plt.figure()
            plt.plot(self.csv_RO_az, self.csv_RO_el)
            plt.scatter(self.csv_RO_az[0], self.csv_RO_el[0], color="green", label="Start Point")
            plt.scatter(self.csv_RO_az[-1], self.csv_RO_el[-1], color="red", label="End Point")
            plt.scatter(
                self.csv_RO_az[self.L_shape_idx0],
                self.csv_RO_el[self.L_shape_idx0],
                color="orange",
                label="Lissajous Start Point",
            )
            plt.scatter(
                self.csv_RO_az[self.L_shape_idxf],
                self.csv_RO_el[self.L_shape_idxf],
                color="purple",
                label="Lissajous End Point",
            )
            plt.scatter(self.csv_RO_az[self.RI_RO_idxf],
                        self.csv_RO_el[self.RI_RO_idxf],
                        color="cyan",
                        label="RI->RO Transition End Point",
            )
            plt.scatter(
                self.csv_RO_az[self.RO_RI_idx0],
                self.csv_RO_el[self.RO_RI_idx0],
                color="blue",
                label="RI->RO Transition Start Point",
            )
            plt.legend()
            plt.show()

        if self.RO_RI_idx0 is None:
            raise ValueError(
                "No valid start point found for the RO->RI transition in the reel-out data."
            )

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
        CL_slice = self.CL_cyc[i0 : i1 + 1]
        CD_slice = self.CD_cyc[i0 : i1 + 1]
        time_slice = self.time_cyc[i0 : i1 + 1] - self.time_cyc[i0]
        Vtan_slice = self.Vtan_cyc[i0 : i1 + 1]
        Vr_slice = self.Vr_cyc[i0 : i1 + 1]
        Mech_Power_slice = self.Mech_Power_cyc[i0 : i1 + 1]
        tension_slice = self.tension_tether_ground_cyc[i0 : i1 + 1]

        setattr(self, f"{prefix}_az", az_slice)
        setattr(self, f"{prefix}_el", el_slice)
        setattr(self, f"{prefix}_x", x_slice)
        setattr(self, f"{prefix}_y", y_slice)
        setattr(self, f"{prefix}_z", z_slice)
        setattr(self, f"{prefix}_dx", dx_slice)
        setattr(self, f"{prefix}_dy", dy_slice)
        setattr(self, f"{prefix}_dz", dz_slice)
        setattr(self, f"{prefix}_r", r_slice)
        setattr(self, f"{prefix}_CL", CL_slice)
        setattr(self, f"{prefix}_CD", CD_slice)
        setattr(self, f"{prefix}_time", time_slice)
        setattr(self, f"{prefix}_Vtan", Vtan_slice)
        setattr(self, f"{prefix}_Vr", Vr_slice)
        setattr(self, f"{prefix}_Mech_Power", Mech_Power_slice)
        setattr(self, f"{prefix}_tension_tether_ground", tension_slice)
        setattr(self, f"{prefix}_p0_sph", np.array([az_slice[0], el_slice[0]]))
        setattr(self, f"{prefix}_pf_sph", np.array([az_slice[-1], el_slice[-1]]))
        setattr(
            self, f"{prefix}_p0_cart", np.array([x_slice[0], y_slice[0], z_slice[0]])
        )
        setattr(
            self, f"{prefix}_pf_cart", np.array([x_slice[-1], y_slice[-1], z_slice[-1]])
        )
        setattr(self, f"{prefix}_r0", self.r_cyc[i0])
        setattr(self, f"{prefix}_r1", self.r_cyc[i1])
        setattr(self, f"{prefix}_crs0", self.crs_cyc[i0])
        setattr(self, f"{prefix}_crsf", self.crs_cyc[i1])
        setattr(self, f"{prefix}_v0", np.array([dx_slice[0], dy_slice[0], dz_slice[0]]))
        setattr(
            self, f"{prefix}_vf", np.array([dx_slice[-1], dy_slice[-1], dz_slice[-1]])
        )
        setattr(self, f"{prefix}_u_vals", self._compute_u(x_slice, y_slice, z_slice))

    def _combine_slices(self, slice1, slice2, i0, i1, prefix="RI_Spline"):
        # Combine the slices for the RI_RO transition
        az_combined = np.concatenate(
            (getattr(self, f"{slice1}_az"), getattr(self, f"{slice2}_az"))
        )
        el_combined = np.concatenate(
            (getattr(self, f"{slice1}_el"), getattr(self, f"{slice2}_el"))
        )
        x_combined = np.concatenate(
            (getattr(self, f"{slice1}_x"), getattr(self, f"{slice2}_x"))
        )
        y_combined = np.concatenate(
            (getattr(self, f"{slice1}_y"), getattr(self, f"{slice2}_y"))
        )
        z_combined = np.concatenate(
            (getattr(self, f"{slice1}_z"), getattr(self, f"{slice2}_z"))
        )
        dx_combined = np.concatenate(
            (getattr(self, f"{slice1}_dx"), getattr(self, f"{slice2}_dx"))
        )
        dy_combined = np.concatenate(
            (getattr(self, f"{slice1}_dy"), getattr(self, f"{slice2}_dy"))
        )
        dz_combined = np.concatenate(
            (getattr(self, f"{slice1}_dz"), getattr(self, f"{slice2}_dz"))
        )
        r_combined = np.concatenate(
            (getattr(self, f"{slice1}_r"), getattr(self, f"{slice2}_r"))
        )
        CL_combined = np.concatenate(
            (getattr(self, f"{slice1}_CL"), getattr(self, f"{slice2}_CL"))
        )
        CD_combined = np.concatenate(
            (getattr(self, f"{slice1}_CD"), getattr(self, f"{slice2}_CD"))
        )
        time_combined = np.concatenate(
            (
                getattr(self, f"{slice1}_time"),
                getattr(self, f"{slice2}_time") + (getattr(self, f"{slice1}_time")[-1]),
            )
        )
        Vr_combined = np.concatenate(
            (getattr(self, f"{slice1}_Vr"), getattr(self, f"{slice2}_Vr"))
        )
        Vtan_combined = np.concatenate(
            (getattr(self, f"{slice1}_Vtan"), getattr(self, f"{slice2}_Vtan"))
        )
        Mech_Power_combined = np.concatenate(
            (
                getattr(self, f"{slice1}_Mech_Power"),
                getattr(self, f"{slice2}_Mech_Power"),
            )
        )
        tension_combined = np.concatenate(
            (
                getattr(self, f"{slice1}_tension_tether_ground"),
                getattr(self, f"{slice2}_tension_tether_ground"),
            )
        )

        setattr(self, f"{prefix}_az", az_combined)
        setattr(self, f"{prefix}_el", el_combined)
        setattr(self, f"{prefix}_x", x_combined)
        setattr(self, f"{prefix}_y", y_combined)
        setattr(self, f"{prefix}_z", z_combined)
        setattr(self, f"{prefix}_dx", dx_combined)
        setattr(self, f"{prefix}_dy", dy_combined)
        setattr(self, f"{prefix}_dz", dz_combined)
        setattr(self, f"{prefix}_r", r_combined)
        setattr(self, f"{prefix}_CL", CL_combined)
        setattr(self, f"{prefix}_CD", CD_combined)
        setattr(self, f"{prefix}_time", time_combined)
        setattr(self, f"{prefix}_Vtan", Vtan_combined)
        setattr(self, f"{prefix}_Vr", Vr_combined)
        setattr(self, f"{prefix}_Mech_Power", Mech_Power_combined)
        setattr(self, f"{prefix}_tension_tether_ground", tension_combined)
        setattr(self, f"{prefix}_p0_sph", np.array([az_combined[0], el_combined[0]]))
        setattr(self, f"{prefix}_pf_sph", np.array([az_combined[-1], el_combined[-1]]))
        setattr(
            self,
            f"{prefix}_p0_cart",
            np.array([x_combined[0], y_combined[0], z_combined[0]]),
        )
        setattr(
            self,
            f"{prefix}_pf_cart",
            np.array([x_combined[-1], y_combined[-1], z_combined[-1]]),
        )
        setattr(self, f"{prefix}_r0", self.r_cyc[i0])
        setattr(self, f"{prefix}_r1", self.r_cyc[i1])
        setattr(self, f"{prefix}_crs0", self.crs_cyc[i0])
        setattr(self, f"{prefix}_crsf", self.crs_cyc[i1])
        setattr(
            self,
            f"{prefix}_v0",
            np.array([dx_combined[0], dy_combined[0], dz_combined[0]]),
        )
        setattr(
            self,
            f"{prefix}_vf",
            np.array([dx_combined[-1], dy_combined[-1], dz_combined[-1]]),
        )
        setattr(
            self,
            f"{prefix}_u_vals",
            self._compute_u(x_combined, y_combined, z_combined),
        )

    def _RI_Spline_segment(self):
        self._assign_transition("pref1", self.RO_RI_idx0, len(self.phase_cyc) - 1)
        self._assign_transition("pref2", 0, self.RI_RO_idxf)
        self._combine_slices(
            "pref1", "pref2", self.RO_RI_idx0, self.RI_RO_idxf, prefix="RI_Spline"
        )
        self.RI_spline_idx0 = self.RO_RI_idx0
        self.RI_spline_idxf = self.RI_RO_idxf

    def _RO_segment(self):
        self._assign_transition("RO", self.RI_RO_idxf, self.RO_RI_idx0)
        self.RO_idx0 = self.RI_RO_idxf
        self.RO_idxf = self.RO_RI_idx0

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

    def plot_RI_Spline_3D(self):
        """Plot the cycle and highlight the Single Spline segment."""
        self._plot3d_generic(
            self.x_cyc,
            self.y_cyc,
            self.z_cyc,
            seg=(self.RI_Spline_x, self.RI_Spline_y, self.RI_Spline_z),
            seg_label="Single Spline Fit",
            start=self.RI_Spline_p0_cart,
            end=self.RI_Spline_pf_cart,
            title="Single Spline Fit",
        )

    def plot_Lissajous_path2D(self):
        """
        2D Lissajous plot (azimuth vs elevation) on the RO Lissajous segment.
        Mirrors your original plotting style and labels.
        """
        plt.figure()
        plt.plot(self.L_shape_az, self.L_shape_el, label="Lissajous (az vs el)")
        # use stored endpoints lists (they contain tuples)
        p0 = (
            self.L_shape_p0[0]
            if self.L_shape_p0
            else (self.L_shape_az[0], self.L_shape_el[0])
        )
        pf = (
            self.L_shape_pf[0]
            if self.L_shape_pf
            else (self.L_shape_az[-1], self.L_shape_el[-1])
        )
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
        ax.plot(self.csv_RO_x, self.csv_RO_y, self.csv_RO_z, label="Reel-Out Path")
        ax.scatter(
            self.csv_RO_x[self.L_shape_idx0],
            self.csv_RO_y[self.L_shape_idx0],
            self.csv_RO_z[self.L_shape_idx0],
            color="green",
            label="Lissajous Start Point",
        )
        ax.scatter(
            self.csv_RO_x[self.L_shape_idxf],
            self.csv_RO_y[self.L_shape_idxf],
            self.csv_RO_z[self.L_shape_idxf],
            color="red",
            label="Lissajous End Point",
        )
        ax.scatter(
            self.csv_RO_x[0],
            self.csv_RO_y[0],
            self.csv_RO_z[0],
            color="blue",
            label="RO Start Point",
        )
        ax.scatter(
            self.csv_RO_x[-1],
            self.csv_RO_y[-1],
            self.csv_RO_z[-1],
            color="orange",
            label="RO End Point",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Kite Path During Reel-Out Phase")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    # File paths
    base_path = "./processed_data/fitting"
    waypoint_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/2025-10-23_09-43-50_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"

    # base_path = "./processed_data/experimental"
    # waypoint_path = f"{base_path}/2024-11-05_12-58-54_ProtoLogger_waypoints.csv"
    # full_path = f"{base_path}/2024-11-05_12-58-54_ProtoLogger.csv"
    # cycle_path = f"{base_path}/2024-11-05_12-58-54_full_log.txt"

    dp = DataProcessing(full_path, cycle_path, waypoint_path)