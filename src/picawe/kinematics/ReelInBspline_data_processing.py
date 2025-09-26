import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class ReelInBspline_data_processing:

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

    def __init__(self, file_path_full, file_path_cycle, cyc_idx=0):
        """Load CSVs, preprocess times, compute cycle + reel-in slices."""
        self.full_df = pd.read_csv(file_path_full)
        self.cycle_df = pd.read_csv(file_path_cycle)
        self.cyc_idx = cyc_idx

        # --- Time preprocessing ---
        self.full_df['time_s'] = np.round(
            self.convert_time_to_seconds(self.full_df['time_of_day']), 1
        )
        self.cycle_df['start_time_s'] = np.round(
            self.convert_time_to_seconds(self.cycle_df['start_time_cycle_LT']), 1
        )

        # --- Extract full data ---
        self.time_full = self.full_df['time_s'].to_numpy()
        self.time_cycle = self.cycle_df['start_time_s'].to_numpy()
        self.az = self.full_df['kite_azimuth'].to_numpy(dtype=float)
        self.el = self.full_df['kite_elevation'].to_numpy(dtype=float)
        self.r = self.full_df['kite_distance'].to_numpy(dtype=float)
        self.phase = self.full_df['flight_phase'].to_numpy(dtype=str)
        self.crs = self.full_df['kite_course'].to_numpy(dtype=float)

        # Cartesian coords + derivatives for full dataset
        self.x, self.y, self.z = self.sph2cart(self.az, self.el, self.r)
        self.dx, self.dy, self.dz = (
            np.gradient(self.x),
            np.gradient(self.y),
            np.gradient(self.z),
        )

        # --- Cycle start/end ---
        start_indices = [i for i, t in enumerate(self.time_full) if t in self.time_cycle]
        if self.cyc_idx >= len(start_indices) - 1:
            raise IndexError("cyc_idx out of range")

        self.cyc_idx0 = start_indices[self.cyc_idx]
        self.cyc_idxf = (
            start_indices[self.cyc_idx + 1] - 1
            if self.cyc_idx + 1 < len(start_indices)
            else len(self.time_full) - 1
        )

        # Cycle slices
        self.az_cyc = self.az[self.cyc_idx0 : self.cyc_idxf + 1]
        self.el_cyc = self.el[self.cyc_idx0 : self.cyc_idxf + 1]
        self.r_cyc = self.r[self.cyc_idx0 : self.cyc_idxf + 1]
        self.phase_cyc = self.phase[self.cyc_idx0 : self.cyc_idxf + 1]
        self.crs_cyc = self.crs[self.cyc_idx0 : self.cyc_idxf + 1]
        self.x_cyc = self.x[self.cyc_idx0 : self.cyc_idxf + 1]
        self.y_cyc = self.y[self.cyc_idx0 : self.cyc_idxf + 1]
        self.z_cyc = self.z[self.cyc_idx0 : self.cyc_idxf + 1]
        self.dx_cyc = self.dx[self.cyc_idx0 : self.cyc_idxf + 1]
        self.dy_cyc = self.dy[self.cyc_idx0 : self.cyc_idxf + 1]
        self.dz_cyc = self.dz[self.cyc_idx0 : self.cyc_idxf + 1]

        # --- Reel-In start/end ---
        self.ri_idx0 = next(
            (i for i, tag in enumerate(self.phase_cyc)
             if tag.lower() in ["pp-ri", "pp-rori", "pp-riro"]),
            None
        )
        if self.ri_idx0 is None:
            raise ValueError("Reel-In start not found in this cycle")
        self.ri_idxf = len(self.phase_cyc) - 1

        # Reel-in slices
        self.az_ri = self.az_cyc[self.ri_idx0 : self.ri_idxf + 1]
        self.el_ri = self.el_cyc[self.ri_idx0 : self.ri_idxf + 1]
        self.r_ri = self.r_cyc[self.ri_idx0 : self.ri_idxf + 1]
        self.crs_ri = self.crs_cyc[self.ri_idx0 : self.ri_idxf + 1]
        self.x_ri = self.x_cyc[self.ri_idx0 : self.ri_idxf + 1]
        self.y_ri = self.y_cyc[self.ri_idx0 : self.ri_idxf + 1]
        self.z_ri = self.z_cyc[self.ri_idx0 : self.ri_idxf + 1]
        self.dx_ri = self.dx_cyc[self.ri_idx0 : self.ri_idxf + 1]
        self.dy_ri = self.dy_cyc[self.ri_idx0 : self.ri_idxf + 1]
        self.dz_ri = self.dz_cyc[self.ri_idx0 : self.ri_idxf + 1]

        # Start/end points for reel-in
        self.ri_p0_sph = np.array([self.az_ri[0], self.el_ri[0]])
        self.ri_pf_sph = np.array([self.az_ri[-1], self.el_ri[-1]])
        self.ri_p0_cart = np.array([self.x_ri[0], self.y_ri[0], self.z_ri[0]])
        self.ri_pf_cart = np.array([self.x_ri[-1], self.y_ri[-1], self.z_ri[-1]])
        self.ri_crs0 = self.crs_ri[0]
        self.ri_crsf = self.crs_ri[-1]
        self.ri_v0 = np.array([self.dx_ri[0], self.dy_ri[0], self.dz_ri[0]])
        self.ri_vf = np.array([self.dx_ri[-1], self.dy_ri[-1], self.dz_ri[-1]])

    def convert_time_to_seconds(self, time_series):
        """Convert HH:MM:SS.sss strings to total seconds."""
        return np.array([
            float(h) * 3600 + float(m) * 60 + float(s)
            for h, m, s in (str(t).split(":") for t in time_series)
        ])

    def sph2cart(self, az, el, r):
        """Convert spherical to Cartesian coords."""
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    def plot_path_3D(self):
        """3D plot of the full path."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.x_cyc, self.y_cyc, self.z_cyc, label='Cycle Path')
        ax.plot(self.ri_p0_cart[0], self.ri_p0_cart[1], self.ri_p0_cart[2], 'go', label='Reel-In Start')
        ax.plot(self.ri_pf_cart[0], self.ri_pf_cart[1], self.ri_pf_cart[2], 'ro', label='Reel-In End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    full_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/ProtoLogger_csv/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = "/home/theophile/src/Simulation_Results/trial_Uri_valid_2/cycles/cycle_data_sheet_lines.csv"

    obj = ReelInBspline_data_processing(file_path_full=full_path, file_path_cycle=cycle_path, cyc_idx=0)
    obj.plot_path_3D()