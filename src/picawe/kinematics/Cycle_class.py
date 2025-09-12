import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- Helpers -----------------
def convert_time_to_seconds(time_array):
    """Convert time in HH:MM:SS.sss format to total seconds."""
    seconds_array = []
    for time_str in time_array:
        parts = re.split(r"[:.]", str(time_str))
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2]) + float("0." + parts[3]) if len(parts) > 3 else float(parts[2])
        seconds = h * 3600 + m * 60 + s
        seconds_array.append(seconds)
    return np.array(seconds_array)

def bspline_basis_matrix(u_vals, p, U, n_ctrl):
    """Numeric B-spline basis matrix using Cox-de Boor recursion."""
    u_vals = np.asarray(u_vals)
    N = np.zeros((len(u_vals), n_ctrl))
    for i in range(n_ctrl):
        left, right = U[i], U[i+1]
        if left <= right:
            N[:, i] = np.where((u_vals >= left) & (u_vals < right), 1.0, 0.0)
    N[np.isclose(u_vals, 1.0), -1] = 1.0

    for k in range(1, p+1):
        N_k = np.zeros_like(N)
        for i in range(n_ctrl):
            left_den = U[i+k] - U[i]
            right_den = U[i+k+1] - U[i+1]
            left = (u_vals - U[i]) / left_den * N[:, i] if left_den != 0 else 0
            right = (U[i+k+1] - u_vals) / right_den * N[:, i+1] if (right_den != 0 and i+1 < n_ctrl) else 0
            N_k[:, i] = left + right
        N = N_k
    return N

# B-spline parameters
p = 3
n_ctrl = 7
U = [0.0, 0.0, 0.0, 0.0, 1/4, 2/4, 3/4, 1.0, 1.0, 1.0, 1.0]

# ----------------- Cycle Class -----------------
class Cycle:
    def __init__(self, full_df, cycle_df, cycle_idx=1):
        self.full_df = full_df.copy()
        self.cycle_df = cycle_df.copy()
        self.cycle_idx = cycle_idx
        self._prepare_time_and_columns()
        self._compute_useful_cycles_idx()
        self.x, self.y, self.z, self.time = self.get_cycle_cartesian(cycle_idx)

    @staticmethod
    def from_files(file_path_full, file_path_cycle, cycle_idx=1):
        full_data = pd.read_csv(file_path_full, header=0, sep=r"\s+")
        cycle_data = pd.read_csv(file_path_cycle, header=0)
        return Cycle(full_data, cycle_data, cycle_idx)

    def _prepare_time_and_columns(self):
        self.full_df['time_s'] = np.round(convert_time_to_seconds(self.full_df['time_of_day'].to_numpy()), 1)
        self.cycle_df['start_time_s'] = np.round(convert_time_to_seconds(self.cycle_df['start_time_cycle_LT'].to_numpy()), 1)
        self.az = self.full_df['kite_azimuth'].to_numpy()      # radians
        self.el = self.full_df['kite_elevation'].to_numpy()    # radians
        self.r = self.full_df['kite_distance'].to_numpy()
        self.time_full = self.full_df['time_s'].to_numpy()
        self.time_cycle = self.cycle_df['start_time_s'].to_numpy()
        self.phase = self.full_df['flight_phase'].to_numpy()

    def _compute_useful_cycles_idx(self):
        idx_start_cycle = np.array([i for i, t in enumerate(self.time_full) for tc in self.time_cycle if t == tc])
        if len(idx_start_cycle) < 3:
            raise ValueError("Not enough cycle indices found in full and cycle files")
        self.useful_cycles_idx = idx_start_cycle[1:-1]

    def sph2cart_cycle(self, az, el, r):
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    def get_cycle_cartesian(self, cycle_idx):
        start_idx, end_idx = self.useful_cycles_idx[cycle_idx], self.useful_cycles_idx[cycle_idx+1]
        az_cyc, el_cyc, r_cyc = self.az[start_idx:end_idx], self.el[start_idx:end_idx], self.r[start_idx:end_idx]
        time_cyc = self.time_full[start_idx:end_idx] - self.time_full[start_idx]
        x, y, z = self.sph2cart_cycle(az_cyc, el_cyc, r_cyc)
        return np.array(x), np.array(y), np.array(z), time_cyc

    def find_start_end_RI(self, cycle_idx):
        """Find start/end of RI in spherical coords, plus gradients (numerical)."""
        start_idx, end_idx = self.useful_cycles_idx[cycle_idx], self.useful_cycles_idx[cycle_idx+1]
        az_cyc, el_cyc, r_cyc = self.az[start_idx:end_idx], self.el[start_idx:end_idx], self.r[start_idx:end_idx]
        phase_cyc = self.phase[start_idx:end_idx]
        start_RI = end_RI = None
        i_start, i_end = None, None

        def is_RI(tag):
            if tag in ["pp-ri", "pp-rori", "pp-riro"]:
                return True
            return False

        def is_RO(tag):
            if tag == "pp-ro":
                return True
            return False

        for i in range(1, len(phase_cyc)):
            prev_phase, curr_phase = (phase_cyc[i-1]).lower(), (phase_cyc[i]).lower()

            if is_RO(prev_phase) and is_RI(curr_phase) and start_RI is None:
                i_start, start_RI = i, (az_cyc[i], el_cyc[i], r_cyc[i])

            i_end, end_RI = len(phase_cyc)-1, (az_cyc[-1], el_cyc[-1], r_cyc[-1])

        if i_start is None or i_end is None:
            raise ValueError("RI phase not found in this cycle")

        # gradients in spherical coords
        az_grad, el_grad, r_grad = np.gradient(az_cyc), np.gradient(el_cyc), np.gradient(r_cyc)

        v0 = np.array([az_grad[i_start],
              el_grad[i_start],
              r_grad[i_start]])
        vf = np.array([az_grad[i_end],
              el_grad[i_end],
              r_grad[i_end]])
        
        # Return clean tuple for spline fitting
        return start_RI, v0, end_RI, vf, az_cyc[i_start:i_end+1], el_cyc[i_start:i_end+1], r_cyc[i_start:i_end+1], i_start, i_end


    def fit_RI_spline(self, T=1.0):
        """Fit B-spline in spherical coords only over RI."""
        start_RI, v0, end_RI, vf, az_cyc, el_cyc, r_cyc, i_start, i_end = self.find_start_end_RI(self.cycle_idx)
        # print(start_RI, end_RI)
        if not (i_start and i_end):
            raise ValueError("RI phase not found in cycle")

        S_sph = np.vstack([az_cyc, el_cyc, r_cyc]).T

        s_vals = np.linspace(0, T, len(S_sph))
        u_vals = s_vals / T
        N = bspline_basis_matrix(u_vals, p, U, n_ctrl)

        C_unknown, _, _, _ = np.linalg.lstsq(N[:, 2:5], S_sph, rcond=None)
        self.c2, self.c3, self.c4 = C_unknown.T

        # Build full control point array (7x3)
        C = np.zeros((n_ctrl, 3))

        C[0] = S_sph[0]        # fix first ctrl pt
        C[1] = C[0] + (1/3) * v0     # maybe also fix second? depends on BC

        C[2:5] = C_unknown     # fitted unknowns
        
        C[6] = S_sph[-1]       # fix last
        C[5] = C[6]  + (1/3) * vf     # fix near end
        # Evaluate fitted B-spline curve in spherical coords
        S_fitted_sph = N @ C   # (n_samples × n_ctrl) @ (n_ctrl × 3) = (n_samples × 3)

        # Convert to cartesian
        S_fitted_cart = np.array([self.sph2cart_cycle(*sph) for sph in S_fitted_sph])

        return start_RI, end_RI, S_fitted_sph, S_fitted_cart

    def plot_RI_fit(self, start_RI, end_RI, S_fitted_cart):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(self.x, self.y, self.z, label=f"Cycle {self.cycle_idx+1} Trajectory", alpha=0.6)
        ax.plot(S_fitted_cart[:,0], S_fitted_cart[:,1], S_fitted_cart[:,2], "r--", label="RI B-spline fit")
        ax.scatter(*self.sph2cart_cycle(*start_RI), color="green", s=50, label="Start RI")
        ax.scatter(*self.sph2cart_cycle(*end_RI), color="red", s=50, label="End RI")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.set_title(f"Cycle {self.cycle_idx+1}: Reel-In Spline Fit")
        ax.legend(); ax.set_box_aspect([1,1,1])
        plt.show()

# ----------------- Usage -----------------
if __name__ == "__main__":
    file_path_full = "/home/theophile/src/Simulation_Results/trial_Uri_valid/ProtoLogger_csv/2025-09-10_11-31-10_ProtoLogger.csv"
    file_path_cycle = "/home/theophile/src/Simulation_Results/trial_Uri_valid/cycles/cycle_data_sheet_lines.csv"

    cycle = Cycle.from_files(file_path_full, file_path_cycle, cycle_idx=1)

    start_RI, end_RI, S_fitted_sph, S_fitted_cart = cycle.fit_RI_spline(T=10.0)
    cycle.plot_RI_fit(start_RI, end_RI, S_fitted_cart)
