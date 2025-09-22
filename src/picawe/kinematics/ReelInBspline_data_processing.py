import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
# from scripts.src.picawe.kinematics import ReelInBspline

class ReelInBspline_data_processing():
    def __init__(self, file_path_full, file_path_cycle, cycle_idx=0):
        """Initialize Cycle object from CSV files and cycle index."""
        # Load CSVs
        self.full_df = pd.read_csv(file_path_full)
        self.cycle_df = pd.read_csv(file_path_cycle)
        self.cycle_idx = cycle_idx

        # Preprocess times
        self.full_df['time_s'] = np.round(self.convert_time_to_seconds(self.full_df['time_of_day'].to_numpy()), 1)
        self.cycle_df['start_time_s'] = np.round(self.convert_time_to_seconds(self.cycle_df['start_time_cycle_LT'].to_numpy()), 1)

        # Extract variables
        self.time_full = self.full_df['time_s'].to_numpy()
        self.time_cycle = self.cycle_df['start_time_s'].to_numpy()
        self.az = self.full_df['kite_azimuth'].to_numpy()
        self.el = self.full_df['kite_elevation'].to_numpy()
        self.r = self.full_df['kite_distance'].to_numpy()
        self.phase = self.full_df['flight_phase'].to_numpy()
        self.course = self.full_df['kite_course'].to_numpy()

        # Compute cycle boundaries
        self._compute_cycle_indices()
        self._extract_cycle_data()


    @staticmethod
    def convert_time_to_seconds(self, time_array):
        """Convert HH:MM:SS.sss to seconds (float)."""
        seconds_array = []
        for time_str in time_array:
            h, m, s = map(float, str(time_str).split(":"))
            seconds_array.append(h * 3600 + m * 60 + s)
        return np.array(seconds_array)

    @staticmethod
    def sph2cart(self, az, el, r):
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    @staticmethod
    def cart2sph(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        az = np.arctan2(y, x)
        el = np.arcsin(z / r)
        return az, el, r

    def _compute_cycle_indices(self):
        """Find start/end indices of the selected cycle."""
        self.start_indices = np.array([
            i for i, t in enumerate(self.time_full) 
            for tc in self.time_cycle if t == tc
        ])
        if self.cycle_idx >= len(self.start_indices)-1:
            raise IndexError("cycle_idx out of range")
        self.cycle_start_idx = self.start_indices[self.cycle_idx]
        self.cycle_end_idx   = self.start_indices[self.cycle_idx + 1] - 1 if self.cycle_idx + 1 < len(self.start_indices) else len(self.time_full)-1

    def _extract_cycle_data(self):
        """Extract only the selected cycle data (spherical and cartesian)."""
        self.az_cyc = self.az[self.cycle_start_idx:self.cycle_end_idx+1]
        self.el_cyc = self.el[self.cycle_start_idx:self.cycle_end_idx+1]
        self.r_cyc  = self.r[self.cycle_start_idx:self.cycle_end_idx+1]
        self.phase_cyc = self.phase[self.cycle_start_idx:self.cycle_end_idx+1]
        self.course_cyc = self.course[self.cycle_start_idx:self.cycle_end_idx+1]

        self.x_cyc, self.y_cyc, self.z_cyc = self.sph2cart(self.az_cyc, self.el_cyc, self.r_cyc)
        self.dx_cyc, self.dy_cyc, self.dz_cyc = np.gradient(self.x_cyc), np.gradient(self.y_cyc), np.gradient(self.z_cyc)
        self.num_points = len(self.x_cyc)

    def investigate_course_computations(self):
        """Compare different course computations over the Reel-In segment in one plot."""

        # -----------------------------
        # Cartesian course from dx, dy
        # -----------------------------
        course_cart = np.arctan2(
            self.dy_cyc[self.RI_start_idx:self.RI_end_idx],
            self.dx_cyc[self.RI_start_idx:self.RI_end_idx]
        )
        course_cart = np.mod(course_cart, 2 * np.pi)

        # -----------------------------
        # Spherical course from az, el
        # -----------------------------
        az_dot = np.gradient(self.az_RI)
        el_dot = np.gradient(self.el_RI)

        course_sph = -np.arctan2(az_dot * np.cos(self.el_RI), el_dot) + 2*np.pi
        course_sph = np.mod(course_sph, 2 * np.pi)

        # -----------------------------
        # Reference course from CSV
        # -----------------------------
    
        course_ref = np.mod(self.course_RI, 2 * np.pi)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(course_cart, label="Cartesian course", color="C0")
        ax.plot(course_sph, label="Spherical course", color="C1")
        ax.plot(course_ref, label="Reference (CSV)", color="C2")

        ax.set_ylabel("Course [rad]")
        ax.set_xlabel("Sample index")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("Comparison of Course Computations over Reel-In Segment")

        plt.show()

    def compute_course_average_sph(self, C, U, u, start):
        """Compute average course over first or last k points of spline."""
        k = 2  # Number of points to average over

        if not start:
            S_sph, dS_sph = self.self.evaluate_bspline(C, p=3, U=U, u=u, return_derivative=True)
            course = -np.arctan2(dS_sph[-k:,0]*np.cos(S_sph[-k:,1]), dS_sph[-k:,1]) + 2*np.pi
            course = np.mod(course, 2 * np.pi)
            self.course_avg = np.mean(course)
            return self.course_avg
        if start:
            S_sph, dS_sph = self.self.evaluate_bspline(C, p=3, U=U, u=u, return_derivative=True)
            course = -np.arctan2(dS_sph[:k,0]*np.cos(S_sph[:k,1]), dS_sph[:k,1]) + 2*np.pi
            course = np.mod(course, 2 * np.pi)
            self.course_avg = np.mean(course)
            return self.course_avg

    def get_RI_RO_boundaries(self):
        """Compute start/end points and velocities of Reel-In."""
        # Find start of Reel-In
        RI_start_idx = None
        for i, tag in enumerate(self.phase_cyc):
            if tag.lower() in ["pp-ri", "pp-rori", "pp-riro"]:
                RI_start_idx = i
                break
        if RI_start_idx is None:
            raise ValueError("Reel-In start not found in this cycle")

        RI_end_idx = len(self.phase_cyc) - 1  # Last point of cycle

        self.RI_start_idx, self.RI_end_idx = RI_start_idx, RI_end_idx

        # Store points and velocities
        self.ri_start_point_sph = np.array([self.az_cyc[RI_start_idx], self.el_cyc[RI_start_idx]])
        self.ri_end_point_sph = np.array([self.az_cyc[RI_end_idx], self.el_cyc[RI_end_idx]])
        self.ri_start_point = np.array([self.x_cyc[RI_start_idx], self.y_cyc[RI_start_idx], self.z_cyc[RI_start_idx]])
        self.ri_end_point   = np.array([self.x_cyc[RI_end_idx], self.y_cyc[RI_end_idx], self.z_cyc[RI_end_idx]])
        self.ri_start_velocity = np.array([self.dx_cyc[RI_start_idx], self.dy_cyc[RI_start_idx], self.dz_cyc[RI_start_idx]])
        self.ri_end_velocity   = np.array([self.dx_cyc[RI_end_idx], self.dy_cyc[RI_end_idx], self.dz_cyc[RI_end_idx]])
        self.ri_start_course = self.course_cyc[RI_start_idx]
        self.ri_end_course   = self.course_cyc[RI_end_idx]

        # Spherical RI segment
        self.az_RI = self.az_cyc[RI_start_idx:RI_end_idx+1]
        self.el_RI = self.el_cyc[RI_start_idx:RI_end_idx+1]
        self.r_RI  = self.r_cyc[RI_start_idx:RI_end_idx+1]
        self.course_RI = self.course_cyc[RI_start_idx:RI_end_idx+1]

        # Cartesian RI segment
        self.x_RI, self.y_RI, self.z_RI = self.x_cyc[RI_start_idx:RI_end_idx+1], self.y_cyc[RI_start_idx:RI_end_idx+1], self.z_cyc[RI_start_idx:RI_end_idx+1]

        return (self.ri_start_point, self.ri_end_point,
                self.ri_start_velocity, self.ri_end_velocity,
                self.az_RI, self.el_RI, self.r_RI,
                self.RI_start_idx, self.RI_end_idx)



if __name__ == "__main__":
    pass