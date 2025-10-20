import numpy as np
from picawe.kinematics.my_data_processing import DataProcessing
import json
import matplotlib.pyplot as plt

# Load JSON
with open("src/picawe/kinematics/pp_ws6-9_GS3_KCU4.A_KiteV9.60.A.json") as f:
    data = json.load(f)

# Access the list
json_trajectory = data["trajectory"]

'''
Json file format:

"trajectory" is a list of dictionaries, one dictionary for each waypoint.
Each waypoint dictionary contains:

- "attractor_point"
- "depower"
- "depower_control_type"
- "description"
- "events"
- "name" (waypoint name)
- "switch_criteria" (a list of dictionaries)
- "type"
- "winch_control"
    - "wcs_drum_control" : {"force_control_curve": {"force_knee" and "force_slope_factor"}, "controller_type"}
    - "winch_control_v1" : {"f_high", "f_low", "reelout_speed"}

'''

''' 
DISCLAIMER:

The parameters extracted from the JSON file, are not actual numbers that make sense physically.
These parameters are taken through modifications to lead to the final winch controller curve.
These are thus unitless but I will explain what they mean and how they affect the winch controller curve:

- f_low: This affects the low force limit
- f_high: This affects the high force limit
- reelout_speed: This is an offset that will shift the curve left and right (affecting minimum force to reel out)
- force_knee: This affects when the curve breaks the linear pattern and tapers off towards max force
- kp_v: This affects the slope of the curve, (less negative values lead to flatter slope, but also shifts the curve right)
- kp_f: This affects the slope of the curve very slightly, higher values lead to steeper slope, and also shifts the curve up
- force_slope_factor: This affects the curvature at the knee, higher values lead to less sharp knee (higher curvature -> more rounded).
    The force_slope_factor also affects the final F_max of the curve, a higher value for the slope factor leads to a higher F_max, as the knee is less sharp.
'''

''' 
How do the JSON parameters relate to Uri's parameters:

Uri's parameters:
"force_model": "quadratic",  # "linear" or "quadratic"
"max_tether_force": 3.5e4,  # N, only for force reeling
"min_tether_force": 4400.0,  # N, only for force reeling
"softplus": True,
"softplus_beta": 1e-4,
"softminus": True,
"softminus_beta": 1e-3,
"slope": 1800,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
"offset": -2,  # m/s

---------------------------------------------------------------

Connections (Uri's parameters  <->  JSON parameters):
- max_tether_force  <-> f_high
- min_tether_force <-> f_low
- offset <-> reelout_speed
- slope <-> kp_v and kp_f (combined effect)
- softplus and softminus <-> force_knee and force_slope_factor (combined effect)

'''

class Winch_and_Depower_data_processing(DataProcessing):
    def __init__(self, file_path_full, file_path_cycle, file_path_waypoints, json_trajectory, cyc_idx=0):
        super().__init__(file_path_full, file_path_cycle, file_path_waypoints, cyc_idx)

        self.json_trajectory = json_trajectory

        # RIRO means reel in to reel out (Leftover before Lissajous)
        # RORI means reel out to reel in (Leftover after Lissajous)
        # RI means reel in (csv RI + RIRO + RORI)
        # RO means reel out (csv RO)



        # Start of RIRO:
        self.RIRO_t0 = self.time_cyc[self.RI_RO_idx0]

        # Start of reel out:
        self.RO_t0 = self.time_cyc[self.Lissajous_idx0]

        # Start of RORI:
        self.RORI_t0 = self.time_cyc[self.RO_RI_idx0]

        # Start of reel in:
        self.RI_t0 = self.time_cyc[self.RI_idx0-21]

        # print(f"\nStarting indices: \n RIRO: {self.RIRO_idx0} \n RO: {self.RO_idx0} \n RORI: {self.RORI_idx0} \n RI: {self.RI_idx0} \n")
        # print(f"Length of cycle: {len(self.az_cyc)} \n")

        # Waypoint data
        self._find_cycle_wp0_wpf()
        self._extrapolate_wp_names()
        self._winch_and_depower_dictionary()
        self._create_settings_lists_for_cyc()
        self._plot_settings_over_cycle_time()
    
    # -------------------------
    # Waypoint inter/extrapolation
    # -------------------------

    def _find_cycle_wp0_wpf(self):

        self.wp0_idx = None
        self.wpf_idx = None
        self.wp0 = None
        self.wpf = None

        # print(self.time_waypoints, "\n")
        # print(self.time_cyc[0], "\n", self.time_cyc[-1], "\n")

        for i, t in enumerate(self.time_waypoints):
            if t > self.time_cyc[0] and self.wp0 is None:
                # print("Got the start!", t)
                self.wp0_idx = i-1
                self.wp0 = self.wp_names[self.wp0_idx]
            elif t >= self.time_cyc[-1] and self.wpf is None:
                # print("got the end!", t)
                self.wpf_idx = i
                self.wpf = self.wp_names[self.wpf_idx]
                break

        # print(f"Cycle starts at waypoint {self.wp0} (idx {self.wp0_idx}) and ends at waypoint {self.wpf} (idx {self.wpf_idx})")
        
    def _extrapolate_wp_names(self):

        self.cyc_switch_idx = []
        self.extrapolated_wp_names = []

        current_idx = self.wp0_idx
        last_time = 0

        for t in self.time_waypoints[self.wp0_idx+1:self.wpf_idx+1]:
            for i, t_cyc in enumerate(self.time_cyc[last_time:]):
                if t_cyc < t:
                    self.extrapolated_wp_names.append(self.wp_names[current_idx])
                else:
                    last_time += i
                    self.cyc_switch_idx.append(i)
                    current_idx += 1
                    break
    
    def _winch_and_depower_dictionary(self):
        my_dict = {}

        for waypoint in self.json_trajectory:
            wp_name = waypoint["name"].replace(" ", "_")

            # handle possible missing winch_control section
            winch_ctrl = waypoint.get("winch_control", {})
            wcs_drum = winch_ctrl.get("wcs_drum_control", {})
            winch_v1 = winch_ctrl.get("winch_control_v1", {})
            force_curve = wcs_drum.get("force_control_curve", {})

            my_dict[wp_name] = {
                "depower": waypoint.get("depower", {}).get("depower"),
                "depower_control_type": waypoint.get("depower_control_type"),
                "controller_type": wcs_drum.get("controller_type"),
                "force_knee": force_curve.get("force_knee"),
                "force_slope_factor": force_curve.get("force_slope_factor"),
                "f_high": winch_v1.get("f_high"),
                "f_low": winch_v1.get("f_low"),
                "reelout_speed": winch_v1.get("reelout_speed"),
                "kp_f": wcs_drum.get("pid_force", {}).get("Kp"),
                "kp_v": wcs_drum.get("pid_velocity", {}).get("Kp"),
            }

        self.waypoint_data_dictionary = my_dict
        # print(self.waypoint_data_dictionary)
    
    def _create_settings_lists_for_cyc(self):
        self.f_low = []
        self.f_high = []
        self.reelout_speed = []
        self.force_slope_factor = []
        self.force_knee = []
        self.depower = []
        self.kp_v = []
        self.kp_f = []

        for name in self.extrapolated_wp_names:
            self.f_low.append([self.waypoint_data_dictionary[name]["f_low"] if self.waypoint_data_dictionary[name]["f_low"] is not None else np.nan])
            self.f_high.append([self.waypoint_data_dictionary[name]["f_high"] if self.waypoint_data_dictionary[name]["f_high"] is not None else np.nan])
            self.reelout_speed.append([self.waypoint_data_dictionary[name]["reelout_speed"] if self.waypoint_data_dictionary[name]["reelout_speed"] is not None else np.nan])
            self.force_slope_factor.append([self.waypoint_data_dictionary[name]["force_slope_factor"] if self.waypoint_data_dictionary[name]["force_slope_factor"] is not None else np.nan])
            self.force_knee.append([self.waypoint_data_dictionary[name]["force_knee"] if self.waypoint_data_dictionary[name]["force_knee"] is not None else np.nan])
            self.depower.append([self.waypoint_data_dictionary[name]["depower"] if self.waypoint_data_dictionary[name]["depower"] is not None else np.nan])
            self.kp_v.append([self.waypoint_data_dictionary[name]["kp_v"] if self.waypoint_data_dictionary[name]["kp_v"] is not None else np.nan])
            self.kp_f.append([self.waypoint_data_dictionary[name]["kp_f"] if self.waypoint_data_dictionary[name]["kp_f"] is not None else np.nan])

    def _plot_settings_over_cycle_time(self):
        time_cyc = self.time_cyc - self.time_cyc[0]
        
        # Calculate phase starting times relative to cycle start
        RIRO_t0_rel = self.RIRO_t0 - self.time_cyc[0]
        RO_t0_rel = self.RO_t0 - self.time_cyc[0]
        RORI_t0_rel = self.RORI_t0 - self.time_cyc[0]
        RI_t0_rel = self.RI_t0 - self.time_cyc[0]

        plt.figure(figsize=(20, 20))

        # Define phase lines and labels
        phase_times = [RIRO_t0_rel, RO_t0_rel, RORI_t0_rel, RI_t0_rel]
        phase_labels = ['RIRO', 'RO', 'RORI', 'RI']
        phase_colors = ['blue', 'green', 'orange', 'red']

        plt.subplot(7, 1, 1)
        plt.plot(time_cyc, self.f_low, label="f_low")
        plt.plot(time_cyc, self.f_high, label="f_high")
        for i, (t, label, color) in enumerate(zip(phase_times, phase_labels, phase_colors)):
            plt.axvline(x=t, color=color, linestyle='--', alpha=0.7, label=label)
        plt.ylabel("Force")
        plt.legend()
        plt.grid()

        plt.subplot(7, 1, 2)
        plt.plot(time_cyc, self.reelout_speed, label="reelout_speed", color='orange')
        for t, label, color in zip(phase_times, phase_labels, phase_colors):
            plt.axvline(x=t, color=color, linestyle='--', alpha=0.7, label=label)
        plt.ylabel("Reelout Speed offset")
        plt.legend()
        plt.grid()

        plt.subplot(7, 1, 3)
        plt.plot(time_cyc, self.depower, label="depower", color='green')
        for t, label, color in zip(phase_times, phase_labels, phase_colors):
            plt.axvline(x=t, color=color, linestyle='--', alpha=0.7, label=label)
        plt.ylabel("Depower Setting")
        plt.legend()
        plt.grid()

        plt.subplot(7, 1, 4)
        plt.plot(time_cyc, self.force_slope_factor, label="force_slope_factor", color='purple')
        for t, label, color in zip(phase_times, phase_labels, phase_colors):
            plt.axvline(x=t, color=color, linestyle='--', alpha=0.7, label=label)
        plt.ylabel("Force Slope Factor")
        plt.legend()
        plt.grid()

        plt.subplot(7, 1, 5)
        plt.plot(time_cyc, self.force_knee, label="force_knee", color='red')
        for i, (t, label, color) in enumerate(zip(phase_times, phase_labels, phase_colors)):
            plt.axvline(x=t, color=color, linestyle='--', alpha=0.7, label=label)
        plt.ylabel("Force Knee")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid()

        plt.subplot(7, 1, 6)
        plt.plot(time_cyc, self.kp_v, label="kp_v", color='red')
        for i, (t, label, color) in enumerate(zip(phase_times, phase_labels, phase_colors)):
            plt.axvline(x=t, color=color, linestyle='--', alpha=0.7, label=label)
        plt.ylabel("Kp Velocity")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid()

        plt.subplot(7, 1, 7)
        plt.plot(time_cyc, self.kp_f, label="kp_f", color='red')
        for i, (t, label, color) in enumerate(zip(phase_times, phase_labels, phase_colors)):
            plt.axvline(x=t, color=color, linestyle='--', alpha=0.7, label=label)
        plt.ylabel("Kp Force")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # File paths
    base_path = "./processed_data/fitting"
    waypoint_path = f"{base_path}/2025-09-25_11-48-58_ProtoLogger_waypoints.csv"
    full_path = f"{base_path}/2025-09-25_11-48-58_ProtoLogger.csv"
    cycle_path = f"{base_path}/cycle_data_sheet_lines.csv"
    obj = Winch_and_Depower_data_processing(full_path, cycle_path, waypoint_path, json_trajectory)