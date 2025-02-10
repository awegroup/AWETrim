import numpy as np
DEFAULT_BOUNDS = {
    "tension_tether_ground": [0, 1e5],
    "input_steering": [-10, 10],
    "s_dot": [0, 50],
    "s_ddot": [-50, 50],
    "speed_tangential": [0, 50],
    "angle_roll": [-np.pi/4, np.pi/4],
    "timeder_angle_course": [-np.pi, np.pi],
    "angle_pitch": [-np.pi/4, np.pi/4],
    "angle_yaw": [-np.pi/4, np.pi/4],

}