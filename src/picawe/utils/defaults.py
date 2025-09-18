import numpy as np

DEFAULT_BOUNDS = {
    "tension_tether_ground": [0, 1e12],
    "input_steering": [-1, 1],
    "s_dot": [-10, 30],
    "s_ddot": [-100, 100],
    "speed_tangential": [0, 400],
    "angle_roll": [-np.pi / 2, np.pi / 2],
    "timeder_angle_course": [-np.pi, np.pi],
    "angle_pitch": [-np.pi / 4, np.pi / 4],
    "angle_yaw": [-np.pi / 4, np.pi / 4],
    "angle_elevation": [-np.pi, np.pi],
    "speed_radial": [-10, 10],
    "length_tether": [0, 1000],
    "distance_radial": [0, 1000],
}

PLOT_LABELS = {
    "acceleration_normal": "$a_n$ ($m/s^2$)",
    "acceleration_radial": "$a_r$ ($m/s^2$)",
    "acceleration_tangential": "$a_{\\tau}$ ($m/s^2$)",
    "angle_course": "$\\chi$ ($^\\circ$)",
    "angle_flight_path_aerodynamic": "$\\gamma_a$ ($^\\circ$)",
    "angle_heading_aerodynamic": "$\\chi_a$ ($^\\circ$)",
    "steering_input": "$u_s$",
    "angle_roll": "$\\phi_a$ ($^\\circ$)",
    "dchi_ds": "$d\\chi / ds$",
    "distance_radial": "$r$ (m)",
    "force_tether_talmar": "$F_{t, \\mathrm{Talmar}}}$",
    "input_steering": "$u_s$",
    "phi_unwrapped": "$\\phi_a$ ($^\\circ$)",
    "ratio_kinematic": "$\\kappa$",
    "ratio_tether": "$\\frac{F_t}{F_{t, \\mathrm{Talmar}}}}$",
    "s": "s (-)",
    "s_dot": "$\\dot{s}$ (-)",
    "s_ddot": "$\\ddot{s}$ (-)",
    "speed": "$v_k$ (m/s)",
    "speed_radial": "$v_r$ (m/s)",
    "speed_tangential": "$v_\\tau$ (m/s)",
    "speed_wind_true": "$v_{w,true}$ (m/s)",
    "speed_wind_apparent": "$v_{w,app}$ (m/s)",
    "tension_tether_ground": "$F_{t,g}$ (N)",
    "tension_kite": "$F_{t,k}$ (N)",
    "power_ground": "$P_g$ (W)",
    "kite_angle_of_attack": "$\\alpha$ ($^\\circ$)",
    "time": "$t$ (s)",
    "timeder_angle_course": "$\\dot{\\chi}$ ($^\\circ$/s)",
    "timeder_speed_radial": "$\\dot{v}_r$ ($m/s^2$)",
    "timeder_speed_tangential": "$\\dot{v}_\\tau$ ($m/s^2$)",
    "x": "x (m)",
    "y": "y (m)",
    "z": "z (m)",
    "input_steering": "$u_s$",
    "phase": "$\\Phi (^\circ)$",
    "angle_azimuth": "$\\phi (^\\circ)$",
    "angle_elevation": "$\\beta (^\\circ)$",
    "angle_of_attack": "$\\alpha (^\\circ)$",
}

PLOT_PARAMETERS = [
    "speed_tangential",
    "speed_radial",
    "input_steering",
    "tension_tether_ground",
]

DEFAULT_PATTERN_CONFIG = {
    "pattern_type": "helix",
    "initial_parameters": {
        "omega": -1.0,
        "r0": 200.0,
        "d0": 100.0,
        "vr": 0.2,
        "beta": 0.4,
        "kappa": 1,
    },
    "optimization_parameters": {
        # Add any optimization-related parameters here if needed as list of names
        "d0",
        "kappa",
    },
}

# defaults.py (This is your file containing the limits)
DEFAULT_OPTI_LIMITS = {
    "tension_tether_ground": (1e-2, 1e9),  # Range for tension_tether_ground: 0 to 1e9
    "input_steering": (-1, 1),  # Range for input_steering: -pi/2 to pi/2
    "s_dot": (0.01, 40),  # Range for s_dot: 0 to 30
    "s_ddot": (-100, 100),  # Range for s_ddot: -100 to 100
    "s": (0, 300),  # Range for s: 0 to 10
    "angle_elevation": (0.0, np.pi / 2),  # Range for angle_elevation: 0 to pi
    # HElix
    "kappa": (0, 1),  # Range for kappa: 0 to 5
    # "vr": (-10, 10),      # Range for vr: 0 to 100
    "beta0": (0.35, 1),  # Range for beta: 20 ot 50 degrees
    "d0": (40, 500),  # Range for d0: 0 to 100
    # Figure Eight
    "ry": (60, 180),  # Range for ry: 0 to 100
    "rz": (60, 180),  # Range for rz: 0 to 100
    "ky": (0.5, 1),  # Range for ky: 0 to 100
    "kz": (0.5, 1),  # Range for kz: 0 to 100
    "vr": (0.5, 4),  # Range for vr: 0 to 100
    "az_amp0": (
        np.radians(0),
        np.radians(50),
    ),  # Range for azimuth amplitude: 10 to 30 degrees
    "beta_amp0": (
        np.radians(0),
        np.radians(30),
    ),  # Range for beta amplitude: 5 to 20 degrees
    "beta_coeffs": (-1, 1),  # Range for beta coefficients: -1 to 1
    "az_coeffs": (-1, 1),  # Range for azimuth coefficients: -1 to 1
    "speed_radial": (0.2, 6),
    "distance_radial": (100, 2000),
    "k_vr": (0.5, 1.5),
}
