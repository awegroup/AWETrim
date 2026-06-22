# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

DEFAULT_BOUNDS = {
    "tension_tether_ground": [0, 1e12],
    "input_steering": [-3, 3],
    "s_dot": [1e-8, 5],
    "s_ddot": [-100, 100],
    "speed_tangential": [0, 200],
    "angle_roll": [-np.pi / 2, np.pi / 2],
    "timeder_angle_course": [-np.pi, np.pi],
    "angle_pitch": [-np.pi / 4, np.pi / 4],
    "angle_yaw": [-np.pi / 4, np.pi / 4],
    "angle_elevation": [-np.pi, np.pi],
    "speed_radial": [-10, 15],
    "length_tether": [0, 1000],
    "distance_radial": [0, 2000],
    "speed_friction": [0, 5],
    "angle_of_attack": [np.radians(-2), np.radians(18)],
    "direction_wind": [-np.pi, np.pi],
    "tension_tether_kite": [300, 1e9],
    "tether_length": [0, 2000],
    # ``azimuth_last_element`` is widened to ``[-2π, 2π]`` so IPOPT has slack
    # to cross the spherical-azimuth ``±π`` wrap (active when the kite flies
    # near the wind-frame zenith) without locking up at the bound.
    "azimuth_last_element": [-2 * np.pi, 2 * np.pi],
    "elevation_last_element": [-np.pi / 2, np.pi / 2],
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

# Default configuration for the winch subsystem
# Values align with common bounds used elsewhere
# and provide reasonable physical limits for simulations.
DEFAULT_WINCH_CONFIG = {
    "max_tether_length": 2000.0,  # m
    "min_tether_length": 100.0,  # m
    "max_speed": 8.0,  # m/s (reel-out positive)
    "min_speed": -6.0,  # m/s (reel-in negative)
    "max_acceleration": 2.0,  # m/s^2
    "min_acceleration": -2.0,  # m/s^2
    # Softplus sharpness for force limiting in winch model
    "sharpness_beta": 1e-4,
}

# defaults.py (This is your file containing the limits)
DEFAULT_OPTI_LIMITS = {
    "tension_tether_ground": (
        300,
        8.4e6,
    ),  # Range for tension_tether_ground: 0 to 8.4e5 N (max tether force)
    # u_s = -kcu_actual_steering/100; 2019 V3 flight max deflection was kcu +-35.
    "input_steering": (-0.35, 0.35),
    "s_dot": (0.0, 40),  # Range for s_dot: 0 to 30
    "s_ddot": (-100, 100),  # Range for s_ddot: -100 to 100
    "s": (0, 300),  # Range for s: 0 to 10
    "angle_elevation": (0.0, np.radians(160)),  # Range for angle_elevation: 0 to pi
    # HElix
    "kappa": (0, 1),  # Range for kappa: 0 to 1
    "kbeta": (0, 1),  # Range for kbeta: 0 to 2
    # "vr": (-10, 10),      # Range for vr: 0 to 100
    "beta0": (np.radians(12), np.radians(50)),  # Range for beta: 20 ot 50 degrees
    "phi0": (np.radians(-5), np.radians(5)),  # Range for phi0: 0 to 360 degrees
    "az_amp0": (
        np.radians(5),
        np.radians(40),
    ),  # Range for azimuth amplitude: 10 to 30 degrees
    "beta_amp0": (
        np.radians(2),
        np.radians(20),
    ),  # Range for beta amplitude: 5 to 20 degrees
    "beta_coeffs": (-1, 1),  # Range for beta coefficients: -1 to 1
    "az_coeffs": (-1, 1),  # Range for azimuth coefficients: -1 to 1
    "speed_radial": (-10, 10),
    "distance_radial": (100, 360),
    "k_vr": (0.5, 1.5),
    "slope_winch_ro": (3000, 30000),  # Range for slope in winch model
    "offset_winch_ro": (-6, 0),  # Range for offset in winch model
    "slope_winch_ri": (0, 10000),  # Range for slope in winch model
    "offset_winch_ri": (-6, -1),  # Range for offset in winch model
    # "max_tether_force": (20000, 50000),  # Range for max tether force in winch model
    "end_angle": (0.6, 30),
    "elevation_start_riro": (np.radians(30), np.radians(110)),
    "height": (50, 400),
    "r0": (180, 300),
    # d(u_s)/dt [1/s]; 2019 V3 flight max slew was ~0.29 (~29 kcu/s).
    "steering_rate": (-0.29, 0.29),
    # input_depower is the absolute power-tape length l_dp [m] (new convention).
    # Spans the physical depower range (legacy u_p in [-1, 1] ~ l_dp [1.20, 2.13] m).
    "input_depower": (1.1, 2.3),
    # d(l_dp)/dt [m/s]; KCU power-tape actuation speed limit, applied when the
    # depower profile is optimized per node (sim_parameters.optimize_depower_profile).
    # Placeholder physical value -- override via sim_parameters["depower_rate"].
    "depower_rate": (-0.2, 0.2),
    "speed_tangential": (10, 400),
    "angle_of_attack": (np.radians(0), np.radians(14)),
    "C_phi": (-0.8, 0.8),  # Range for C_phi in Fourier pattern
    "C_beta": (0.01, 0.9),  # Range for C_beta in Fourier pattern
}


DEFAULT_REELIN_PATTERN_CONFIG = {
    "n_points": 100,
    "time_step": 0.1,
    "reeling_speed": -5,
    "r0": 300,
    "r1": 200,
    "input_depower": 1,
}


DEFAULT_RADIAL_PARAMETERS = {
    "reeling_strategy": "force",
    "force_model": "quadratic",
    "reeling_speed": 1.0,
    "max_tether_force": 2e4,
    "min_tether_force": 5000.0,
    "softplus": True,
    "softplus_beta": 1e-4,
    "softminus": True,
    "softminus_beta": 1e-3,
    "slope": 2716,
    "offset": -3,
}
