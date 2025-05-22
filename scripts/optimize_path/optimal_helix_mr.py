import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.system.kite import Kite
from picawe.kinematics.Kinematics import ParametrizedKinematics, KiteKinematics
from picawe import SystemModel
from picawe.utils.color_palette import set_plot_style, get_color_list
from picawe.timeseries.phase_parametrized import PhaseParameterized
import json
from picawe.system.tether import RigidLumpedTether




# Define aerodynamic input
file_path = "./data/ap2_aero_input.json"
file_path = "./data/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)


# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------
pattern_config = {
    "pattern_type": "helix",
    "initial_parameters": {
        "omega": -1.0,
        "r0": 200.0,
        "d0": 80.0,
        "vr": 0.2,
        "beta": 0.35,
        "kappa": 0
    },
    "optimization_parameters": {
        # Add any optimization-related parameters here if needed as list of names
        "d0",
        # "kappa",
        # "beta",
    }
}

start_state = {
    "t": 0,
    "s": -np.pi/2,
    "s_dot": 2,
    "s_ddot": 0,
    "length_tether": 199.5,
    "input_steering": 0,
    "tension_tether_ground": 1e5
}
time = np.arange(0, 50, 0.1)
s_array = np.linspace(np.pi/2, 9*np.pi/2, 200)
dof = 3
# -----------------------------------------------
# Define the system and aerodynamic model
# -----------------------------------------------
colors = get_color_list()
tension_tether_results = {}
phases = {}
parameters = ["speed_tangential", "tension_tether_ground", "angle_roll"]
x_param = "s"
# fig, axs = plt.subplots(len(parameters),1, figsize=(10, 4), sharex=True)
mass_ratio_values = np.linspace(0, 5, 4)
mass_ratio_values = [2]

area_wing = 20

for i,mr in enumerate(mass_ratio_values):
    pattern_config = {
        "pattern_type": "helix",
        "initial_parameters": {
            "omega": -1.0,
            "r0": 200.0,
            "d0": 103.0,
            "vr":  2,
            "beta": 0.35,
            "kappa": 0
        },
        "optimization_parameters": {
            # Add any optimization-related parameters here if needed as list of names
            "d0",
            # "kappa",
            # "beta",
        }
    }
    pattern_config = {
        "pattern_type": "figure_eight",
        "parameters": {
            "omega": -1.0,
            "r0": 200.0,
            "ry": 60,
            "rz": 60,
            "ky": 0.78,
            "kz": .93,
            "vr": 3.5,
            "beta0": 0.35,
            "kappa": 0
        },
        "start_path_angle": -np.pi/2,
        "end_path_angle": 3*np.pi/2 + np.pi,
        "n_points": 200,
        "optimization_parameters": {
            # Add any optimization-related parameters here if needed as list of names
            "ry",
            "rz",
            "ky",
            "kz",
            # "kappa",
            "beta0",
            # "vr",
        }
    }
    for quasi_steady in [True,False]:  # Loop over both dynamic and quasi-steady cases
        if quasi_steady:
            linestyle = "--"
            label = None
        else:
            linestyle = "-"
            label = r"$\frac{m}{S}=$"+str(mr)
        mass_wing =  mr * area_wing
        tether = RigidLumpedTether()
        # Define kite model with current parameters
        kite = Kite(mass_wing=15, area_wing=area_wing, aero_input=aero_input, mass_kcu=28, steering_control="asymmetric")
        kite_model = SystemModel(dof=dof, quasi_steady=quasi_steady, kite=kite, wind_model="logarithmic", tether = tether)
        kite_model.wind.speed_friction = 0.6
        # kite_model.wind.speed_wind_ref = 12
        kite_model.input_depower = 0
        phase = PhaseParameterized(kite_model, quasi_steady=quasi_steady, pattern_config=pattern_config)

        # # Run simulation
        # if mr == 0 and not quasi_steady:
        #     pass
        # else:
            # phase = PhaseParameterized(kite_model, quasi_steady=quasi_steady, pattern_config=pattern_config)
            # phase.set_optimal_speed_radial()

        if quasi_steady:
            phase.optimize_pattern(start_state=start_state)
            # start_state = phase.states[0]
            pattern_config = phase.pattern_config

        # kite = Kite(mass_wing=mass_wing, area_wing=area_wing, aero_input=aero_input, mass_kcu=0, steering_control="asymmetric")
        # kite_model = SystemModel(dof=dof, quasi_steady=quasi_steady, kite=kite, wind_model="uniform")#, tether = tether)
        # kite_model.wind.speed_wind_ref = 15
        # kite_model.input_depower = 0
        # phase.kite_model = kite_model
        print(quasi_steady)
        # phase = PhaseParameterized(kite_model, quasi_steady=quasi_steady, pattern_config=pattern_config)
        # phase.set_optimal_angle_pitch_tether()
        # phase.run_simulation(start_state=start_state, s_array=s_array)
        phase.run_simulation(start_state=start_state)
        start_state = phase.states[0]
        # Extract variables
        s = phase.return_variable("s")
        s_dot = phase.return_variable("s_dot")
        aoa = phase.return_variable("angle_of_attack")
        vr = phase.return_variable("speed_radial")

        print(np.mean(aoa)*180/np.pi)
        phases[(mr, quasi_steady)] = phase
        print(s_dot[0])
        # start_state["s_dot"] = s_dot[0]


