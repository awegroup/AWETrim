import json
import numpy as np
from picawe import Cycle

# -------------------- Load Aero Input --------------------
with open("./data/V11/v11_aero_input.json", "r") as file:
    aero_input = json.load(file)

with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
    aero_input = json.load(file)

# -------------------- Simulation Config --------------------
SIMULATION_CONFIG = {
    "dof": 3,
    "area_wing": 47,
    "mass_wing": 78,
    "mass_kcu": 0,
    "tether_diameter": 0.014,
    "wind_model": "logarithmic",
    "speed_friction": 0.45,
    "z0": 0.02,
    "steering_control": "roll",
}

# -------------------- Pattern Config --------------------
PATTERN_CONFIG = {
    "pattern_type": "figure_eight",
    "parameters": {
        "omega": -1.0,
        "r0": 210.0,
        "ry": 120,
        "rz": 94,
        "ky": 0.7,
        "kz": 0.7,
        "vr": 1.5,
        "beta0": 0.6775,
        "kappa": 0,
    },
    "control": {
        "input_depower": 0.0,
    },
    "start_path_angle": -np.pi / 4,
    "end_path_angle": np.pi / 4 + 2 * np.pi + np.pi / 8,
    "n_points": 400,
    "quasi_steady": True,
}

CYCLE_SETTINGS = {
    "reelout": PATTERN_CONFIG,
    "reelin": {
        "quasi_steady": False,
        "control": {
            "max_elevation": np.radians(100),
            "min_elevation": np.radians(25),
            "reeling_speed": -3,
            "min_tether_force": SIMULATION_CONFIG["mass_wing"] * 9.81,
            "length_tether_ro": PATTERN_CONFIG["parameters"]["r0"],
            "ri_elevation": np.radians(40),  # Initial elevation for reeling in
        },
        "initial_state": {
            "angle_course": 0,
            "input_steering": 0,
            "input_depower": 0,
            "speed_tangential": 60,
            "timeder_angle_course": 0,
            "tension_tether_ground": 1e6,
        },
        "time_step": 0.1,
        "quasi_steady": True,
    },
}

# -------------------- Run Cycle --------------------
wind_speed = (
    SIMULATION_CONFIG["speed_friction"] / 0.4 * np.log(100 / SIMULATION_CONFIG["z0"])
)
print("Wind speed at 100m:", wind_speed)


reeling_speeds = [-1, -2, -3, -4, -5, -6, -7]  # m/s (negative means reeling in)
mean_powers = []
for rs in reeling_speeds:
    CYCLE_SETTINGS["reelin"]["control"]["reeling_speed"] = rs

    print(f"\nRunning simulation with reeling_speed = {rs} m/s")
    cycle_sim = Cycle(aero_input, SIMULATION_CONFIG)
    reelout_phase, reelin_phase = cycle_sim.run_cycle(CYCLE_SETTINGS)

    t_ro = reelout_phase.return_variable("t")
    pow_ro = reelout_phase.return_variable("mechanical_power")
    dt_ro = np.diff(t_ro, prepend=t_ro[0])
    energy_ro = np.sum(pow_ro * dt_ro)

    t_ri = reelin_phase.return_variable("t")
    pow_ri = reelin_phase.return_variable("mechanical_power")
    dt_ri = np.diff(t_ri, prepend=t_ri[0])
    energy_ri = np.sum(pow_ri * dt_ri)

    t_total = np.concatenate([t_ro, t_ri])
    total_energy = energy_ro + energy_ri
    total_duration = t_total[-1] - t_total[0]
    mean_power = total_energy / total_duration

    mean_powers.append(mean_power / 1000)  # convert to kW

# Plot result
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(reeling_speeds, mean_powers, marker="o")
plt.xlabel("Reeling-in speed [m/s]")
plt.ylabel("Mean Cycle Power [kW]")
plt.title("Impact of Reeling Speed on Mean Power")
plt.grid(True)
plt.tight_layout()
plt.show()
