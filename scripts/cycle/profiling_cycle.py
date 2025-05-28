
import json
import numpy as np
from picawe import Cycle

# -------------------- Load Aero Input --------------------
with open("./data/v9_aero_input.json", "r") as file:
    aero_input = json.load(file)

# -------------------- Simulation Config --------------------
SIMULATION_CONFIG = {
    "mass_ratio": 2,
    "dof": 3,
    "area_wing": 47,
    "mass_wing": 78,
    "tether_diameter": 0.014,
    "quasi_steady": True,
    "wind_model": "logarithmic",
    "speed_friction": 0.4,
    "z0": 0.01,
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
        "ky": 1,
        "kz": 1,
        "vr": 1.5,
        "beta0": 0.6775,
        "kappa": 1,
    },
    "control": {
        "input_depower": 0.0,
    },
    "start_path_angle": -np.pi/2,
    "end_path_angle": 3*np.pi/2 + np.pi,
    "n_points": 200,
}

CYCLE_SETTINGS = {
    "reelout": PATTERN_CONFIG,
    "reelin": {
        "control": {
            "max_elevation": np.degrees(85),
            "reeling_speed": -3.5,
            "min_tether_force": SIMULATION_CONFIG["mass_wing"] * 9.81,
            "length_tether_ro": PATTERN_CONFIG["parameters"]["r0"],
        },
        "initial_state": {
            "angle_course": 0,
            "input_steering": 0,
            "input_depower": 0,
            "speed_tangential": 40,
            "timeder_angle_course": 0,
            "tension_tether_ground": 1e4,
        },
        "time_step": 0.1
    }
}

if __name__ == "__main__":
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    cycle_sim = Cycle(aero_input, SIMULATION_CONFIG)
    reelout_phase, reelin_phase = cycle_sim.run_cycle(CYCLE_SETTINGS)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.dump_stats("cycle_profile.prof")
    print("Profiling complete. Use snakeviz or another viewer to analyze 'cycle_profile.prof'")
