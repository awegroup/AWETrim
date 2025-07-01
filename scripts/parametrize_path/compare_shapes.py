import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools
from picawe.kinematics.parametrized_patterns import Helix
from picawe import SystemModel, State
from picawe.utils.color_palette import set_plot_style, get_color_list
from picawe.timeseries.phase_parametrized import PhaseParameterized
from picawe.system.kite import Kite
from picawe.system.tether import FlexibleLumpedTether
from picawe.utils.defaults import PLOT_LABELS


def load_aero_input(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def create_model_and_phase(kite, tether, pattern_config, start_state):

    model = SystemModel(dof=3, quasi_steady=True, kite=kite, tether=tether)
    model.wind.speed_wind_ref = 10
    model.input_depower = 0
    model.speed_radial = 0

    phase = PhaseParameterized(model, quasi_steady=True, pattern_config=pattern_config)
    phase.run_simulation(start_state=start_state)
    return phase


def plot_results(phases):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_color_list()

    for i, ((ry, rz, ky, kz), phase) in enumerate(phases.items()):
        vtau = phase.return_variable("speed_tangential")
        azimuth = np.degrees(phase.return_variable("angle_azimuth"))
        elevation = np.degrees(phase.return_variable("angle_elevation"))
        ax.scatter(
            azimuth, elevation, c=vtau, cmap="viridis", s=10, label=f"ky={ky}, kz={kz}"
        )

    ax.set_xlabel(PLOT_LABELS["angle_azimuth"])
    ax.set_ylabel(PLOT_LABELS["angle_elevation"])
    ax.legend()
    set_plot_style()
    plt.tight_layout()
    # plt.savefig("./results/figures/translational_paper/figure8_scatter_comparison.pdf", bbox_inches='tight')
    plt.show()


def simulate_parameter_grid():
    file_path = "./data/LEI-V3-KITE/v3_aero_input.json"
    aero_input = load_aero_input(file_path)

    start_state = State(
        t=0,
        s=np.pi / 2,
        s_dot=2,
        s_ddot=0,
        length_tether=200,
        input_steering=0,
        angle_roll=0,
        angle_pitch=0,
        angle_yaw=0,
    )

    mass_ratio = 2
    area_wing = 20
    mass_wing = mass_ratio * area_wing

    tether = FlexibleLumpedTether()
    kite = Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        steering_control="asymmetric",
    )

    ry_vals = [80]
    rz_vals = [80]
    ky_vals = [0.2, 0.8]
    kz_vals = [0.2, 0.8]
    phases = {}

    for ry, rz, ky, kz in itertools.product(ry_vals, rz_vals, ky_vals, kz_vals):
        pattern_config = {
            "pattern_type": "figure_eight",
            "parameters": {
                "omega": -1.0,
                "r0": 200.0,
                "ry": ry,
                "rz": rz,
                "ky": ky,
                "kz": kz,
                "vr": 1,
                "beta0": 0.45,
                "kappa": 0,
            },
            "start_path_angle": -np.pi / 2,
            "end_path_angle": np.pi / 2 + 5 * np.pi,
            "n_points": 400,
        }

        phase = create_model_and_phase(kite, tether, pattern_config, start_state)
        phases[(ry, rz, ky, kz)] = phase

    plot_results(phases)


if __name__ == "__main__":
    simulate_parameter_grid()
