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

import time
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim import SystemModel, State
from awetrim.system.factory import create_wind_model_from_config
import numpy as np


class Cycle:
    def __init__(self, aero_input, sim_config):
        self.aero_input = aero_input
        self.sim_config = sim_config
        # Any analytic/tabulated profile; the amplitude keys
        # (speed_friction / speed_wind_ref) are applied by the YAML factory.
        self.wind_model = create_wind_model_from_config(
            {
                **sim_config,
                "model": sim_config["wind_model"],
                # Cycle historically left the direction as a free symbol.
                "direction_wind": sim_config.get("direction_wind"),
            }
        )
        self.kite = Kite(
            mass_wing=sim_config["mass_wing"],
            area_wing=sim_config["area_wing"],
            mass_kcu=self.sim_config.get("mass_kcu", 0),
            aero_input=aero_input,
            steering_control=sim_config.get("steering_control", "roll"),
        )
        self.tether = RigidLumpedTether(diameter=sim_config["tether_diameter"])

    def create_model(self, quasi_steady=True):
        model = SystemModel(
            dof=self.sim_config["dof"],
            quasi_steady=quasi_steady,
            kite=self.kite,
            wind_model=self.wind_model,
            tether=self.tether,
        )
        # Re-apply the configured amplitude on the (possibly copied) model.
        if "speed_friction" in self.sim_config:
            model.wind.z0 = self.sim_config.get("z0", model.wind.z0)
            model.wind.speed_friction = self.sim_config["speed_friction"]
        elif "speed_wind_ref" in self.sim_config:
            model.wind.speed_wind_ref = self.sim_config["speed_wind_ref"]
        return model

    def run_cycle(self, cycle_settings):
        pattern_config = cycle_settings["reelout"]
        model_ro = self.create_model()
        model_ro.input_depower = 0
        print(cycle_settings["reelout"])
        phase_ro = PhaseParameterized(
            model_ro,
            quasi_steady=cycle_settings["reelout"]["quasi_steady"],
            pattern_config=pattern_config,
        )
        print("Running reelout...")
        t0 = time.time()
        base_start_state = State(
            t=0,
            s=-np.pi / 4,
            s_dot=2,
            s_ddot=0,
            input_steering=0,
            tension_tether_ground=1e8,
        )

        phase_ro.run_simulation(start_state=base_start_state)
        print("Reelout time:", time.time() - t0, "seconds")

        model_ri = self.create_model(
            quasi_steady=cycle_settings["reelin"]["quasi_steady"]
        )
        phase_ri = ReelinPhase(model_ri)

        init = cycle_settings["reelin"]["initial_state"]
        start_state_ri = State(
            t=phase_ro.return_variable("t")[-1],
            distance_radial=phase_ro.return_variable("distance_radial")[-1],
            angle_elevation=phase_ro.return_variable("angle_elevation")[-1],
            angle_azimuth=phase_ro.return_variable("angle_azimuth")[-1],
            angle_course=phase_ro.return_variable("angle_course")[-1],
            input_steering=phase_ro.return_variable("input_steering")[-1],
            input_depower=phase_ro.return_variable("input_depower")[-1],
            speed_tangential=phase_ro.return_variable("speed_tangential")[-1],
            timeder_angle_course=phase_ro.return_variable("timeder_angle_course")[-1],
            speed_radial=phase_ro.return_variable("speed_radial")[-1],
            tension_tether_ground=phase_ro.return_variable("tension_tether_ground")[-1],
            timeder_speed_tangential=phase_ro.return_variable(
                "timeder_speed_tangential"
            )[-1],
        )

        cycle_settings["reelin"]["control"]["riro_elevation"] = (
            phase_ro.return_variable("angle_elevation")[0]
        )
        cycle_settings["reelin"]["control"]["riro_azimuth"] = phase_ro.return_variable(
            "angle_azimuth"
        )[0]

        print("Running reelin...")
        t0 = time.time()
        phase_ri.run_simulation(
            start_state=start_state_ri, settings=cycle_settings["reelin"]
        )
        print("Reelin time:", time.time() - t0, "seconds")

        return phase_ro, phase_ri
