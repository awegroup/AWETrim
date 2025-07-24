import time
from picawe.timeseries.phase_parametrized import PhaseParameterized
from picawe.timeseries.reelin_phase import ReelinPhase
from picawe.system.kite import Kite
from picawe.system.tether import RigidLumpedTether
from picawe import SystemModel, State
from picawe.environment.Wind import Wind


class Cycle:
    def __init__(self, aero_input, sim_config):
        self.aero_input = aero_input
        self.sim_config = sim_config
        if sim_config["wind_model"] == "logarithmic":
            self.wind_model = Wind(
                wind_model=sim_config["wind_model"], z0=sim_config["z0"]
            )
            self.wind_model.speed_friction = sim_config["speed_friction"]
        elif sim_config["wind_model"] == "uniform":
            self.wind_model = Wind(wind_model=sim_config["wind_model"])
            self.wind_model._speed_wind_ref = sim_config["speed_wind_ref"]
        elif sim_config["wind_model"] == "tabulated":
            self.wind_model = Wind(
                wind_model=sim_config["wind_model"],
                tabulated_heights=sim_config["tabulated_heights"],
                tabulated_speeds=sim_config["tabulated_speeds"],
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
        if self.sim_config["wind_model"] == "logarithmic":
            model.wind.z0 = self.sim_config["z0"]
            model.wind.speed_friction = self.sim_config["speed_friction"]
        elif self.sim_config["wind_model"] == "uniform":
            model.wind.speed_wind_ref = self.sim_config["speed_wind_ref"]
        # model.wind.speed_friction = self.sim_config["speed_friction"]
        return model

    def run_cycle(self, cycle_settings):
        pattern_config = cycle_settings["reelout"]
        model_ro = self.create_model()
        print(cycle_settings["reelout"])
        phase_ro = PhaseParameterized(
            model_ro,
            quasi_steady=cycle_settings["reelout"]["quasi_steady"],
            pattern_config=pattern_config,
        )
        print("Running reelout...")
        t0 = time.time()
        phase_ro.run_simulation()
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
