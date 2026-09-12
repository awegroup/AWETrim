import os.path
import time as time
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

# Todo: this python path update is added because of the import from data.process_KP_data. We could move process_KP_data
#  to the examples dir to avoid this.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from awes_ekf.ekf.initialize_and_update_ekf import initialize_ekf, propagate_state_EKF
from awes_ekf.load_data.read_data import read_processed_flight_data
from awes_ekf.load_data.create_input_from_csv import (
    create_input_from_csv,
    find_initial_state_vector,
)
from awetrim.experimental.settings import (
    load_config,
    save_ekf_results,
    SimulationConfig,
    TuningParameters,
)
from awes_ekf.setup.kite import PointMassEKF
from awes_ekf.setup.tether import Tether
from awes_ekf.setup.kcu import KCU
from awes_ekf.ekf.ekf_output import convert_ekf_output_to_df
from awes_ekf.postprocess.postprocessing import postprocess_results
import importlib


def list_available_flights(log_directory: Path, file_extension: str = ".csv") -> list:
    """
    Lists all available flight log files in the specified directory.

    :param log_directory: Path to the directory containing flight logs.
    :param file_extension: Extension of the log files (default is ".csv").
    :return: List of available flight log files.
    """
    flight_logs = list(log_directory.glob(f"*{file_extension}"))

    if not flight_logs:
        print(f"No flight logs found in {log_directory}.")
        return []

    print(f"Available flight logs in {log_directory}:")
    [print(f"{i + 1}: {log_file.name}") for i, log_file in enumerate(flight_logs)]

    return flight_logs


class AnalyzeAweFromCsvLog:

    def __init__(self, config_data: dict, date: datetime, log_directory: Path):

        self.config_data = config_data
        self.update_configuration_for_execution(date.strftime("%Y-%m-%d"))

        self.log_directory = log_directory

        self.run()

    def update_configuration_for_execution(self, date: str) -> None:
        self.config_data["year"], self.config_data["month"], self.config_data["day"] = (
            date.split("-")
        )

    def pre_process_log(self) -> None:
        # Directory where data transformer scripts are stored
        transformer_directory = "src/awetrim/experimental/data_preprocessors"

        # List all Python files in the transformers directory
        transformer_files = [
            f
            for f in os.listdir(transformer_directory)
            if f.endswith(".py") and f.startswith("process_")
        ]

        # Prompt user to select a transformer script
        print("Available data pre-process scripts:")
        for index, filename in enumerate(transformer_files, start=1):
            print(f"{index}: {filename}")

        # Get user selection
        selection = int(input("Select a pre-process script by number: ")) - 1

        # Ensure selection is valid
        if 0 <= selection < len(transformer_files):
            selected_file = transformer_files[selection]
            module_name = selected_file[:-3]  # Remove '.py' extension for import

            # Dynamically import the selected module
            module = importlib.import_module(f"awetrim.experimental.data_preprocessors.{module_name}")

            # Call the transform_data function from the selected module
            if hasattr(module, "process_data"):
                module.process_data(
                    config_data=self.config_data, log_directory=self.log_directory
                )
            else:
                raise AttributeError(
                    f"The module {module_name} does not contain a 'process_data' function."
                )

            print(f"Data pre-processed using: {selected_file}")
        else:
            raise ValueError("Invalid selection. Please choose a valid file number.")

    def filter_by_time(self, start_minute: int, end_minute: int) -> None:
        """Filter the flight data by the given start and end minute."""
        # TODO: Frequency of the data is 10Hz, hardcoded
        if start_minute > 0 or end_minute > 0:
            if end_minute > 0 and end_minute <= len(self.flight_data / 600):
                self.flight_data = self.flight_data.iloc[
                    start_minute * 600 : end_minute * 600
                ].reset_index(drop=True)
            else:
                self.flight_data = self.flight_data.iloc[
                    start_minute * 600 :
                ].reset_index(drop=True)
            print(f"Filtered data from minute {start_minute} to {end_minute}.")
        else:
            print("No time filtering applied.")

    def run_analysis(self) -> None:
        # Todo: This function could be split into multiple to mserve as a clearer documentation of how the ekf and
        #  simulation have to be used
        year = str(self.config_data["year"])
        month = str(self.config_data["month"])
        day = str(self.config_data["day"])
        kite_model = self.config_data["kite"]["model_name"]

        flight_data = read_processed_flight_data(year, month, day, kite_model)
        duration = flight_data["time"].max() - flight_data["time"].min()
        duration = duration / 60  # in minutes
        print(f"Duration of the flight: {duration:.2f} minutes.")
        start_minute = int(
            input("Enter the start minute for analysis or skip: ").strip() or 0
        )
        end_minute = int(
            input("Enter the end minute for analysis or skip: ").strip() or 0
        )
        dt = flight_data["time"].diff().mean()

        if start_minute > 0 or end_minute > 0:
            if end_minute > 0 and end_minute <= len(flight_data) / 60 * dt:
                flight_data = flight_data.iloc[
                    int(start_minute * 60 / dt) : int(end_minute * 60 / dt)
                ].reset_index(drop=True)
            else:
                flight_data = flight_data.iloc[
                    int(start_minute * 60 / dt) :
                ].reset_index(drop=True)
            print(f"Filtered data from minute {start_minute} to {end_minute}.")
        else:
            print("No time filtering applied.")

        # %% Initialize EKF
        sim_parameters = dict(self.config_data["simulation_parameters"])
        by_date = sim_parameters.pop("initial_wind_velocity_by_date", None) or {}
        coeffs_by_date = (
            sim_parameters.pop("initial_coefficients_by_date", None) or {}
        )
        params_by_date = (
            sim_parameters.pop("simulation_parameters_by_date", None) or {}
        )
        flight_date = f"{year}-{month}-{day}"
        # Per-flight overrides of arbitrary simulation_parameters keys (e.g.
        # the 2017 flights disable the apparent-windspeed measurement: no
        # pitot flew). Dict values merge one level deep so a measurements
        # override only touches the keys it names.
        overrides = params_by_date.get(flight_date) or {}
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(sim_parameters.get(key), dict):
                sim_parameters[key] = {**sim_parameters[key], **value}
            else:
                sim_parameters[key] = value
        if overrides:
            print(
                f"Simulation-parameter overrides for {flight_date}: {overrides} "
                "(per-flight override from ekf_config.yaml)"
            )
        # Aero-coefficient seed for find_initial_state_vector (CL/CD/CS),
        # companion of the wind seed below: the coefficient states converge in
        # seconds rather than minutes, but on a record that starts at launch
        # even that transient lands on flight data.
        initial_coeffs = dict(coeffs_by_date.get(flight_date) or {})
        if initial_coeffs:
            print(
                f"Initial coefficient seed for {flight_date}: {initial_coeffs} "
                "(per-flight override from ekf_config.yaml)"
            )
        if flight_date in by_date:
            # The filter walks its wind state in from this guess, and a guess
            # pointing the wrong way costs FLIGHT time, not spin-up time: a
            # record that starts AT launch (2025-10-09) has no lead-in to burn,
            # so the transient lands on the first pumping cycle. See the config
            # note next to the key.
            sim_parameters["initial_wind_velocity"] = list(by_date[flight_date])
            print(
                f"Initial wind velocity for {flight_date}: "
                f"{sim_parameters['initial_wind_velocity']} "
                "(per-flight override from ekf_config.yaml)"
            )
        simConfig = SimulationConfig(**sim_parameters)

        # Create system components
        kite = PointMassEKF(simConfig, **self.config_data["kite"])
        if self.config_data["kcu"]:
            kcu = KCU(**self.config_data["kcu"])
        else:
            kcu = None
        tether = Tether(kite, kcu, simConfig.obsData, **self.config_data["tether"])
        kite.calc_fx = kite.get_fx_fun(tether)

        tuningParams = TuningParameters(
            self.config_data["tuning_parameters"], simConfig
        )

        # Create input classes
        ekf_input_list = create_input_from_csv(
            flight_data, kite, kcu, tether, simConfig, kite_sensor=0
        )

        # Find initial state vector
        x0 = find_initial_state_vector(
            tether,
            ekf_input_list[0],
            simConfig,
            wind_velocity=simConfig.initial_wind_velocity,
            **initial_coeffs,
        )

        ekf, ekf_input_list = initialize_ekf(
            ekf_input_list, simConfig, tuningParams, x0, kite, kcu, tether
        )

        # %% Main loop
        ekf_output_list = []  # List of instances of EKFOutput
        start_time = time.time()
        mins = -1
        k_nis = 1000
        flight_time = 0
        import numpy as np

        # TODO: Add a timeseries class
        for k, ekf_input in enumerate(ekf_input_list):
            # if ekf_input.kite_acceleration[2]>17:
            #     ekf_input.kite_position = np.array([np.nan, np.nan, np.nan])
            #     ekf_input.kite_velocity = np.array([np.nan, np.nan, np.nan])
            #     print("Acceleration is too high to 1get good GPS data")

            try:
                ekf, ekf_ouput = propagate_state_EKF(
                    ekf, ekf_input, simConfig, tether, kite, kcu
                )
                # Store results
                ekf_output_list.append(ekf_ouput)
            except Exception as e:
                print(e)
                try:
                    print("Integration error at iteration: ", k)
                    # Re-seed with the SAME wind the run was configured with:
                    # without it the recovery silently fell back to the
                    # function default [1e-3, 8, 0] and re-injected a wind
                    # transient mid-flight (found 2026-08-27).
                    x0 = find_initial_state_vector(
                        tether,
                        ekf_input,
                        simConfig,
                        wind_velocity=simConfig.initial_wind_velocity,
                        **initial_coeffs,
                    )
                except:
                    print("Tether model error at iteration: ", k)
                    x0 = ekf.x_k1_k1
                    # continue
                ekf, ekf_input_list[k::] = initialize_ekf(
                    ekf_input_list[k::],
                    simConfig,
                    tuningParams,
                    x0,
                    kite,
                    kcu,
                    tether,
                    find_offsets=False,
                )
                flight_data.drop(k, inplace=True)
                continue

            flight_time += ekf_input.timestep
            # Print progress
            if np.round(flight_time, 2) % 60 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()  # Record end time
                mins += 1
                print(
                    f"Real time: {mins} minutes.  Elapsed time: {elapsed_time:.2f} seconds"
                )

        # Postprocess results
        ekf_output_df = convert_ekf_output_to_df(ekf_output_list)
        ekf_output_df.dropna(subset=["kite_pitch"], inplace=True)
        ekf_output_df.reset_index(drop=True, inplace=True)
        rows_to_keep = ekf_output_df.index
        print(rows_to_keep)
        flight_data = flight_data.iloc[rows_to_keep]
        flight_data.reset_index(drop=True, inplace=True)
        indices = ekf_output_df.index
        flight_data = flight_data.iloc[indices]
        ekf_output_df, flight_data = postprocess_results(
            ekf_output_df,
            flight_data,
            kite,
            kcu,
            self.config_data,
        )
        # %% Store results
        save_ekf_results(
            ekf_output_df,
            flight_data,
            kite_model,
            year,
            month,
            day,
            self.config_data,
        )

    def run(self):

        self.pre_process_log()
        self.run_analysis()


def main() -> None:
    default_log_dir = Path("./data/LEI-V3-KITE/flight_logs/")

    from prompt_toolkit import prompt
    from prompt_toolkit.completion import PathCompleter

    # Set up a PathCompleter to enable tab-completion for file paths
    path_completer = PathCompleter(expanduser=False)  # Enables ~ expansion

    # Prompt the user with tab-completion support for nested directories
    user_input = prompt(
        f"Enter the directory with the flight logs [default: {default_log_dir}]: ",
        completer=path_completer,
    ).strip()

    # Use the default if the user didn't provide input
    log_dir = Path(user_input) if user_input else default_log_dir

    # List available flights and select one
    available_flights = list_available_flights(log_dir)
    if not available_flights:
        sys.exit("No flights available to analyze.")

    selected_flight_index = (
        int(input(f"Select a flight to analyze (1-{len(available_flights)}): ")) - 1
    )
    selected_flight = available_flights[selected_flight_index]
    print(f"Selected flight: {selected_flight.name}")

    # Extract the date from the filename. The date is not always the first
    # token: 2019/2025 logs are "2019-10-08_11-36-20_V3.csv" but the 2017
    # Valkenburg logs are "DA_2017-03-30_FL01_11-30-29_12-03-06.csv".
    date = None
    for part in selected_flight.stem.split("_"):
        try:
            date = datetime.strptime(part, "%Y-%m-%d").date()
            break
        except ValueError:
            continue
    if date is None:
        sys.exit(
            f"Error: no YYYY-MM-DD date found in the filename: {selected_flight.name}"
        )

    config_data = (
        load_config()
    )  # Todo: In this function we should have a check if the config has all required data.

    # Hand the exact selection to the preprocessor: matching by date alone is
    # ambiguous when one date has several flights (2017 FL01/FL02).
    config_data["log_filename"] = selected_flight.name

    print("Starting analysis with:")
    print(f"Kite Model: {config_data['kite']['model_name']}")
    print(f"Date: {date.isoformat()}, Log file: {selected_flight.name}")
    print(f"Log directory: {log_dir}")

    AnalyzeAweFromCsvLog(config_data=config_data, date=date, log_directory=log_dir)


if __name__ == "__main__":
    main()
