# Step 5: flight data

Optional: when the kite has flown, use the flight to correct the ROM from
step 3 and to validate it. AWETrim does not process raw flight logs itself.
It reads the flight as reconstructed by the extended Kalman filter in
[EKF-AWE](https://github.com/awegroup/EKF-AWE), one `.h5` file per flight.

For the LEI-V3, EKF-AWE ships the raw logs. Run it on the 2019-10-08 flight
and place the output at `results/LEI-V3-KITE/ekf/LEI-V3 Kite_2019-10-08.h5`.

| Script | What it does | Writes |
|--------|--------------|--------|
| [`identify_rom_flight_correction.py`](identify_rom_flight_correction.py) (**not yet reviewed**: the flight correction is provisional) | Identifies, on the ROM's own solve with the logged steering and on the training cycles, the model tapes that reproduce the flown depower (`flight_match`, used by both ROMs) and one scale on the roll gain that matches the flown turn rate. Lift, drag and `theta_b` stay simulated. | `data/LEI-V3-KITE/identification.yaml` (`flight_match`), `rom_config_aerostructural_flight_corrected.yaml`, `results/LEI-V3-KITE/identification/rom_flight_correction/` |
| [`validate_quasi_steady_state_v3.py`](validate_quasi_steady_state_v3.py) | Takes the measured kinematics and wind as given and solves the ROM's quasi-steady force balance for tether force, steering input and tangential speed; compares predicted and measured. `--rom` picks the ROM file, `--tether rigid` or `williams`. | `results/<kite>/rom_validation/` |
| [`validate_spline_v3.py`](validate_spline_v3.py) | Fits B-spline patterns to the measured trajectory cycle by cycle, simulates them with the ROM and compares against the flight. | |

The validators reproduce a specific flight, so they use the **as-flown** KCU
mass (`data/LEI-V3-KITE/system_flown_2019.yaml`), not the optimisation
configuration.
