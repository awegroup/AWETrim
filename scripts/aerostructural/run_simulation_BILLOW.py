"""Run one coupled aerostructural simulation of the kite with the FULL Billow model.

The aerodynamics (Vortex Step Method) and the structure are iterated to a
converged deformed shape, trimmed at the flight state set below. The structure
is Billow's full fidelity: bridle lines and pulleys, plus inflatable Timoshenko tube beams (leading edge and struts) and a wrinkling
membrane canopy, read from struc_geometry_FEM_full.yaml -- the same file
run_simulation_PSM.py reduces to a wireframe, so with the same inputs the two
scripts differ only in the structural model. About a minute.

Edit the inputs below, then from the project root:

    python scripts/aerostructural/run_simulation_BILLOW.py

The case is written to results/<KITE>/aerostructural/billow/<case>/ and, with
SHOW_RESULT on, opened in the 3-D viewer with its trim characteristics (or
afterwards: python scripts/aerostructural/plot_simulation.py <case folder>).
Solver settings come from data/<KITE>/as_config.yaml (block structural_billow).
"""

from pathlib import Path

from awetrim.aerostructural.billow.driver import solve_deformation
from awetrim.aerostructural.case import build_flight_case_folder
from awetrim.aerostructural.results import aerostructural_results_root

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
KITE = "LEI-V3-KITE"  # a folder under data/ with a struc_geometry_FEM_full.yaml
SYSTEM_FILE = "system.yaml"  # in data/<KITE>/: masses (KCU) and tether

# Flight state. Angles are those of the course-aligned spherical frame: the
# tether direction is set by elevation and azimuth, the flight direction on the
# sphere by the course angle (90 deg = flying across the wind window).
ELEVATION_DEG = 0.0  # beta, tether above the ground
AZIMUTH_DEG = 0.0  # phi, tether away from the downwind direction
COURSE_DEG = 90.0  # chi
TETHER_LENGTH_M = 300.0  # r
REEL_OUT_SPEED_MS = 1.5  # v_r, positive = reeling out
# At the reference height of the kite's wind model. 6 m/s puts the LEI-V3 at
# an apparent wind of about 22 m/s and a tether force above 4000 N at the
# default state: inside the load range the Billow tube law is calibrated on,
# and about twice the load below which the centre-bay trailing edge of the
# Billow canopy loses its symmetric equilibrium (an unsteered case then
# converges asymmetric). Keep it equal in both scripts to compare.
WIND_SPEED_MS = 6.0
WITH_GRAVITY = False

# Actuation [m], relative to the tape lengths stored in the geometry. Stepped
# in from the stored lengths, one converged state per step.
DEPOWER_TAPE_EXTENSION_M = 0.0  # + lengthens the depower tape
STEERING_TAPE_EXTENSION_M = 0.0  # half-difference: + shortens one steering tape, lengthens the other

SHOW_RESULT = True  # print the trim table and open the 3-D viewer when done
# ---------------------------------------------------------------------------


def main():
    project_dir = Path(__file__).resolve().parents[2]
    overrides = {
        "angle_elevation_deg": ELEVATION_DEG,
        "angle_azimuth_deg": AZIMUTH_DEG,
        "angle_course_deg": COURSE_DEG,
        "distance_radial": TETHER_LENGTH_M,
        "speed_radial": REEL_OUT_SPEED_MS,
        "wind_speed_wind_ref": WIND_SPEED_MS,
        "is_with_gravity": WITH_GRAVITY,
        "power_tape_final_extension": DEPOWER_TAPE_EXTENSION_M,
        "steering_tape_final_extension": STEERING_TAPE_EXTENSION_M,
        # The same aerodynamics in both scripts: 27 sections x 2 = 54 panels
        # of the full geometry's mesh, and VSM's base circulation loop. With
        # artificial viscosity on, the LEI-V3's stalled tips trap the Anderson
        # loop (as_config's) in a post-stall limit cycle and the tip loads
        # never converge. The Billow studies use base too.
        "aerodynamic": {"n_aero_panels_per_struc_section": 2, "gamma_loop_type": "base"},
    }
    results_dir = (
        aerostructural_results_root(project_dir, KITE)
        / "billow"
        / build_flight_case_folder(overrides)
    )
    result = solve_deformation(
        config_overrides=overrides,
        kite_name=KITE,
        project_dir=project_dir,
        results_dir=results_dir,
        system_config_path=project_dir / "data" / KITE / SYSTEM_FILE,
    )
    print(f"\nCase written to {result['results_dir']}")
    if SHOW_RESULT:
        from plot_simulation import report

        report(result["results_dir"])


if __name__ == "__main__":
    main()
