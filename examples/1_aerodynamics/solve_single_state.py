"""Solve one VSM aerodynamic trim state.

Edit the Inputs block and run from the project root::

    python examples/1_aerodynamics/solve_single_state.py

The kite is the one ``data/<KITE>`` describes (``common.load_kite``): its
wing, bridle, mass and model choices come from the data files, the operating
point from ``CASE``. ``SHAPE`` picks the wing: ``"built"`` (structure.yml with
as_config's billowing), or the deformed shape a step-2 run of the same CASE
left (``"wireframe"`` / ``"billow"``); ``DEFORMED_FROM`` points at any other
results case folder.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from common import (
    load_kite,
    output_dir,
    print_kite,
    print_trim_summary,
    save_figure,
    trim,
    write_json,
)
from awetrim.aerostructural.case import Case
from awetrim.plotting.kite_tether import draw_kite_tether, tether_info_str

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
KITE = "LEI-V3-KITE"  # a folder under data/

#: The step-2 PSM example's case (examples/2_aerostructural/run_simulation_PSM.py),
#: so SHAPE = "wireframe" finds that example's result.
CASE = Case(
    elevation_deg=0.0,  # beta
    azimuth_deg=0.0,  # phi
    course_deg=90.0,  # chi
    distance_radial=300.0,  # r [m]
    speed_radial=1.5,  # v_r [m/s], positive = reeling out
    timeder_speed_tangential=0.0,  # steady flight
    timeder_speed_radial=0.0,
    wind={"model": "uniform", "speed_wind_ref": 6.0},
    # [m]; the built shape is at 0: a tape command needs a deformed SHAPE
    power_tape_extension=0.0,
    steering_tape_extension=0.0,
)
SHAPE = "built"  # "built" | "wireframe" | "billow"
DEFORMED_FROM = None  # a results case folder; overrides SHAPE
WITH_GRAVITY = True

OUTPUT_DIR = None  # default results/aerodynamics/single_state
SHOW = True


def main() -> None:
    out_dir = output_dir("single_state", OUTPUT_DIR)
    kite = load_kite(
        KITE,
        CASE,
        shape=SHAPE,
        deformed_from=DEFORMED_FROM,
        overrides={"coupling": {"is_with_gravity": WITH_GRAVITY}},
    )
    print_kite(kite)

    result, _ = trim(kite)
    print_trim_summary(result)

    williams = "williams_tether_length" in result
    if williams:
        print("--- Williams tether ---")
        print(f"  elevation_last [deg] : {result['williams_elevation_last_deg']:.4f}")
        print(f"  azimuth_last   [deg] : {result['williams_azimuth_last_deg']:.4f}")
        print(f"  tether_length  [m]   : {result['williams_tether_length']:.4f}")
        print(f"  ground residual [m]  : {result['williams_ground_residual']}")
        print(
            f"  |F_kite_resultant|   : {np.linalg.norm(result['force_kite_resultant']):.3f} N"
        )

    write_json(
        out_dir / "trim_result.json",
        {"case": CASE.to_config(), "kite": kite.properties(), "trim": result},
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["cmx", "cmy", "cmz", "cfx", "cfy"]
    values_plot = np.r_[
        np.asarray(result["cm"], dtype=float), result["cfx"], result["cfy"]
    ]
    ax.bar(
        labels,
        values_plot,
        color=["#4C78A8", "#4C78A8", "#4C78A8", "#F58518", "#F58518"],
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Residual coefficient [-]")
    ax.set_title("VSM aerodynamic trim residuals")
    fig.tight_layout()
    save_figure(fig, out_dir / "trim_residuals.pdf")

    if williams:
        fig2 = plt.figure(figsize=(7, 6))
        ax2 = fig2.add_subplot(111, projection="3d")
        draw_kite_tether(ax2, result, kite.system_model, kite.body, str(kite.structure_path))
        ax2.set_title(
            f"Williams tether shape (kite -> ground)\n"
            f"[{tether_info_str(kite.system_model)}]"
        )
        ax2.legend()
        fig2.tight_layout()
        save_figure(fig2, out_dir / "williams_tether_shape.pdf")

    if SHOW:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
