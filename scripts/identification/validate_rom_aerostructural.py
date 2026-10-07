"""End-to-end check of the LEI-V3 ROMs against the aerostructural trims.

A coefficient fit can look good and still trim wrongly (theta_b, the roll
gain and the polars only meet in the force balance). Here each coupled
centre-window trim of the identification dataset is re-solved with the ROM's
own quasi-steady solver at the same condition -- azimuth 0, elevation 0,
course 90 deg, v_r = 0, the same wind, u_p and u_s, gravity off, no tether
weight or drag (the coupled trims were tetherless) -- and the ROM's tangential
speed, tether force, wing angle of attack and course rate are compared with
the coupled ones. Both ROMs are run: the semi-empirical one is not expected to
match (it is calibrated to flight, not to these simulations); it is the
reference for how far the two models are apart.

Hardware: system_flown_2019.yaml (the 22 kg KCU the trims carried).

Usage (project root; run identify_rom_aerostructural.py first)
    python scripts/identification/validate_rom_aerostructural.py
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np
import pandas as pd

from awetrim.system.factory import create_system_model_from_yaml
from awetrim.system.state import State
from awetrim.utils.config_paths import (
    LEI_V3_ROM_AEROSTRUCTURAL_CONFIG,
    LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG,
    LEI_V3_SYSTEM_FLOWN_2019_CONFIG,
)

from identify_rom_aerostructural import OUT_DIR, UP_BAND

#: Radial distance of the coupled REF state [m].
DISTANCE_RADIAL = 286.6
ROMS = {
    "aerostructural": LEI_V3_ROM_AEROSTRUCTURAL_CONFIG,
    "semi_empirical": LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG,
}


def build_model(rom_path):
    with contextlib.redirect_stdout(io.StringIO()):
        model = create_system_model_from_yaml(
            LEI_V3_SYSTEM_FLOWN_2019_CONFIG, aero_yaml_path=rom_path,
            tether_config={"model": "rigid_lumped"},
        )
    # Gravity-free and tetherless, like the coupled trims: kite.g reaches the
    # wing AND the KCU weight (the latter inside the bridle pitch), which the
    # override_gravity switch does not; the tether gets negligible weight/drag.
    model.kite.g = 0.0
    model.tether.density_tether = 1e-9
    model.tether.diameter_tether = 1e-6
    model.wind.wind_model = "uniform"
    model.wind.direction_wind = 0.0
    return model


def solve(model, row) -> dict:
    model.wind.speed_wind_ref = float(row.wind_speed)
    state = State(
        distance_radial=DISTANCE_RADIAL, angle_elevation=0.0, angle_azimuth=0.0,
        angle_course=np.pi / 2, speed_radial=0.0, timeder_speed_radial=0.0,
        timeder_speed_tangential=0.0, input_depower=float(row.u_p),
        input_steering=float(row.u_s), speed_tangential=float(row.v_tau),
        timeder_angle_course=float(row.chi_dot), tension_tether_ground=3000.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        out = model.solve_quasi_steady(state)
    if out is None:
        return {"ok": False}
    return {
        "ok": True,
        "v_tau": float(out.speed_tangential),
        "chi_dot": float(out.timeder_angle_course),
        "tension": float(out.tension_tether_ground),
        "alpha_w": float(getattr(out, "angle_of_attack", np.nan)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()
    trims = pd.read_csv(OUT_DIR / "trim_samples.csv")

    rows = []
    for name, path in ROMS.items():
        model = build_model(path)
        for row in trims.itertuples():
            res = solve(model, row)
            rows.append({"rom": name, "anchor": row.anchor, "u_p": row.u_p,
                         "u_s": row.u_s, "v_a": row.v_a, **{f"as_{k}": getattr(row, k)
                         for k in ("v_tau", "chi_dot", "tension", "alpha_w")},
                         **{f"rom_{k}": v for k, v in res.items()}})
    table = pd.DataFrame(rows)
    # Score only where the aerostructural ROM was fitted (outside, the
    # polynomials extrapolate); the table keeps every row.
    in_band = table.u_p.between(*UP_BAND)
    table.to_csv(OUT_DIR / "rom_vs_as_trims.csv", index=False)

    ok = table[table.rom_ok & in_band]
    for name, g in ok.groupby("rom"):
        n_fail = int((~table[(table.rom == name) & in_band].rom_ok).sum())
        err_v = 100 * (g.rom_v_tau / g.as_v_tau - 1)
        err_t = 100 * (g.rom_tension / g.as_tension - 1)
        err_a = np.rad2deg(g.rom_alpha_w - g.as_alpha_w)
        steer = g[g.u_s != 0]
        print(f"\n{name}: {len(g)} solved, {n_fail} failed (u_p in {UP_BAND} m)")
        print(f"  v_tau     error  mean {err_v.mean():+6.2f} %  rms {err_v.std():5.2f} %  "
              f"max |{err_v.abs().max():5.2f}| %")
        print(f"  tension   error  mean {err_t.mean():+6.2f} %  rms {err_t.std():5.2f} %  "
              f"max |{err_t.abs().max():5.2f}| %")
        print(f"  alpha_w   error  mean {err_a.mean():+6.2f} deg rms {err_a.std():5.2f} deg")
        if len(steer):
            ratio = steer.rom_chi_dot / steer.as_chi_dot
            print(f"  chi_dot   ROM/AS on steered rows: median {ratio.median():.3f} "
                  f"(IQR {ratio.quantile(.25):.3f}-{ratio.quantile(.75):.3f})")


if __name__ == "__main__":
    main()
