"""Show what conserving the panel pitching moment does to the aero to structure load map.

The chordwise weights decide where along each chord a panel's aerodynamic force
ends up acting on the structure, and that station *is* the panel's local
pitching moment. The historical mapping reads one measured ``Delta C_p`` shape
from a single-angle file and applies it to every panel, so the resultant sits at
one fixed chordwise station whatever the panel's own ``C_m`` says. The wing's
trim is a moment balance, so the coupled loop then re-trims each iteration
against a moment the structure was never given.

``aero2struc.weights_at_centre_of_pressure`` keeps the measured shape as a prior
and tilts it onto the centre of pressure VSM already computes from each panel's
own force and moment. This script solves one VSM state on the Billow geometry
and draws the difference.

Usage (from project root):
    python scripts/aerostructural/plot_chordwise_moment_matching.py
"""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml

from awetrim import plotting
from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.billow import structural_billow
from awetrim.aerostructural.coupled import aero2struc, read_struc_geometry_yaml
from awetrim.aerostructural.mapping import LinearStructuralToAeroMapper
from awetrim.aerostructural.utils import calculate_cg, load_yaml, rotate_geometry
from awetrim.system.tether import RigidLumpedTether
from awetrim.utils.system_config import get_tether
from common import (
    DEFAULT_KITE_NAME,
    build_system_model,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)

STRUC_GEOMETRY_FILENAME = "struc_geometry_FEM_full.yaml"
N_CHORDWISE = 10


def solve_one_state():
    """One VSM trim on the relaxed Billow geometry; returns panels and results."""
    project_dir = Path(__file__).resolve().parents[2]
    kite_name = DEFAULT_KITE_NAME
    config_path, aero_geometry_path, _ = resolve_kite_paths(project_dir, kite_name)
    struc_geometry_path = project_dir / "data" / kite_name / STRUC_GEOMETRY_FILENAME
    system_config_path = project_dir / "data" / kite_name / "system.yaml"

    with system_config_path.open("r", encoding="utf-8") as handle:
        system_config = _yaml.safe_load(handle)
    config = load_yaml(config_path)
    config["structural_solver"] = "billow"
    cp_rel = config.get("aero2struc", {}).get("cp_distribution_path")
    if cp_rel:
        config["aero2struc"]["cp_distribution_path"] = str(project_dir / cp_rel)

    struc_geometry = load_yaml(struc_geometry_path)
    n_struc_ribs = len(struc_geometry["wing_particles"]["data"]) / 2
    n_panels_aero = (n_struc_ribs - 1) * config["aerodynamic"][
        "n_aero_panels_per_struc_section"
    ]
    body_aero, vsm_solver, _, initial_polar_data = aerodynamic_vsm.initialize(
        aero_geometry_path, config, n_panels_aero, bridle_path=None
    )

    reader = read_struc_geometry_yaml.main(
        struc_geometry, config=config, system_config=system_config
    )
    (struc_nodes, m_arr, le_indices, te_indices, *_rest) = reader
    (
        canopy_sections,
        strut_sections,
        _simplified,
        kite_connectivity_arr,
        _bridle_conn,
        _bridle_diam,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        pulley_line_indices,
        _pulley_dict,
    ) = reader[7:]

    struc_nodes = rotate_geometry(
        struc_nodes, **resolve_initial_geometry_rotation_kwargs(config)
    )
    structure = structural_billow.instantiate(
        config=config,
        struc_geometry=struc_geometry,
        struc_nodes=struc_nodes,
        kite_connectivity_arr=kite_connectivity_arr,
        l0_arr=l0_arr,
        k_arr=k_arr,
        c_arr=c_arr,
        m_arr=m_arr,
        linktype_arr=linktype_arr,
        pulley_line_indices=pulley_line_indices,
        canopy_sections=canopy_sections,
        strut_sections=strut_sections,
    )
    struc_nodes = structure.model.nodes.copy()

    update = LinearStructuralToAeroMapper().map(
        struc_nodes,
        le_indices,
        te_indices,
        config["aerodynamic"]["n_aero_panels_per_struc_section"],
    )
    tether_struct = get_tether(system_config)["structure"]
    system_model = build_system_model(
        system_config_path,
        RigidLumpedTether(
            diameter=tether_struct["diameter"],
            density=tether_struct.get("density", 970.0),
        ),
        m_arr,
        config,
    )

    forces, body_aero, results = aerodynamic_vsm.run_vsm_package(
        body_aero=copy.deepcopy(body_aero),
        solver=copy.deepcopy(vsm_solver),
        system_model=system_model,
        center_of_gravity=calculate_cg(struc_nodes=struc_nodes, m_arr=m_arr),
        le_arr=update.leading_edge_points,
        te_arr=update.trailing_edge_points,
        aero_input_type="reuse_initial_polar_data",
        initial_polar_data=copy.deepcopy(initial_polar_data),
        include_gravity=config["is_with_gravity"],
        is_with_plot=False,
    )
    return body_aero.panels, np.asarray(forces), results, config


def panel_geometry(panels, results):
    """Spanwise station, chord, and centre-of-pressure fraction for each panel."""
    stations, chords, cp_fraction = [], [], []
    centres = np.asarray(results["panel_cp_locations"], dtype=float)
    for index, panel in enumerate(panels):
        le_mid = 0.5 * (panel.LE_point_1 + panel.LE_point_2)
        te_mid = 0.5 * (panel.TE_point_1 + panel.TE_point_2)
        chord = te_mid - le_mid
        chord_squared = float(chord @ chord)
        stations.append(0.5 * (le_mid[1] + te_mid[1]))
        chords.append(np.sqrt(chord_squared))
        cp_fraction.append(float((centres[index] - le_mid) @ chord) / chord_squared)
    return np.asarray(stations), np.asarray(chords), np.asarray(cp_fraction)


def build_weight_maps(config, panels, cp_fraction):
    """Chordwise weight rows for both mappings: (n_panels, N_CHORDWISE)."""
    stations = np.linspace(0.0, 1.0, N_CHORDWISE)
    prior = aero2struc.chordwise_weights_from_cp_file(
        config["aero2struc"]["cp_distribution_path"], stations
    )
    fixed = np.tile(prior, (len(panels), 1))
    matched = np.stack(
        [
            aero2struc.weights_at_centre_of_pressure(prior, stations, fraction)
            for fraction in cp_fraction
        ]
    )
    return stations, prior, fixed, matched


def pitching_moment_error(weights, stations, cp_fraction, chords, forces):
    """Per-panel pitching-moment error of a chordwise mapping (N m).

    The load acts along the chord line, so the moment it produces about the
    panel's span axis is set entirely by the centroid of the weights. The error
    is the lever-arm shortfall times the normal force.
    """
    centroid = weights @ stations
    normal_force = np.linalg.norm(forces, axis=1)
    return (centroid - cp_fraction) * chords * normal_force


def main():
    plotting.set_plot_style()
    panels, forces, results, config = solve_one_state()
    stations_y, chords, cp_fraction = panel_geometry(panels, results)
    stations, prior, fixed, matched = build_weight_maps(config, panels, cp_fraction)

    prior_centroid = float(prior @ stations)
    error_fixed = pitching_moment_error(fixed, stations, cp_fraction, chords, forces)
    error_matched = pitching_moment_error(
        matched, stations, cp_fraction, chords, forces
    )

    print(f"panels                        {len(panels)}")
    print(f"fixed chordwise station       {prior_centroid:.4f} c")
    print(
        f"actual centre of pressure     {cp_fraction.min():.4f} .. "
        f"{cp_fraction.max():.4f} c  (mean {cp_fraction.mean():.4f})"
    )
    print(
        f"pitching-moment error, fixed  {np.abs(error_fixed).sum():8.2f} N m total, "
        f"{np.abs(error_fixed).max():6.2f} N m worst panel"
    )
    print(
        f"pitching-moment error, matched{np.abs(error_matched).sum():8.2e} N m total, "
        f"{np.abs(error_matched).max():6.2e} N m worst panel"
    )

    # PALETTE is a name -> hex map (Okabe-Ito); pick explicitly rather than
    # by index so the figure reads the same if the palette order changes.
    colours = plotting.PALETTE
    actual, reference = colours["Blue"], colours["Vermillion"]
    accents = [colours["Bluish Green"], colours["Orange"], colours["Reddish Purple"]]
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))

    # (a) where the load actually acts, along the span
    ax = axes[0, 0]
    ax.axhline(
        prior_centroid,
        color=reference,
        lw=2.0,
        label=f"fixed $\\Delta C_p$ file ({prior_centroid:.3f})",
    )
    ax.plot(
        stations_y, cp_fraction, color=actual, lw=1.8, label="panel centre of pressure"
    )
    ax.fill_between(
        stations_y,
        prior_centroid,
        cp_fraction,
        color=actual,
        alpha=0.18,
        label="imposed moment error",
    )
    ax.set_xlabel("spanwise station $y$ (m)")
    ax.set_ylabel("chordwise position of the resultant $x/c$")
    ax.set_title("(a) Where the panel load is placed")
    ax.legend(loc="best", fontsize=8)
    ax.invert_yaxis()

    # (b) the chordwise weights themselves, at three stations
    ax = axes[0, 1]
    picks = [
        int(np.argmin(np.abs(stations_y - stations_y.min() * 0.9))),
        int(len(panels) // 2),
        int(np.argmin(np.abs(stations_y - stations_y.max() * 0.9))),
    ]
    ax.plot(
        stations,
        prior,
        color=reference,
        lw=2.2,
        marker="o",
        ms=4,
        label=f"prior, every panel ({prior_centroid:.3f})",
    )
    for offset, index in enumerate(picks):
        ax.plot(
            stations,
            matched[index],
            color=accents[offset % len(accents)],
            lw=1.5,
            marker="s",
            ms=3,
            label=f"tilted, $y$ = {stations_y[index]:+.2f} m ({cp_fraction[index]:.3f})",
        )
    ax.set_xlabel("chordwise station $x/c$")
    ax.set_ylabel("share of the panel force")
    ax.set_title("(b) Chordwise weights: prior and tilted")
    ax.legend(loc="best", fontsize=8)

    # (c) the load map difference
    ax = axes[1, 0]
    magnitude = np.linalg.norm(forces, axis=1)
    difference = (matched - fixed) * magnitude[:, None]
    limit = float(np.abs(difference).max())
    mesh = ax.pcolormesh(
        stations,
        stations_y,
        difference,
        cmap="awetrim_diverging",
        vmin=-limit,
        vmax=limit,
        shading="nearest",
    )
    figure.colorbar(mesh, ax=ax, label="change in nodal load (N)")
    ax.set_xlabel("chordwise station $x/c$")
    ax.set_ylabel("spanwise station $y$ (m)")
    ax.set_title("(c) Load moved by conserving the moment")

    # (d) what that is worth in pitching moment
    ax = axes[1, 1]
    ax.plot(
        stations_y,
        error_fixed,
        color=reference,
        lw=1.8,
        label="fixed $\\Delta C_p$ file",
    )
    ax.plot(
        stations_y,
        error_matched,
        color=actual,
        lw=1.8,
        label="moment matched",
    )
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("spanwise station $y$ (m)")
    ax.set_ylabel("panel pitching-moment error (N m)")
    ax.set_title("(d) Local moment the structure receives")
    ax.legend(loc="best", fontsize=8)

    figure.tight_layout()
    output = Path(__file__).resolve().parents[2] / "chordwise_moment_matching.png"
    figure.savefig(output, dpi=180)
    print(f"\nwritten to {output}")
    plt.show()


if __name__ == "__main__":
    main()
