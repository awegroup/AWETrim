import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

PALETTE = {
    "Black": "#000000",
    "Orange": "#E69F00",
    "Sky Blue": "#56B4E9",
    "Bluish Green": "#009E73",
    "Yellow": "#F0E442",
    "Blue": "#0072B2",
    "Vermillion": "#D55E00",
    "Reddish Purple": "#CC79A7",
}


def set_plot_style():
    """
    Set the default style for plots using LaTeX and custom color palette.

    Tips:
    - If you specify colors, they will still be used.
    - If you want to change the axis margins:
        1. try with ax.xlim and ax.ylim
        2. try by changing the 'axes.autolimit_mode' parameter to data
    - more?
    """

    # Define the color palette as a list of colors
    color_cycle = [
        PALETTE["Black"],
        PALETTE["Orange"],
        PALETTE["Sky Blue"],
        PALETTE["Bluish Green"],
        PALETTE["Yellow"],
        PALETTE["Blue"],
        PALETTE["Vermillion"],
        PALETTE["Reddish Purple"],
    ]

    # Apply Seaborn style and custom settings
    # plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "DejaVu Serif",
                "Times New Roman",
                "serif",
            ],
            ## Axes settings
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#C5C5C5",
            "axes.labelcolor": "black",
            "axes.autolimit_mode": "round_numbers",
            "axes.xmargin": 0,  # Remove extra margin
            "axes.ymargin": 0,  # Remove extra margin
            ## Grid settings
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.alpha": 0.5,
            "grid.color": "#C5C5C5",
            "grid.linestyle": "-",
            "grid.linewidth": 1.0,
            ## Line settings
            "lines.linewidth": 1,
            "lines.markersize": 6,
            # "lines.color": "grey",,
            "figure.titlesize": 15,
            "pgf.texsystem": "pdflatex",  # Use pdflatex
            "pgf.rcfonts": False,
            "figure.figsize": (15, 5),  # Default figure size
            "axes.prop_cycle": cycler(
                "color", color_cycle
            ),  # Set the custom color cycle
            ## tick settings
            "xtick.color": "#C5C5C5",
            "ytick.color": "#C5C5C5",
            "xtick.labelcolor": "black",
            "ytick.labelcolor": "black",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.top": True,  # Show ticks on both sides
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.right": True,
            "xtick.direction": "in",  # Direction for x-axis ticks
            "ytick.direction": "in",  # Direction for y-axis ticks
            ## legend settings
            "legend.fontsize": 15,
        }
    )


def plot_normalized_elongation(
    ax,
    kite_connectivity_arr,
    struc_nodes,
    rest_lengths,
    pulley_line_indices,  # kept for signature; used only to skip
    pulley_line_to_other_node_pair_dict,  # kept for signature; used to get (cj, ck)
):

    cmap = plt.get_cmap("viridis")

    num_conns = len(kite_connectivity_arr)
    rest_lengths = np.asarray(rest_lengths, dtype=float)
    norm_values = np.full(num_conns, np.nan, dtype=float)

    # map undirected pair -> index (if duplicates exist, last one wins; that’s fine for plotting)
    pair_to_idx = {}
    for idx, (i, j, *_) in enumerate(kite_connectivity_arr):
        ci, cj = int(i), int(j)
        pair_to_idx[frozenset((ci, cj))] = idx

    # treat pulley indices as a set for O(1) membership
    pulley_set = set(int(i) for i in (pulley_line_indices or []))

    pulley_triples = []  # will store (ci, cj, ck) for later plotting with cmap

    # compute norm only for non-pulley lines; collect pulley triples
    for idx, (i, j, *_) in enumerate(kite_connectivity_arr):
        if (
            idx in pulley_set
            and str(idx) in (pulley_line_to_other_node_pair_dict or {}).keys()
        ):
            pulley_other_info = pulley_line_to_other_node_pair_dict[str(idx)]
            logging.debug(f"cj-loop: {j}, cj-other info: {int(pulley_other_info[0])}")
            ci = int(i)
            cj = int(pulley_other_info[0])
            ck = int(pulley_other_info[1])
            pulley_triples.append((ci, cj, ck))
            logging.debug(f"pulley triplet: {(ci, cj, ck)}")
            continue  # skip pulley lines in this pass

        ci, cj = int(i), int(j)
        p1, p2 = np.asarray(struc_nodes[ci]), np.asarray(struc_nodes[cj])
        curr_len = float(np.linalg.norm(p2 - p1))
        rl = rest_lengths[idx] if idx < len(rest_lengths) else np.nan

        norm_values[idx] = (
            0.0 if (not np.isfinite(rl) or rl == 0.0) else 100.0 * (curr_len - rl) / rl
        )

    # color scaling based only on finite (non-pulley) values
    finite_mask = np.isfinite(norm_values)
    if not np.any(finite_mask):
        return  # nothing to draw

    vmin = float(np.nanmin(norm_values))
    vmax = float(np.nanmax(norm_values))

    # plot only non-pulley lines with cmap
    for idx, (i, j, *_) in enumerate(kite_connectivity_arr):
        if idx in pulley_set:
            continue
        ci, cj = int(i), int(j)
        p1, p2 = struc_nodes[ci], struc_nodes[cj]
        val = norm_values[idx]
        t = (val - vmin) / (vmax - vmin) if np.isfinite(val) and vmax > vmin else 0.5
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=cmap(t),
            linewidth=2,
        )

    # --- NEW: plot pulley segments using the same cmap, with two-segment norm ---
    for ci, cj, ck in pulley_triples:
        # current lengths
        p_ci, p_cj, p_ck = (
            np.asarray(struc_nodes[ci]),
            np.asarray(struc_nodes[cj]),
            np.asarray(struc_nodes[ck]),
        )
        cl_ci_cj = float(np.linalg.norm(p_cj - p_ci))
        cl_cj_ck = float(np.linalg.norm(p_ck - p_cj))
        curr_total = cl_ci_cj + cl_cj_ck

        # rest lengths from connectivity (driver is (ci,cj), mate is (cj,ck) if present)
        driver_idx = pair_to_idx.get(frozenset((ci, cj)))
        mate_idx = pair_to_idx.get(frozenset((cj, ck)))

        if driver_idx is not None and mate_idx is not None:
            rest_total = float(rest_lengths[driver_idx]) + float(rest_lengths[mate_idx])
        elif driver_idx is not None:
            # fallback if mate not in connectivity
            rest_total = float(rest_lengths[driver_idx])
        else:
            # nothing we can do; skip
            continue

        norm_val = (
            0.0 if rest_total == 0.0 else 100.0 * (curr_total - rest_total) / rest_total
        )
        # map to same colormap range (clamp into [0,1] if outside)
        if vmax > vmin:
            t = (norm_val - vmin) / (vmax - vmin)
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        else:
            t = 0.5
        color = cmap(t)

        # draw both segments with identical color
        ax.plot(
            [p_ci[0], p_cj[0]],
            [p_ci[1], p_cj[1]],
            [p_ci[2], p_cj[2]],
            color=color,
            linewidth=2,
        )
        ax.plot(
            [p_cj[0], p_ck[0]],
            [p_cj[1], p_ck[1]],
            [p_cj[2], p_ck[2]],
            color=color,
            linewidth=2,
        )

    # legend + colorbar
    ax.legend(loc="center right", bbox_to_anchor=(1.05, 0.5))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label(r"Normalized rest length change (\%)")
    cbar.ax.text(
        1.5,
        -0.1,
        "contracted",
        va="center",
        ha="left",
        fontsize=11,
        color="black",
        transform=cbar.ax.transAxes,
    )
    cbar.ax.text(
        1.5,
        1.1,
        "elongated",
        va="center",
        ha="left",
        fontsize=11,
        color="black",
        transform=cbar.ax.transAxes,
    )


def main(
    struc_nodes,
    kite_connectivity_arr,
    rest_lengths,
    struc_nodes_initial=None,
    f_ext=None,
    f_bridle=None,
    f_inertial=None,
    title="PSM State",
    body_aero=None,
    vel_app=None,
    ##TODO: V3 chord-length used for scaling vectors -> should not be hardcoded
    chord_length=2.6,
    is_with_node_indices=False,
    lightred_color="#FF5F5F",
    pulley_line_indices=None,
    pulley_line_to_other_node_pair_dict=None,
    label_current_particles="Current",
):
    """
    Plot the current (and optionally initial) structure state in 3D.

    Args:
        struc_nodes (np.ndarray): Current node positions (n_nodes, 3).
        kite_connectivity_arr (array-like): List/array of [i, j, ...] giving spring connections.
        struc_nodes_initial (np.ndarray, optional): Initial node positions (n_nodes, 3).
        f_ext (np.ndarray or None): Optional external forces, shape (n_nodes, 3) or flat.
        f_bridle (np.ndarray or None): Optional bridle-only forces, shape (n_nodes, 3) or flat.
        f_inertial (np.ndarray or None): Optional inertial-only forces, shape (n_nodes, 3) or flat.
        title (str): Figure title.
        chord_length (float): Maximum length for force vectors (for scaling).

    Returns:
        None. Displays a 3D plot.
    """
    kite_connectivity_arr = np.array(kite_connectivity_arr)  # Ensure numeric array

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Try to maximize the plot window (works for Qt backends)
    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except Exception:
        # Fallback: set a large figure size
        fig.set_size_inches(16, 9, forward=True)

    # Plot initial state if provided (single color for reference)
    if struc_nodes_initial is not None:
        label_current_particles = "Final"
        ax.scatter(
            *(struc_nodes_initial.T), color="blue", marker="o", s=10, label="Initial"
        )
        # Draw initial lines in pink
        for idx, (i, j, *rest) in enumerate(kite_connectivity_arr):
            p1, p2 = struc_nodes_initial[i], struc_nodes_initial[j]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color="blue",
                linewidth=1,
                alpha=0.5,
            )

    # Plot current state
    ax.scatter(
        *(struc_nodes.T), color="black", marker="o", s=10, label=label_current_particles
    )

    # Plot normalized elongation
    plot_normalized_elongation(
        ax,
        kite_connectivity_arr,
        struc_nodes,
        rest_lengths,
        pulley_line_indices,
        pulley_line_to_other_node_pair_dict,
    )

    # Optionally plot external forces
    if f_ext is not None:
        arr = np.array(f_ext)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        # Scale all vectors so the longest has length chord_length/5 (larger arrows)
        norms = np.linalg.norm(arr, axis=1)
        max_norm = np.max(norms) if np.max(norms) > 0 else 1.0
        scale = (chord_length) / max_norm
        arr_scaled = arr * scale
        ax.quiver(
            struc_nodes[:, 0],
            struc_nodes[:, 1],
            struc_nodes[:, 2],
            arr_scaled[:, 0],
            arr_scaled[:, 1],
            arr_scaled[:, 2],
            length=1,
            normalize=False,
            color=lightred_color,
        )

    # Optionally overlay bridle-only force vectors for visibility
    if f_bridle is not None:
        arr_bridle = np.array(f_bridle)
        if arr_bridle.ndim == 1:
            arr_bridle = arr_bridle.reshape(-1, 3)
        norms_bridle = np.linalg.norm(arr_bridle, axis=1)
        max_norm_bridle = np.max(norms_bridle) if np.max(norms_bridle) > 0 else 1.0
        scale_bridle = (chord_length) / max_norm_bridle
        arr_bridle_scaled = arr_bridle * scale_bridle
        ax.quiver(
            struc_nodes[:, 0],
            struc_nodes[:, 1],
            struc_nodes[:, 2],
            arr_bridle_scaled[:, 0],
            arr_bridle_scaled[:, 1],
            arr_bridle_scaled[:, 2],
            length=1,
            normalize=False,
            color="tab:green",
            label="Bridle Aero",
        )

    # Optionally overlay inertial-only force vectors for visibility
    if f_inertial is not None:
        arr_inertial = np.array(f_inertial)
        if arr_inertial.ndim == 1:
            arr_inertial = arr_inertial.reshape(-1, 3)
        norms_inertial = np.linalg.norm(arr_inertial, axis=1)
        max_norm_inertial = (
            np.max(norms_inertial) if np.max(norms_inertial) > 0 else 1.0
        )
        scale_inertial = (chord_length) / max_norm_inertial
        arr_inertial_scaled = arr_inertial * scale_inertial
        ax.quiver(
            struc_nodes[:, 0],
            struc_nodes[:, 1],
            struc_nodes[:, 2],
            arr_inertial_scaled[:, 0],
            arr_inertial_scaled[:, 1],
            arr_inertial_scaled[:, 2],
            length=1,
            normalize=False,
            color="tab:purple",
            label="Inertial",
        )

    if is_with_node_indices:
        # Annotate node indices
        for i, pos in enumerate(struc_nodes):
            ax.text(pos[0], pos[1], pos[2], str(i), color="black", fontsize=8)
        if struc_nodes_initial is not None:
            for i, pos in enumerate(struc_nodes_initial):
                ax.text(pos[0], pos[1], pos[2], str(i), color="blue", fontsize=8)

    # If aero mesh nodes are provided, plot them
    if body_aero is not None:
        aero_mesh_nodes = []
        for panel in body_aero.panels:
            for cp in panel.corner_points:
                aero_mesh_nodes.append(cp)
        # make unique
        aero_mesh_nodes = np.unique(np.array(aero_mesh_nodes), axis=0)
        ax.scatter(
            *(aero_mesh_nodes.T),
            color=lightred_color,
            marker="^",
            s=10,
            label="Aero Mesh",
        )

    if vel_app is not None:
        vel_app_scaled = vel_app * (chord_length / 2) / np.linalg.norm(vel_app)
        ax.quiver(
            0,
            0,
            0,
            vel_app_scaled[0],
            vel_app_scaled[1],
            vel_app_scaled[2],
            length=1,
            linewidth=3,
            arrow_length_ratio=0.3,
            normalize=False,
            color="black",
            label="Apparent Wind",
        )
        # add a text label for the apparent wind vector
        ax.text(
            vel_app_scaled[0],
            vel_app_scaled[1],
            vel_app_scaled[2],
            r"$V_\mathrm{a}$",
            color="black",
            fontsize=10,
        )

    # Set aspect ratio to equal
    all_pts = (
        struc_nodes
        if struc_nodes_initial is None
        else np.vstack((struc_nodes, struc_nodes_initial))
    )
    bb = all_pts.max(axis=0) - all_pts.min(axis=0)
    ax.set_box_aspect(bb)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plt.show()

