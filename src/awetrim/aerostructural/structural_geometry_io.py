"""Read structural geometry YAML into solver arrays."""

import numpy as np
import logging

from awetrim.aerostructural.utils import calculate_cg, calculate_inertia


def compute_wing_stats_from_pss(struc_geometry):
    """Compute wing mass, center of mass, and inertia tensor from a PSS struc_geometry dict.

    Mass is distributed to nodes by splitting each element's m value equally between
    its two endpoints, mirroring initialize_wing_structure. The inertia tensor is the
    full 3x3 computed about the center of mass in the struc_geometry frame.

    Returns a dict with keys: mass, center_of_mass, inertia_tensor.
    """
    node_positions = {
        int(row[0]): np.array(row[1:4], dtype=float)
        for row in struc_geometry["wing_particles"]["data"]
    }
    wing_elements = {
        row[0]: dict(zip(struc_geometry["wing_elements"]["headers"][1:], row[1:]))
        for row in struc_geometry["wing_elements"]["data"]
    }

    node_mass = {nid: 0.0 for nid in node_positions}
    for conn_name, ci, cj in struc_geometry["wing_connections"]["data"]:
        m = wing_elements[conn_name]["m"]
        node_mass[ci] += m / 2
        node_mass[cj] += m / 2

    node_ids = sorted(node_positions.keys())
    struc_nodes = np.array([node_positions[nid] for nid in node_ids])
    m_arr = np.array([node_mass[nid] for nid in node_ids])

    total_mass = float(np.sum(m_arr))
    cg = calculate_cg(struc_nodes, m_arr)
    I = calculate_inertia(
        [(struc_nodes[i], m_arr[i]) for i in range(len(node_ids))], desired_point=cg
    )

    # Attempt to compute geometric properties (span, projected area) from
    # structural LE/TE node coordinates. Prefer VSM's Wing geometry utilities
    # when available; otherwise fall back to simple triangulation/projection.
    node_positions = {
        int(row[0]): np.array(row[1:4], dtype=float)
        for row in struc_geometry["wing_particles"]["data"]
    }
    le_ids = sorted([nid for nid in node_positions if nid % 2 != 0])
    te_ids = sorted([nid for nid in node_positions if nid % 2 == 0])

    le_pts = (
        np.array([node_positions[n] for n in le_ids]) if le_ids else np.empty((0, 3))
    )
    te_pts = (
        np.array([node_positions[n] for n in te_ids]) if te_ids else np.empty((0, 3))
    )

    span_val = None
    projected_area_val = None
    side_area_val = None
    flat_area_val = None

    try:
        from VSM.core.WingGeometry import Wing

        polar_list = [np.zeros((0, 4)) for _ in range(len(le_pts))]
        n_panels = max(1, len(le_pts) - 1)
        wing_geo = Wing(n_panels=n_panels)
        if len(le_pts) and len(te_pts) and len(le_pts) == len(te_pts):
            wing_geo.update_wing_from_points(
                le_pts, te_pts, "reuse_initial_polar_data", polar_list
            )
            span_val = float(wing_geo.span)
            # projected onto XY (default)
            projected_area_val = float(wing_geo.compute_projected_area())
            # side projection onto YZ (normal = x-axis)
            side_area_val = float(
                wing_geo.compute_projected_area(np.array([1.0, 0.0, 0.0]))
            )
            # flat / planform surface area (VSM may provide compute_flat_area)
            if hasattr(wing_geo, "compute_flat_area"):
                try:
                    flat_area_val = float(wing_geo.compute_flat_area())
                except Exception:
                    flat_area_val = None
    except Exception:
        # Fallback: compute flat and projected areas by triangulating successive LE/TE quads
        def tri_area(a, b, c):
            return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

        if len(le_pts) and len(te_pts) and len(le_pts) == len(te_pts):
            # span: y-extent of all LE/TE points
            all_y = np.concatenate([le_pts[:, 1], te_pts[:, 1]])
            span_val = float(np.max(all_y) - np.min(all_y))

            S_flat = 0.0
            S_proj_xy = 0.0
            S_side = 0.0
            for i in range(len(le_pts) - 1):
                A = le_pts[i]
                B = te_pts[i]
                C = te_pts[i + 1]
                D = le_pts[i + 1]
                # flat surface area (3D) triangles
                S_flat += tri_area(A, B, C) + tri_area(A, C, D)

                def tri_area_proj(a, b, c, drop_axis=2):
                    a2, b2, c2 = [np.delete(v, drop_axis) for v in [a, b, c]]
                    return 0.5 * abs(np.cross(b2 - a2, c2 - a2))

                # projected onto XY (drop z)
                S_proj_xy += tri_area_proj(A, B, C, drop_axis=2) + tri_area_proj(
                    A, C, D, drop_axis=2
                )
                # projected onto YZ (drop x)
                S_side += tri_area_proj(A, B, C, drop_axis=0) + tri_area_proj(
                    A, C, D, drop_axis=0
                )

            projected_area_val = float(S_proj_xy)
            side_area_val = float(S_side)
            flat_area_val = float(S_flat)

    result = {
        "mass": round(total_mass, 4),
        "center_of_mass": [round(float(v), 4) for v in cg],
        "inertia_tensor": [[round(float(v), 4) for v in row] for row in I],
    }

    # Include computed geometric fields when available
    if span_val is not None:
        result["span"] = round(float(span_val), 4)
    if projected_area_val is not None:
        result["projected_surface_area"] = round(float(projected_area_val), 4)
    if flat_area_val is not None:
        result["planform_surface_area"] = round(float(flat_area_val), 4)
    if side_area_val is not None:
        result["side_projected_area"] = round(float(side_area_val), 4)

    return result


def _resolve_kcu_mass(struc_geometry, config=None, system_config=None):
    """Resolve KCU mass.

    Priority:
      1. system_config (awesIO format): components.control_system.structure.mass
      2. config (aerostructural YAML): kcu.mass or kcu_mass
      3. struc_geometry fallback (legacy)
    """
    if isinstance(system_config, dict):
        kite = system_config.get("components", {}).get(
            "kite", system_config.get("components", {})
        )
        cs_struct = kite.get("control_system", {}).get("structure", {})
        if "mass" in cs_struct:
            return float(cs_struct["mass"])

    if isinstance(config, dict):
        kcu_cfg = config.get("kcu", {})
        if isinstance(kcu_cfg, dict) and "mass" in kcu_cfg:
            return float(kcu_cfg["mass"])
        if "kcu_mass" in config:
            return float(config["kcu_mass"])

    if "kcu_mass" in struc_geometry:
        return float(struc_geometry["kcu_mass"])

    logging.warning(
        "KCU mass not found in system_config, config, or structural geometry; defaulting to 0.0 kg."
    )
    return 0.0


def initialize_wing_structure(
    struc_geometry,
    struc_nodes,
    m_arr,
    kite_connectivity_arr,
    l0_arr,
    k_arr,
    c_arr,
    linktype_arr,
):
    """
    Create the structural nodes and connectivity for the kite structure.

    Args:
        geometry_dict (dict): Kite configuration dictionary.

    Returns:
        tuple: (struc_nodes, wing_ci, wing_cj, bridle_ci, bridle_cj, struc_node_le_indices,
                struc_node_te_indices, pulley_point_indices, tubular_frame_line_idx_list,
                te_line_idx_list, n_struc_ribs)
    """
    ### node level ###
    # node_indices = []
    struc_node_le_indices = []
    struc_node_te_indices = []

    # Then append rest of the defined wing_particles
    for node_idx, x, y, z in struc_geometry["wing_particles"]["data"]:
        # node_indices.append(node_idx)
        struc_nodes.append(np.array([x, y, z]))
        m_arr.append(0)

        # if uneven --> this is a leading-edge node
        if node_idx % 2 != 0:
            struc_node_le_indices.append(node_idx)
        else:
            struc_node_te_indices.append(node_idx)

    ### element level ###
    # Create an element dict of dicts: { name → {rest_length:..., diameter:..., ...} }
    wing_elements_dict = {
        row[0]: dict(zip(struc_geometry["wing_elements"]["headers"][1:], row[1:]))
        for row in struc_geometry["wing_elements"]["data"]
    }
    tubular_frame_line_idx_list = []
    te_line_idx_list = []
    for conn_idx, (conn_name, ci, cj) in enumerate(
        struc_geometry["wing_connections"]["data"]
    ):

        m_element = wing_elements_dict[conn_name]["m"]
        m_arr[ci] += m_element / 2
        m_arr[cj] += m_element / 2

        kite_connectivity_arr.append([ci, cj])
        l0_arr.append(wing_elements_dict[conn_name]["l0"])
        k_arr.append(wing_elements_dict[conn_name]["k"])
        c_arr.append(wing_elements_dict[conn_name]["c"])
        linktype_arr.append(wing_elements_dict[conn_name]["linktype"])

        if "le" in conn_name.lower() or "strut" in conn_name.lower():
            tubular_frame_line_idx_list.append(conn_idx)
        elif "te" in conn_name.lower():
            te_line_idx_list.append(conn_idx)

    return (
        # node level
        struc_nodes,
        m_arr,
        struc_node_le_indices,
        struc_node_te_indices,
        # element level
        kite_connectivity_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        tubular_frame_line_idx_list,
        te_line_idx_list,
    )


def initialize_bridle_line_system(
    struc_geometry,
    struc_nodes,
    m_arr,
    kite_connectivity_arr,
    l0_arr,
    k_arr,
    c_arr,
    linktype_arr,
):
    """
    Initialize the bridle line system for the kite.

    Returns:
        tuple: (bridle_ci, bridle_cj, pulley_point_indices)
    """

    ### node level ###

    # First append the bridle_point_node, as this node (KCU) should have index 0
    # Then append rest of the defined bridle_particles

    for node_idx, x, y, z in struc_geometry["bridle_particles"]["data"]:
        struc_nodes.append(np.array([x, y, z]))
        m_arr.append(0.0)

    ### element level ###

    # Create an element dict of dicts: { name → {l0:..., d:..., ...} }
    bridle_lines_dict = {
        row[0]: dict(zip(struc_geometry["bridle_lines"]["headers"][1:], row[1:]))
        for row in struc_geometry["bridle_lines"]["data"]
    }

    # initialize a connectivity counter, that starts with the number of wing_connections
    conn_idx_counter = len(kite_connectivity_arr)
    bridle_connectivity_arr = []
    bridle_diameter_arr = []
    pulley_node_indices = []
    pulley_line_indices = []
    pulley_line_to_other_node_pair_dict = {}
    steering_tape_indices = []
    for _, conn_data in enumerate(struc_geometry["bridle_connections"]["data"]):

        conn_name = conn_data[0]
        ci = int(conn_data[1])
        cj = int(conn_data[2])

        # computing the mass of the bridle line, and adding it 0.5 to each particle using m_arr
        l0 = bridle_lines_dict[conn_name]["l0"]
        material = bridle_lines_dict[conn_name]["material"]
        cross_sectional_area = np.pi * (bridle_lines_dict[conn_name]["d"] / 2) ** 2
        m_line = struc_geometry[material]["density"] * cross_sectional_area * l0
        m_arr[ci] += m_line / 2
        m_arr[cj] += m_line / 2

        # If there is third connections, this line is a pulley!
        # In here we will treat both ci-cj and cj-ck
        if len(conn_data[1:]) == 3:
            logging.debug(
                f"-- linktype should be pulley, linktype: {bridle_lines_dict[conn_name]["linktype"]}"
            )
            # adding pulley_node_indices
            pulley_node_indices.append(cj)

            # making the third node an integer
            ck = int(conn_data[3])

            # add the pulley mass, to the pulley_index cj
            m_arr[cj] += struc_geometry["pulley_mass"]

            #######################################
            # Computing k,c
            # Compute the straight-line distances between the nodes involved in the pulley:
            # - len_ci_cj: distance between ci and cj (the current segment)
            # - len_cj_ck: distance between cj and ck (the other segment)
            len_ci_cj = np.linalg.norm(
                np.array(struc_nodes[ci]) - np.array(struc_nodes[cj])
            )
            len_cj_ck = np.linalg.norm(
                np.array(struc_nodes[cj]) - np.array(struc_nodes[ck])
            )

            # The total straight-line length is the sum of both segments
            len_ci_cj_ck = len_ci_cj + len_cj_ck

            # Divide the rest length proportionally between the two segments,
            # based on their straight-line distances ratios to the total length
            l0_len_ci_cj = (len_ci_cj / len_ci_cj_ck) * l0
            l0_len_cj_ck = (len_cj_ck / len_ci_cj_ck) * l0

            k = (struc_geometry[material]["youngs_modulus"] * cross_sectional_area) / (
                l0
            )
            c = struc_geometry[material]["damping_per_stiffness"] * k

            #######################################
            # Dealing with ci-cj
            # add this new connection to the connectivity array, and also increase counter
            kite_connectivity_arr.append([ci, cj])
            bridle_connectivity_arr.append([ci, cj])
            bridle_diameter_arr.append(bridle_lines_dict[conn_name]["d"])
            l0_arr.append(l0)
            k_arr.append(k)
            c_arr.append(c)
            linktype_arr.append(bridle_lines_dict[conn_name]["linktype"])

            # Create a special mapping for the Structural Particle System Solver
            # key: pulley_line_index
            # value: [cj, ck, line_len_other]
            # This is used to connect the pulley line to the other node pair
            pulley_line_to_other_node_pair_dict[str(conn_idx_counter)] = np.array(
                [
                    cj,
                    ck,
                    l0_len_cj_ck,
                    l0_len_ci_cj,
                    ci,
                ]
            )

            # Mark the indices of connectivity
            pulley_line_index_ci_cj = conn_idx_counter
            pulley_line_indices.append(pulley_line_index_ci_cj)
            conn_idx_counter += 1

            #######################################
            # Dealing with cj-ck
            # add this new connection to the connectivity array, and also increase counter
            kite_connectivity_arr.append([cj, ck])
            bridle_connectivity_arr.append([cj, ck])
            bridle_diameter_arr.append(bridle_lines_dict[conn_name]["d"])
            l0_arr.append(l0)
            k_arr.append(k)
            c_arr.append(c)
            linktype_arr.append(bridle_lines_dict[conn_name]["linktype"])

            # Create a special mapping for the Structural Particle System Solver
            # key: pulley_line_index
            # value: [cj, ck, line_len_other]
            # This is used to connect the pulley line to the other node pair
            pulley_line_to_other_node_pair_dict[str(conn_idx_counter)] = np.array(
                [
                    cj,
                    ci,
                    l0_len_ci_cj,
                    l0_len_cj_ck,
                    ci,
                ]
            )

            # Mark the indices of connectivity
            pulley_line_index_cj_ck = conn_idx_counter
            pulley_line_indices.append(pulley_line_index_cj_ck)

        # if there is no third connections this line represents a knot-to-knot line, a regular spring damper
        elif len(conn_data[1:]) == 2:
            logging.debug(
                f"-- linktype should be noncompressive, linktype: {bridle_lines_dict[conn_name]["linktype"]}"
            )
            # add this new connection to the connectivity array, and also increase counter
            k = (struc_geometry[material]["youngs_modulus"] * cross_sectional_area) / l0
            c = (
                struc_geometry[material]["damping_per_stiffness"] * k
            )  # Rayleigh damping
            kite_connectivity_arr.append([ci, cj])
            bridle_connectivity_arr.append([ci, cj])
            bridle_diameter_arr.append(bridle_lines_dict[conn_name]["d"])
            l0_arr.append(l0)
            k_arr.append(k)
            c_arr.append(c)
            linktype_arr.append(bridle_lines_dict[conn_name]["linktype"])

        else:
            raise ValueError(
                "bridle_connections should have 2 or 3 connections (ci,cj,ck), not more or less"
            )

        if conn_name == "Power Tape":
            power_tape_index = conn_idx_counter

        if conn_name == "Steering Tape":
            steering_tape_indices.append(conn_idx_counter)

        ## increasing the counter
        conn_idx_counter += 1

    return (
        # node level
        struc_nodes,
        m_arr,
        power_tape_index,
        steering_tape_indices,
        pulley_node_indices,
        # element level
        kite_connectivity_arr,
        bridle_connectivity_arr,
        bridle_diameter_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        pulley_line_indices,
        pulley_line_to_other_node_pair_dict,
    )


def compute_bridle_stats_from_pss(struc_geometry):
    """Compute bridle summary statistics from a PSS struc_geometry dict.

    Returns a dict with:
      total_nominal_line_length  — sum of unique bridle_line l0 values (m)
      avg_line_diameter          — connection-length-weighted average diameter (m)
      bridle_line_count          — number of unique line definitions
      bridle_particle_count      — number of bridle particles (excluding KCU node)
      bridle_connection_count    — number of connection entries
      pulley_count               — number of 3-node (pulley) connections
      mass                       — total bridle mass: lines + pulleys (kg)
    """
    bridle_lines_dict = {
        row[0]: dict(zip(struc_geometry["bridle_lines"]["headers"][1:], row[1:]))
        for row in struc_geometry["bridle_lines"]["data"]
    }

    total_nominal_line_length = sum(p["l0"] for p in bridle_lines_dict.values())

    mass_lines = 0.0
    weighted_d_sum = 0.0
    total_conn_length = 0.0
    pulley_count = 0

    for conn_data in struc_geometry["bridle_connections"]["data"]:
        name = conn_data[0]
        line = bridle_lines_dict[name]
        l0, d, material = line["l0"], line["d"], line["material"]
        area = np.pi * (d / 2) ** 2
        density = struc_geometry[material]["density"]
        mass_lines += density * area * l0
        weighted_d_sum += d * l0
        total_conn_length += l0
        if len(conn_data[1:]) == 3:
            pulley_count += 1

    avg_line_diameter = weighted_d_sum / total_conn_length
    pulley_mass_per = struc_geometry.get("pulley_mass", 0.0)
    mass_total = mass_lines + pulley_count * pulley_mass_per

    return {
        "total_nominal_line_length": round(total_nominal_line_length, 5),
        "avg_line_diameter": round(avg_line_diameter, 5),
        "bridle_line_count": len(bridle_lines_dict),
        "bridle_particle_count": len(struc_geometry["bridle_particles"]["data"]),
        "bridle_connection_count": len(struc_geometry["bridle_connections"]["data"]),
        "pulley_count": pulley_count,
        "mass": round(mass_total, 4),
    }


def main(struc_geometry, config=None, system_config=None):

    ### First append the bridle_point_node, as this node (KCU) should have index 0
    struc_nodes = []
    m_arr = []
    struc_nodes.append(np.array(struc_geometry["bridle_point_node"]))
    m_arr.append(
        _resolve_kcu_mass(struc_geometry, config=config, system_config=system_config)
    )

    # initialize element level lists
    kite_connectivity_arr = []
    l0_arr = []
    k_arr = []
    c_arr = []
    linktype_arr = []

    ### Analyze Wing Structure
    (
        # node level
        struc_nodes,
        m_arr,
        struc_node_le_indices,
        struc_node_te_indices,
        # element level
        kite_connectivity_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        tubular_frame_line_idx_list,
        te_line_idx_list,
    ) = initialize_wing_structure(
        struc_geometry,
        struc_nodes,
        m_arr,
        kite_connectivity_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
    )

    ### Analyze Bridle Structure
    (
        # node level
        struc_nodes,
        m_arr,
        power_tape_index,
        steering_tape_indices,
        pulley_node_indices,
        # element level
        kite_connectivity_arr,
        bridle_connectivity_arr,
        bridle_diameter_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        pulley_line_indices,
        pulley_line_to_other_node_pair_dict,
    ) = initialize_bridle_line_system(
        struc_geometry,
        struc_nodes,
        m_arr,
        kite_connectivity_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
    )

    # explicit numpy arrays
    struc_nodes = np.array(struc_nodes)
    kite_connectivity_arr = np.array(kite_connectivity_arr)
    bridle_connectivity_arr = np.array(bridle_connectivity_arr)
    bridle_diameter_arr = np.array(bridle_diameter_arr)
    l0_arr = np.array(l0_arr)
    k_arr = np.array(k_arr)
    c_arr = np.array(c_arr)
    m_arr = np.array(m_arr)
    linktype_arr = np.array(linktype_arr)
    struc_node_le_indices = np.array(struc_node_le_indices)
    struc_node_te_indices = np.array(struc_node_te_indices)

    return (
        # node level
        struc_nodes,
        m_arr,
        struc_node_le_indices,
        struc_node_te_indices,
        power_tape_index,
        steering_tape_indices,
        pulley_node_indices,
        # element level
        kite_connectivity_arr,
        bridle_connectivity_arr,
        bridle_diameter_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        pulley_line_indices,
        pulley_line_to_other_node_pair_dict,
    )
