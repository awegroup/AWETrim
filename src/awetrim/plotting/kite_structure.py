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

"""3-D view of a kite line system, drawn from the structural arrays.

It takes arrays, not a solver object, so it is independent of which structural
backend produced them -- which is why it lives here rather than beside one of
them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_3d_kite_structure(
    struc_nodes,
    kite_connectivity_arr,
    power_tape_index,
    k_arr=None,
    c_arr=None,
    linktype_arr=None,
    fixed_nodes=None,
    pulley_nodes=None,
):
    """
    Plot the 3D structure of a kite with enhanced visualization features.

    Args:
        struc_nodes (np.ndarray): Array of 3D coordinates for each node (n_nodes, 3).
        kite_connectivity_arr (np.ndarray): Array of [ci, cj] node pairs for each connection.
        power_tape_index (int): Index of the power tape connection.
        k_arr (np.ndarray, optional): Array of stiffness values for each connection.
        c_arr (np.ndarray, optional): Array of damping values for each connection.
        linktype_arr (np.ndarray, optional): Array of link types for each connection.
        fixed_nodes (iterable, optional): Indices of fixed nodes.
        pulley_nodes (iterable, optional): Indices of pulley nodes.

    Returns:
        None. Displays a 3D plot.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create sets for fixed and pulley nodes if not provided
    if fixed_nodes is None:
        fixed_nodes = set()
    else:
        fixed_nodes = set(np.atleast_1d(fixed_nodes))

    if pulley_nodes is None:
        pulley_nodes = set()
    else:
        pulley_nodes = set(np.atleast_1d(pulley_nodes))

    # Initialize node masses dictionary (placeholder)
    node_masses = {}
    print(f"kite_connectivity_arr shape: {kite_connectivity_arr.shape}")

    for conn in kite_connectivity_arr:
        i, j = int(conn[0]), int(conn[1])

        # Initialize masses if not already in dictionary
        if i not in node_masses:
            node_masses[i] = 0
        if j not in node_masses:
            node_masses[j] = 0

    # Create sets to track which elements are tubular frame, te_lines, or other noncompressive
    tubular_frame_nodes = set()
    te_line_nodes = set()
    pulley_line_nodes = set()

    # Line style mapping
    line_styles = {
        "default": {
            "color": "black",
            "linestyle": "-",
            "linewidth": 2.5,
            "label": "Tubular Frame",
        },
        "noncompressive": {"color": "green", "linestyle": "-", "linewidth": 1.5},
        "pulley": {
            "color": "purple",
            "linestyle": "-",
            "linewidth": 1.5,
            "label": "Pulley Lines",
        },
    }

    # Track which labels have been used
    used_labels = set()

    # First pass to identify TE lines and bridle lines
    # This is necessary because we need to know which noncompressive lines are TE lines before plotting
    for idx, conn in enumerate(kite_connectivity_arr):
        i, j = int(conn[0]), int(conn[1])

        # Get link type from linktype_arr if available
        if linktype_arr is not None and idx < len(linktype_arr):
            link_type = linktype_arr[idx]
            if hasattr(link_type, "value"):
                link_type = link_type.value
        else:
            link_type = "default"

        # Mark te_line_idx_list nodes (this is a placeholder - in actual code,
        # we would use the te_line_idx_list parameter to identify TE lines)
        # For now, we're just propagating the te_line_nodes set
        if str(link_type).lower() == "noncompressive":
            if i in te_line_nodes or j in te_line_nodes:
                te_line_nodes.add(i)
                te_line_nodes.add(j)

    # Plot connections with appropriate styling
    for idx, conn in enumerate(kite_connectivity_arr):
        i, j = int(conn[0]), int(conn[1])

        # Get k, c, and link_type from separate arrays if available
        k = float(k_arr[idx]) if k_arr is not None and idx < len(k_arr) else 0.0
        c = float(c_arr[idx]) if c_arr is not None and idx < len(c_arr) else 0.0

        if linktype_arr is not None and idx < len(linktype_arr):
            link_type = linktype_arr[idx]
            if hasattr(link_type, "value"):
                link_type = link_type.value
        else:
            link_type = "default"

        x_vals = [struc_nodes[i][0], struc_nodes[j][0]]
        y_vals = [struc_nodes[i][1], struc_nodes[j][1]]
        z_vals = [struc_nodes[i][2], struc_nodes[j][2]]

        # Default styling
        style = line_styles.get(
            str(link_type).lower(), {"color": "gray", "linestyle": "-", "linewidth": 1}
        )

        # Separate noncompressive elements into TE lines and bridle lines
        if str(link_type).lower() == "noncompressive":
            if i in te_line_nodes or j in te_line_nodes:
                style["color"] = "orange"
                if "Canopy TE" not in used_labels:
                    style["label"] = "Canopy TE"
                    used_labels.add("Canopy TE")
                else:
                    style.pop("label", None)

            else:
                style["color"] = "blue"
                if "Bridle Lines" not in used_labels:
                    style["label"] = "Bridle Lines"
                    used_labels.add("Bridle Lines")
                else:
                    style.pop("label", None)

        # Track nodes for tubular frame and pulley lines
        if str(link_type).lower() == "default":
            tubular_frame_nodes.add(i)
            tubular_frame_nodes.add(j)
            if "Tubular Frame" not in used_labels:
                used_labels.add("Tubular Frame")
            else:
                style.pop("label", None)

        if str(link_type).lower() == "pulley":
            pulley_line_nodes.add(i)
            pulley_line_nodes.add(j)
            if "Pulley Lines" not in used_labels:
                used_labels.add("Pulley Lines")
            else:
                style.pop("label", None)

        # Include damping in the label if requested
        if "label" in style and "damping" not in style["label"]:
            style["label"] += f" (k={k:.1f}, c={c:.2f})"

        # Plot the line
        ax.plot(x_vals, y_vals, z_vals, **style)

    # Create legend labels for nodes
    node_handles = []
    node_labels = []

    # Plot nodes - separate loop to ensure nodes are drawn on top of lines
    for i, point in enumerate(struc_nodes):
        # Plot the index of the node
        ax.text(
            point[0] + 0.02,
            point[1] + 0.02,
            point[2] + 0.02,
            str(i),
            color="black",
            fontsize=6,
        )
        if i in fixed_nodes:
            marker = ax.scatter(
                point[0],
                point[1],
                point[2],
                color="red",
                s=5,
                label="",  # We'll add to legend separately
            )
            if "Fixed Node" not in used_labels:
                node_handles.append(marker)
                node_labels.append("Fixed Node")
                used_labels.add("Fixed Node")
        elif i in pulley_nodes:
            marker = ax.scatter(
                point[0],
                point[1],
                point[2],
                color="purple",
                s=25,
                label="",  # We'll add to legend separately
            )
            if "Pulley Node" not in used_labels:
                node_handles.append(marker)
                node_labels.append("Pulley Node")
                used_labels.add("Pulley Node")
        else:
            marker = ax.scatter(
                point[0],
                point[1],
                point[2],
                color="black",
                s=8,
                label="",  # We'll add to legend separately
            )
            if "Free Node" not in used_labels:
                node_handles.append(marker)
                node_labels.append("Free Node")
                used_labels.add("Free Node")

    for idx, conn in enumerate(kite_connectivity_arr):
        i, j = int(conn[0]), int(conn[1])

        # Get coordinates
        p1 = np.array(struc_nodes[i])
        p2 = np.array(struc_nodes[j])

        # Midpoint for label
        midpoint = (p1 + p2) / 2

        # Get k value if available
        k_val = float(k_arr[idx]) if k_arr is not None and idx < len(k_arr) else 0.0

        # label = f"{idx}"
        # compute distance between p1 and p2
        distance = np.linalg.norm(p2 - p1)
        label = f"{1e3*distance:.1f} mm"

        # Add label slightly offset from midpoint
        offset = 0.02 * np.linalg.norm(p2 - p1)
        ax.text(
            midpoint[0] + offset,
            midpoint[1] + offset,
            midpoint[2] + offset,
            label,
            fontsize=6,
            color="blue",
        )
        if power_tape_index is not None and power_tape_index == idx:
            # Highlight the power tape line
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color="red",
                linestyle="-",
                linewidth=3,
                label="Power Tape",
            )
            if "Power Tape" not in used_labels:
                used_labels.add("Power Tape")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Kite Structure")

    # Equal aspect ratio
    if hasattr(struc_nodes, "max") and hasattr(struc_nodes, "min"):
        bb = struc_nodes.max(axis=0) - struc_nodes.min(axis=0)
        ax.set_box_aspect(bb)
    else:
        # If points is not a numpy array with max/min methods
        struc_nodes_arr = np.array(struc_nodes)
        bb = struc_nodes_arr.max(axis=0) - struc_nodes_arr.min(axis=0)
        ax.set_box_aspect(bb)

    # Add legend - use a separate legend for nodes
    # Get existing handles and labels from the lines
    handles, labels = ax.get_legend_handles_labels()

    # Combine with node handles and labels
    all_handles = handles + node_handles
    all_labels = labels + node_labels

    # Create legend outside the plot area to ensure visibility
    plt.legend(
        all_handles,
        all_labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
    )

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend

    plt.show()
