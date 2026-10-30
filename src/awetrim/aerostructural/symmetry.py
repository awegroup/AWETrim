"""Mirror symmetry of a structural geometry about the plane y = 0.

Both geometry readers (``wireframe.structural_geometry_io.main`` and
``coupled.read_struc_geometry_yaml.main``) check what they return: every node
must have a mirror partner at its mirrored position, every element a mirror
element with the same rest length, stiffness and link type, and mirrored nodes
the same mass. An unsteered kite on an asymmetric geometry settles in an
asymmetric shape and turns with zero steering -- a few millimetres of node
position and one 29 mm short trailing-edge wire did exactly that in the V3Kite
emulation (2026-10-28), invisible until the kite turned. A steered snapshot is
asymmetric by design, so the readers only WARN (``is_with_symmetry_check:
false`` silences it).
"""

from __future__ import annotations

import logging

import numpy as np

#: Mirror position mismatch [m] that counts as asymmetric.
POSITION_TOL_M = 1e-4
#: Relative mismatch of rest length or stiffness that counts as asymmetric.
RELATIVE_TOL = 1e-6
#: Relative mass mismatch that counts as asymmetric: the full-model reader
#: derives masses from node coordinates, rounded in the file (~1e-5 apart).
MASS_RELATIVE_TOL = 1e-3
#: A node whose nearest mirror image is further than this [m] has no partner.
NO_PARTNER_M = 0.05


def mirror_partners(nodes) -> tuple[np.ndarray, np.ndarray]:
    """``(partner, mismatch)``: for every node, the index of the node nearest
    its mirror image (itself for a node on the plane) and that distance [m]."""
    nodes = np.asarray(nodes, dtype=float)
    mirrored = nodes * np.array([1.0, -1.0, 1.0])
    distance = np.linalg.norm(nodes[:, None, :] - mirrored[None, :, :], axis=2)
    partner = np.argmin(distance, axis=1)
    return partner, distance[np.arange(len(nodes)), partner]


def check_mirror_symmetry(
    nodes, connectivity, rest_lengths, stiffnesses=None, link_types=None, masses=None,
    position_tol=POSITION_TOL_M, rtol=RELATIVE_TOL,
) -> list[str]:
    """Every way the geometry departs from mirror symmetry, one line each
    (empty: symmetric)."""
    partner, mismatch = mirror_partners(nodes)
    issues = []
    for i in np.where(mismatch > NO_PARTNER_M)[0]:
        issues.append(f"node {i} has no mirror partner (nearest mirror image {mismatch[i]:.3f} m away)")
    for i in np.where((mismatch > position_tol) & (mismatch <= NO_PARTNER_M))[0]:
        if i == partner[i]:
            issues.append(f"node {i}: {1e3 * mismatch[i] / 2:.2f} mm off the symmetry plane")
        elif i < partner[i]:
            issues.append(f"nodes {i} / {partner[i]}: positions {1e3 * mismatch[i]:.2f} mm off mirror")
    paired = mismatch <= NO_PARTNER_M

    def differs(a, b, tol=rtol):
        return abs(a - b) > tol * max(abs(a), abs(b), 1e-12)

    if masses is not None:
        masses = np.asarray(masses, dtype=float)
        for i in np.where(paired)[0]:
            j = partner[i]
            if i < j and differs(masses[i], masses[j], MASS_RELATIVE_TOL):
                issues.append(f"nodes {i} / {j}: masses {masses[i]:.6g} / {masses[j]:.6g} kg")

    connectivity = np.asarray(connectivity, dtype=int)
    rest_lengths = np.asarray(rest_lengths, dtype=float)
    by_nodes = {}
    for e, (a, b) in enumerate(connectivity[:, :2]):
        by_nodes.setdefault(frozenset((int(a), int(b))), []).append(e)
    for e, (a, b) in enumerate(connectivity[:, :2]):
        if not (paired[a] and paired[b]):
            continue
        mirror = by_nodes.get(frozenset((int(partner[a]), int(partner[b]))))
        if not mirror:
            issues.append(f"element {e} ({a}-{b}) has no mirror element ({partner[a]}-{partner[b]})")
            continue
        f = mirror[0]
        if f < e:
            continue
        if differs(rest_lengths[e], rest_lengths[f]):
            issues.append(f"elements {e} ({a}-{b}) / {f}: rest lengths {rest_lengths[e]:.6g} / {rest_lengths[f]:.6g} m")
        if stiffnesses is not None and differs(float(stiffnesses[e]), float(stiffnesses[f])):
            issues.append(f"elements {e} / {f}: stiffnesses {float(stiffnesses[e]):.6g} / {float(stiffnesses[f]):.6g}")
        if link_types is not None and str(link_types[e]) != str(link_types[f]):
            issues.append(f"elements {e} / {f}: link types {link_types[e]} / {link_types[f]}")
    return issues


def warn_if_asymmetric(
    nodes, connectivity, rest_lengths, stiffnesses=None, link_types=None, masses=None, config=None,
) -> list[str]:
    """The readers' check: log one warning listing the asymmetries, unless
    ``config["is_with_symmetry_check"]`` is false. Returns the issues."""
    if not (config or {}).get("is_with_symmetry_check", True):
        return []
    issues = check_mirror_symmetry(nodes, connectivity, rest_lengths, stiffnesses, link_types, masses)
    if issues:
        shown = "\n  ".join(issues[:12]) + (f"\n  ... and {len(issues) - 12} more" if len(issues) > 12 else "")
        logging.warning(
            "Structural geometry is not mirror-symmetric about y = 0 (%d issue(s)); an unsteered kite "
            "will turn. Expected for a steered snapshot, otherwise fix the input "
            "(is_with_symmetry_check: false silences this):\n  %s",
            len(issues), shown,
        )
    return issues
