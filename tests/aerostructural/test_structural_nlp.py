"""Tests for the exact minimum-energy structural inner solver (structural_nlp)."""

import numpy as np
import pytest

from awetrim.aerostructural.pss import structural_nlp, structural_pss


def _make_config():
    return {
        "is_with_initial_point_velocity": False,
        "structural_pss": {
            "dt": 0.005,
            "n_internal_time_steps": 10,
            "abs_tol": 1e-50,
            "rel_tol": 1e-5,
            "max_iter": 50,
            "fixed_point_indices": [0],
        },
    }


def _instantiate(nodes, conn, l0, k, linktypes, pulley_dict=None, m=1.0):
    config = _make_config()
    n = len(nodes)
    psystem, *_ = structural_pss.instantiate(
        config,
        [np.array(p, dtype=float) for p in nodes],
        np.full(n, m),
        np.asarray(conn, dtype=int),
        np.asarray(l0, dtype=float),
        np.asarray(k, dtype=float),
        np.zeros(len(conn)),
        np.asarray(linktypes),
        pulley_dict or {},
    )
    return psystem, config


def _residual_at_free_nodes(f_int, f_ext, fixed=(0,)):
    res = np.asarray(f_int).reshape(-1, 3) + np.asarray(f_ext).reshape(-1, 3)
    res = res.copy()
    for i in fixed:
        res[i] = 0.0
    return res


def test_contract_and_force_balance_with_slack_line():
    # chain: 0 (fixed) --default-- 1 --noncompressive-- 2 --default-- 3,
    # plus a noncompressive 1-3 with a LONG rest length that must end slack.
    nodes = [(0, 0, 0), (0.2, 0.1, 1), (0.5, 0.6, 1.8), (-0.4, -0.1, 2.4)]
    conn = [(0, 1), (1, 2), (2, 3), (1, 3)]
    l0 = [1.0, 1.0, 0.9, 3.0]
    k = [1e4, 5e3, 5e3, 5e3]
    linktypes = ["default", "noncompressive", "default", "noncompressive"]
    psystem, config = _instantiate(nodes, conn, l0, k, linktypes)

    solver = structural_nlp.NlpStructuralSolver.from_psystem(
        psystem, conn, [], config["structural_pss"]
    )
    f_ext = np.zeros((4, 3))
    f_ext[2] = [3.0, 12.0, 20.0]
    f_ext[3] = [-2.0, -5.0, 15.0]

    out = solver(psystem, f_ext.reshape(-1), config["structural_pss"])
    assert isinstance(out, tuple) and len(out) == 4
    psystem_out, converged, struc_nodes, f_int = out
    assert converged
    assert struc_nodes.shape == (4, 3)
    assert np.asarray(f_int).shape == (12,)

    # equilibrium at the free nodes (anchor term is 1e-3 N/m -> sub-mN here)
    res = _residual_at_free_nodes(f_int, f_ext)
    assert np.max(np.abs(res)) < 5e-3

    # fixed node did not move; solved positions written back into the psystem
    assert np.allclose(struc_nodes[0], nodes[0])
    assert np.allclose(
        np.array([p.x for p in psystem_out.particles]), struc_nodes
    )

    # the long noncompressive line must end slack (shorter than rest length)
    length_13 = np.linalg.norm(struc_nodes[1] - struc_nodes[3])
    assert length_13 < 3.0


def test_pulley_pair_shares_one_stretch():
    # rope 0-1-2 over a pulley at node 1; instantiate() splits the rest length
    # per arm and structural_nlp must recombine the pair into one stretch.
    nodes = [(0, 0, 0), (0.3, 0.5, 1.0), (-0.2, 1.2, 0.2), (0.4, 2.0, 1.0)]
    # a fourth element pins the pulley node's slide direction along the rope
    # (zero-stiffness otherwise), as the surrounding bridle does on the kite
    conn = [(0, 1), (1, 2), (2, 3), (1, 3)]
    total_l0 = 2.2
    len_01 = np.linalg.norm(np.subtract(nodes[1], nodes[0]))
    len_12 = np.linalg.norm(np.subtract(nodes[2], nodes[1]))
    share = total_l0 / (len_01 + len_12)
    pulley_dict = {
        "0": np.array([1, 2, share * len_12, share * len_01, 0]),
        "1": np.array([1, 0, share * len_01, share * len_12, 0]),
    }
    l0 = [total_l0, total_l0, 1.1, 1.45]
    k = [4e3, 4e3, 6e3, 5e3]
    linktypes = ["pulley", "pulley", "default", "default"]
    psystem, config = _instantiate(nodes, conn, l0, k, linktypes, pulley_dict)

    solver = structural_nlp.NlpStructuralSolver.from_psystem(
        psystem, conn, [0, 1], config["structural_pss"]
    )
    f_ext = np.zeros((4, 3))
    f_ext[1] = [0.0, 2.0, -8.0]
    f_ext[2] = [0.0, 5.0, -3.0]
    f_ext[3] = [0.0, 9.0, 6.0]

    _, converged, struc_nodes, f_int = solver(
        psystem, f_ext.reshape(-1), config["structural_pss"]
    )
    assert converged
    res = _residual_at_free_nodes(f_int, f_ext)
    assert np.max(np.abs(res)) < 5e-3


def test_matches_pss_equilibrium():
    # both solvers must land on the same equilibrium of the same springs
    nodes = [(0, 0, 0), (0.2, 0.1, 1), (0.5, 0.6, 1.8), (-0.4, -0.1, 2.4)]
    conn = [(0, 1), (1, 2), (2, 3)]
    l0 = [1.0, 1.0, 0.9]
    k = [2e3, 1e3, 1.5e3]
    linktypes = ["default", "default", "default"]
    f_ext = np.zeros((4, 3))
    f_ext[1] = [0.0, 4.0, 6.0]
    f_ext[2] = [0.0, -3.0, 10.0]
    f_ext[3] = [1.0, 2.0, 5.0]

    psystem_a, config = _instantiate(nodes, conn, l0, k, linktypes)
    solver = structural_nlp.NlpStructuralSolver.from_psystem(
        psystem_a, conn, [], config["structural_pss"]
    )
    _, _, nodes_nlp, _ = solver(
        psystem_a, f_ext.reshape(-1), config["structural_pss"]
    )

    psystem_b, config_b = _instantiate(nodes, conn, l0, k, linktypes)
    config_b["structural_pss"]["n_internal_time_steps"] = 4000
    nodes_pss = None
    for _ in range(40):
        psystem_b, _, nodes_pss, f_int_b = structural_pss.run_pss(
            psystem_b, f_ext.reshape(-1), config_b["structural_pss"]
        )
        res = _residual_at_free_nodes(f_int_b, f_ext)
        if np.max(np.abs(res)) < 1e-3:
            break
    assert np.max(np.abs(nodes_nlp - nodes_pss)) < 1e-3


def test_resolve_structural_solver_dispatch():
    nodes = [(0, 0, 0), (0.2, 0.1, 1), (0.5, 0.6, 1.8), (-0.4, -0.1, 2.4)]
    conn = [(0, 1), (1, 2), (2, 3)]
    psystem, config = _instantiate(
        nodes, conn, [1.0, 1.0, 0.9], [1e3, 1e3, 1e3],
        ["default", "default", "default"],
    )

    name, solve = structural_nlp.resolve_structural_solver(
        config["structural_pss"], psystem, conn, []
    )
    assert name == "pss" and solve is structural_pss.run_pss

    config["structural_pss"]["solver"] = "nlp"
    name, solve = structural_nlp.resolve_structural_solver(
        config["structural_pss"], psystem, conn, []
    )
    assert name == "nlp"
    assert isinstance(solve, structural_nlp.NlpStructuralSolver)

    config["structural_pss"]["solver"] = "bogus"
    with pytest.raises(ValueError):
        structural_nlp.resolve_structural_solver(
            config["structural_pss"], psystem, conn, []
        )
