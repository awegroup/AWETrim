"""Solver-free helpers of examples/3_rom_identification/run_identification_sweep.py:
the chain list (what --list prints) and the built tape length it starts from,
read off the structure file without building a structural model."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
KITE = "LEI-V3-KITE"


@pytest.fixture(scope="module")
def sweep():
    path = REPO / "examples/3_rom_identification/run_identification_sweep.py"
    try:
        spec = importlib.util.spec_from_file_location("scripts_identification_sweep", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # its dataclasses look their module up
        spec.loader.exec_module(module)
        return module
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"cannot import {path}: {exc}")


@pytest.fixture(scope="module")
def wireframe_spec(sweep):
    _, spec = sweep.ds.identification_settings(
        REPO / "data" / KITE / "identification.yaml", "sweep", "wireframe")
    return spec


def test_built_power_tape_length_reads_the_structure_file(sweep, wireframe_spec):
    # the LEI-V3 structure.yml is built at 1.90 m of depower tape
    assert sweep.built_power_tape_length(wireframe_spec, KITE) == pytest.approx(1.90)


def test_build_chains_from_the_built_length(sweep, wireframe_spec):
    built = sweep.built_power_tape_length(wireframe_spec, KITE)
    chains = sweep.build_chains(wireframe_spec, built)
    n_winds = len(wireframe_spec["wind_speeds"])
    n_per_wind = len(wireframe_spec["depower"]["targets"]) + len(
        wireframe_spec["steering"]["power_tape_lengths"])
    assert len(chains) == n_winds * n_per_wind
    # per wind, only the first depower chain holds the built state itself
    built_states = [c.name for c in chains if any(s.power_tape_length is None for s in c.steps)]
    assert len(built_states) == n_winds
    assert all(c.steps[0].power_tape_length is None for c in chains if c.name in built_states)
    # a depower chain ends at its target
    target = wireframe_spec["depower"]["targets"][0]
    assert chains[0].steps[-1].power_tape_length == pytest.approx(target)
