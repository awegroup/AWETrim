"""Tests for awetrim.aerostructural.billow.driver (config handling only)."""

from awetrim.aerostructural.billow import driver


def test_merge_config_merges_nested_and_copies():
    base = {"aerodynamic": {"a": 1, "b": 2}, "x": 1}
    merged = driver.merge_config(base, {"aerodynamic": {"b": 3}, "y": 2})
    assert merged == {"aerodynamic": {"a": 1, "b": 3}, "x": 1, "y": 2}
    assert base["aerodynamic"]["b"] == 2


def test_billow_defaults_select_the_billow_solver():
    assert driver.BILLOW_CONFIG_DEFAULTS["structural_solver"] == "billow"
    assert driver.BILLOW_CONFIG_DEFAULTS["power_tape_final_extension"] == 0.0
