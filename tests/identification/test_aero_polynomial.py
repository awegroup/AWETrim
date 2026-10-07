"""Tests for the aerodynamic polynomial identification module.

Covers the candidate-term library, least-squares fitting, BIC model selection,
the CD absolute-value basis, and a serialisation round-trip that confirms the
emitted rom_config terms are evaluated identically by the ROM consumer
(``Kite.aerodynamic_force_coefficients_for``).
"""

import numpy as np
import pytest

from awetrim.identification import aero_polynomial as ap
from awetrim.identification.aero_dataset import aerodynamic_roll


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 500
    data = {
        "alpha": rng.uniform(-0.1, 0.25, n),
        "u_s": rng.uniform(-0.3, 0.3, n),
        "u_p": rng.uniform(0.0, 0.4, n),
        "v_a": rng.uniform(10.0, 30.0, n),
    }
    return data, rng


# ── Term library ────────────────────────────────────────────────────────────
def test_generate_candidate_terms_excludes_intercept_and_dedupes():
    terms = ap.generate_candidate_terms(("alpha", "u_s"), max_degree=2, max_vars_per_term=2)
    keys = [ap.term_key(t) for t in terms]
    assert len(keys) == len(set(keys)), "terms must be unique"
    assert all(t for t in terms), "no empty (intercept) term"
    # alpha, u_s, alpha^2, u_s^2, alpha*u_s
    assert {ap.term_label(t) for t in terms} == {"alpha", "u_s", "alpha^2", "u_s^2", "alpha*u_s"}


def test_max_vars_per_term_limits_interactions():
    terms = ap.generate_candidate_terms(
        ("alpha", "u_s", "u_p"), max_degree=2, max_vars_per_term=1
    )
    assert all(len(t) == 1 for t in terms), "max_vars_per_term=1 -> pure powers only"


def test_squared_interactions_add_alpha2_times_rest():
    terms = ap.generate_candidate_terms(
        ("alpha", "u_s", "u_p", "v_a"),
        max_degree=2,
        max_vars_per_term=2,
        include_squared_interactions=("alpha",),
    )
    labels = {ap.term_label(t) for t in terms}
    assert {"alpha^2*u_s", "alpha^2*u_p", "alpha^2*v_a"} <= labels
    assert "alpha^2*alpha" not in labels and "alpha^3" not in labels


def test_squared_interaction_term_is_recovered():
    rng = np.random.default_rng(3)
    n = 600
    data = {
        "alpha": rng.uniform(-0.05, 0.2, n),
        "u_s": rng.uniform(-0.3, 0.3, n),
        "u_p": rng.uniform(1.5, 1.9, n),
        "v_a": rng.uniform(10.0, 30.0, n),
    }
    # CL with an alpha^2 * v_a modulation that a degree-2 library cannot express.
    y = 0.05 + 6.0 * data["alpha"] - 0.5 * data["alpha"] ** 2 * data["v_a"]
    fit = ap.select_model(
        data, y, target="CL", max_degree=2, max_vars_per_term=2,
        include_squared_interactions=("alpha",),
    )
    coef = {ap.term_label(pm): c for pm, c in fit.terms}
    assert "alpha^2*v_a" in coef
    assert coef["alpha^2*v_a"] == pytest.approx(-0.5, abs=1e-2)


# ── Design matrix ─────────────────────────────────────────────────────────────
def test_design_matrix_shape_and_abs_basis():
    data = {"alpha": np.array([-1.0, 2.0]), "u_s": np.array([1.0, -3.0])}
    terms = [{"alpha": 1}, {"u_s": 1}]
    A = ap.design_matrix(data, terms, include_intercept=True, abs_basis=False)
    assert A.shape == (2, 3)
    assert np.allclose(A[:, 0], 1.0)
    A_abs = ap.design_matrix(data, terms, include_intercept=True, abs_basis=True)
    assert np.allclose(A_abs[:, 1], np.abs(data["alpha"]))
    assert np.allclose(A_abs[:, 2], np.abs(data["u_s"]))


# ── Fitting / selection ───────────────────────────────────────────────────────
def test_fit_recovers_known_coefficients(synthetic_data):
    data, rng = synthetic_data
    a = data["alpha"]
    y = 0.05 + 6.1 * a - 10.0 * a**2 - 0.1 * data["u_p"]
    fit = ap.fit_terms(
        data, y, [{"alpha": 1}, {"alpha": 2}, {"u_p": 1}], target="CL"
    )
    assert fit.intercept == pytest.approx(0.05, abs=1e-6)
    coef = {ap.term_label(pm): c for pm, c in fit.terms}
    assert coef["alpha"] == pytest.approx(6.1, abs=1e-4)
    assert coef["alpha^2"] == pytest.approx(-10.0, abs=1e-4)
    assert coef["u_p"] == pytest.approx(-0.1, abs=1e-4)
    assert fit.metrics["r2"] > 0.999


def test_select_model_picks_correct_terms(synthetic_data):
    data, rng = synthetic_data
    a = data["alpha"]
    y = 0.05 + 6.1 * a - 10.0 * a**2 - 0.1 * data["u_p"] + rng.normal(0, 1e-3, len(a))
    fit = ap.select_model(data, y, target="CL", max_degree=2, max_vars_per_term=2)
    selected = {ap.term_label(pm) for pm, _ in fit.terms}
    assert {"alpha", "alpha^2", "u_p"} <= selected
    assert fit.cv_rmse < 0.05


def test_cd_is_fitted_in_the_plain_basis(synthetic_data):
    data, _ = synthetic_data
    # An odd alpha drag term stays odd: no |.| (and no kink at alpha = 0).
    y = 0.11 + 0.05 * data["alpha"] + 1.5 * data["alpha"] ** 2
    fit = ap.select_model(data, y, target="CD", max_degree=2, max_vars_per_term=1)
    assert fit.abs_basis is False
    coef = {ap.term_label(pm): c for pm, c in fit.terms}
    assert coef.get("alpha") == pytest.approx(0.05, abs=1e-3)


def test_abs_terms_of_a_rom_file_evaluate_and_round_trip():
    aero = {"params": {"CD0": 0.1}, "coefficients": {"CD": [
        {"var": "u_s", "power": 1, "coef": 0.02, "abs": True},
        {"var": "alpha", "power": 1, "coef": 0.05},
    ]}}
    fit = ap.fits_from_rom_aerodynamics(aero)["CD"]
    data = {"u_s": np.array([-0.5, 0.5]), "alpha": np.array([-0.1, -0.1])}
    assert np.allclose(fit.predict(data), 0.1 + 0.02 * 0.5 - 0.005)
    back = ap.build_rom_aerodynamics([fit])["coefficients"]["CD"]
    assert [e.get("abs", False) for e in back] == [True, False]


# ── Serialisation round-trip (fit -> rom_config -> ROM evaluation) ─────────────
def test_serialization_matches_rom_evaluation(synthetic_data):
    data, _ = synthetic_data
    a = data["alpha"]
    y = 0.05 + 6.1 * a - 10.0 * a**2 + 0.3 * data["u_s"] * data["v_a"]
    fit = ap.fit_terms(
        data, y,
        [{"alpha": 1}, {"alpha": 2}, {"u_s": 1, "v_a": 1}],
        target="CL",
    )
    aero = ap.build_rom_aerodynamics([fit])
    assert aero["model"] == "coeffs"
    assert aero["params"]["CL0"] == pytest.approx(fit.intercept)

    # Evaluate the serialised CL block with the SAME logic the ROM uses and
    # confirm it reproduces the fit prediction sample-by-sample.
    sample = {k: np.asarray(v)[:5] for k, v in data.items()}
    rom_pred = np.full(5, aero["params"]["CL0"], dtype=float)
    for i in range(5):
        variables = {k: sample[k][i] for k in sample}
        total = aero["params"]["CL0"]
        for term in aero["coefficients"]["CL"]:
            # Same term evaluation as Kite.aerodynamic_force_coefficients_for
            # ("coeffs" model): product of var**power, or a single var**power.
            if "vars" in term:
                value = 1.0
                for var, power in term["vars"].items():
                    value *= variables[var] ** power
            else:
                value = variables[term["var"]] ** term.get("power", 1)
            total += term["coef"] * value
        rom_pred[i] = total
    assert np.allclose(rom_pred, fit.predict(sample), atol=1e-9)


def test_phi_a_serialized_under_phi_a_key(synthetic_data):
    data, _ = synthetic_data
    y = 0.0 + 1.2 * data["u_s"]
    fit = ap.fit_terms(data, y, [{"u_s": 1}], target="phi_a")
    aero = ap.build_rom_aerodynamics([fit])
    assert "phi_a" in aero["coefficients"]
    assert aero["params"]["phi_a0"] == pytest.approx(fit.intercept, abs=1e-9)
    assert aero["coefficients"]["phi_a"][0]["var"] == "u_s"


# ── phi_a reconstruction helper ───────────────────────────────────────────────
def test_aerodynamic_roll_pure_lift_is_zero():
    radial = np.array([0.0, 0.0, 1.0])
    va = np.array([10.0, 0.0, 0.0])  # lift_dir is along +z
    force = np.array([0.0, 0.0, 50.0])  # pure lift
    assert aerodynamic_roll(force, va, radial) == pytest.approx(0.0, abs=1e-9)


def test_aerodynamic_roll_pure_side_is_ninety_degrees():
    radial = np.array([0.0, 0.0, 1.0])
    va = np.array([10.0, 0.0, 0.0])
    # lift_dir = +z, side_dir = lift x va_unit = z x x = +y
    force = np.array([0.0, 30.0, 0.0])
    assert aerodynamic_roll(force, va, radial) == pytest.approx(np.pi / 2, abs=1e-9)


def test_grouped_folds_never_split_a_group():
    groups = np.repeat(np.arange(12), 7)
    folds = ap._fold_indices(len(groups), 4, seed=1, groups=groups)
    assert sorted(np.concatenate(folds)) == list(range(len(groups)))
    for test_idx in folds:
        in_test = set(groups[test_idx])
        rest = np.setdiff1d(np.arange(len(groups)), test_idx)
        assert in_test.isdisjoint(groups[rest])


def test_cv_criterion_stops_at_the_generating_terms():
    rng = np.random.default_rng(3)
    n_groups, per = 40, 9
    groups = np.repeat(np.arange(n_groups), per)
    u_p = np.repeat(rng.uniform(1.6, 2.2, n_groups), per)
    alpha = np.repeat(rng.uniform(0.0, 0.15, n_groups), per) + np.tile(
        np.linspace(-0.05, 0.05, per), n_groups
    )
    v_a = np.repeat(rng.uniform(13, 25, n_groups), per)
    data = {"alpha": alpha, "u_p": u_p, "v_a": v_a}
    y = 0.1 + 5.0 * alpha - 8.0 * alpha**2 - 0.2 * u_p + rng.normal(0, 1e-3, len(alpha))
    fit = ap.select_model(
        data, y, target="CL", regressors=("alpha", "u_p", "v_a"),
        criterion="cv", cv_groups=groups, min_relative_improvement=0.01,
    )
    labels = {ap.term_label(pm) for pm, _ in fit.terms}
    assert {"alpha", "alpha^2", "u_p"} <= labels
    assert not any("v_a" in label for label in labels)
    assert fit.cv_rmse < 2e-3


def test_unknown_criterion_is_rejected():
    data = {"alpha": np.linspace(0, 0.1, 10)}
    with pytest.raises(ValueError):
        ap.select_model(data, data["alpha"], target="CL", regressors=("alpha",),
                        criterion="aic")


def test_backward_pass_drops_a_term_made_redundant_later():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(30), 8)
    x = rng.uniform(0.0, 1.0, len(groups))
    z = rng.uniform(0.0, 1.0, len(groups))
    data = {"x": x, "z": z}
    # x*z is the single best first pick, but x and z together explain y.
    y = 1.0 * x + 1.0 * z + rng.normal(0, 1e-3, len(x))
    candidates = [{"x": 1, "z": 1}, {"x": 1}, {"z": 1}]
    fwd = ap.select_model(data, y, target="CL", candidate_terms=candidates,
                          regressors=("x", "z"), criterion="cv", cv_groups=groups,
                          min_relative_improvement=0.01)
    both = ap.select_model(data, y, target="CL", candidate_terms=candidates,
                           regressors=("x", "z"), criterion="cv", cv_groups=groups,
                           min_relative_improvement=0.01, backward=True)
    assert len(both.terms) <= len(fwd.terms)
    assert {ap.term_label(pm) for pm, _ in both.terms} == {"x", "z"}


def test_rom_block_round_trips_to_the_same_predictions():
    rng = np.random.default_rng(7)
    data = {"alpha": rng.uniform(0, 0.2, 50), "u_p": rng.uniform(1.6, 2.2, 50)}
    fit = ap.fit_terms(data, 0.1 + data["alpha"] - 0.3 * data["alpha"] * data["u_p"],
                       [{"alpha": 1}, {"alpha": 1, "u_p": 1}], target="CD",
                       regressors=("alpha", "u_p"))
    back = ap.fits_from_rom_aerodynamics(ap.build_rom_aerodynamics([fit]))["CD"]
    assert np.allclose(back.predict(data), fit.predict(data))


def test_selection_path_records_every_step():
    rng = np.random.default_rng(11)
    groups = np.repeat(np.arange(20), 6)
    x = rng.uniform(0, 1, len(groups))
    fit = ap.select_model({"x": x}, 2 * x + rng.normal(0, 1e-3, len(x)), target="CL",
                          regressors=("x",), criterion="cv", cv_groups=groups,
                          min_relative_improvement=0.01)
    assert [step for step, _ in fit.selection_path] == ["1", "+x"]
    assert fit.selection_path[1][1] < fit.selection_path[0][1]
