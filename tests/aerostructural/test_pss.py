def test_aerostructural_import_does_not_import_pss():
    import awetrim.aerostructural as aerostructural

    assert hasattr(aerostructural, "PssKineticDampingSolver")


class _FakeLink:
    """Minimal stand-in for a PSS SpringDamper: only ``k`` is touched."""

    def __init__(self, k):
        self.k = k


class _FakeParticleSystem:
    def __init__(self, stiffnesses):
        self.springdampers = [_FakeLink(k) for k in stiffnesses]


def test_adapt_stiffnesses_only_stiffens_elements_over_the_bound():
    from awetrim.aerostructural.pss.structural_pss import adapt_stiffnesses

    # 0.5% (under), 2% (over), NaN (no rest length -> skipped)
    updated, n_updated, max_seen = adapt_stiffnesses(
        [1000.0, 1000.0, 1000.0],
        [0.005, 0.02, float("nan")],
        max_elongation=0.01,
        factor=2.0,
        max_stiffness=5.0e5,
    )

    assert n_updated == 1
    assert max_seen == 0.02
    assert list(updated) == [1000.0, 2000.0, 1000.0]


def test_adapt_stiffnesses_stops_at_the_stiffness_ceiling():
    from awetrim.aerostructural.pss.structural_pss import adapt_stiffnesses

    updated, n_updated, _ = adapt_stiffnesses(
        [5.0e5], [0.05], max_elongation=0.01, factor=2.0, max_stiffness=5.0e5
    )

    # Already at the ceiling: nothing changes, so the loop can still converge.
    assert n_updated == 0
    assert list(updated) == [5.0e5]


def test_adapt_stiffnesses_leaves_ineligible_elements_alone():
    """Bridle lines are excluded by default: their k is E*A/l0, not a free knob."""
    from awetrim.aerostructural.pss.structural_pss import adapt_stiffnesses

    # Both elongate past the bound, but only element 0 (wing) is eligible.
    updated, n_updated, max_seen = adapt_stiffnesses(
        [1000.0, 1000.0],
        [0.02, 0.05],
        element_indices=range(1),
        max_elongation=0.01,
    )

    assert n_updated == 1
    assert max_seen == 0.02  # the bridle's 5% is not the adapted set's maximum
    assert list(updated) == [2000.0, 1000.0]


def test_stiffness_ramp_factor_runs_from_the_start_factor_to_one():
    from awetrim.aerostructural.pss.structural_pss import stiffness_ramp_factor

    first = stiffness_ramp_factor(0, 10, 0.1)
    middle = stiffness_ramp_factor(4, 10, 0.1)
    last = stiffness_ramp_factor(9, 10, 0.1)

    assert 0.1 <= first < middle < last
    assert last == 1.0
    # Past the ramp, and when disabled, the target stiffness stands.
    assert stiffness_ramp_factor(50, 10, 0.1) == 1.0
    assert stiffness_ramp_factor(0, 0, 0.1) == 1.0


def test_get_and_set_stiffnesses_round_trip():
    from awetrim.aerostructural.pss.structural_pss import (
        get_stiffnesses,
        set_stiffnesses,
    )

    psystem = _FakeParticleSystem([1.0, 2.0, 3.0])
    set_stiffnesses(psystem, [10.0, 20.0, 30.0])

    assert list(get_stiffnesses(psystem)) == [10.0, 20.0, 30.0]
