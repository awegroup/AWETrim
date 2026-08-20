import numpy as np
import pytest

from awetrim.kinematics.parametrized_patterns import (
    LOBE_HANDOVER_PHASE,
    create_pattern_from_dict,
    full_cycle_angles,
    full_cycle_n_loops_for_half_figures,
    make_bspline_path_parameters_from_named_curve,
    named_curve_angles,
    reelin_bump,
    reelin_control_point_mask,
)


def _bump_path_parameters(M, bump_indices, base=0.3, peak=1.0):
    """Periodic-spline params with a flat elevation except a bump (reel-in arc)."""
    C_beta = np.full(M, base)
    C_beta[list(bump_indices)] = peak
    return {
        "M": M,
        "C_phi": np.zeros(M),
        "C_beta": C_beta,
        "s_init": 0.0,
        "s_final": 1.0,
        "downloops": True,
    }


def test_reelin_control_point_mask_flags_elevation_peak_support():
    """The mask flags the control points shaping the elevation peak (the
    reel-in arc of a full-cycle spline) plus their B-spline support, and
    leaves the far figure-eight points untouched."""
    M = 12
    mask = reelin_control_point_mask(_bump_path_parameters(M, [5, 6]))
    assert mask.shape == (M,) and mask.dtype == bool
    assert mask[5] and mask[6]  # the peak itself
    assert not mask[0] and not mask[11]  # far from the arc
    assert 0 < mask.sum() < M  # neither empty nor all-flagged


def test_reelin_control_point_mask_wraps_at_the_periodic_seam():
    """A peak at the s=0 seam flags neighbours on BOTH sides of the wrap."""
    M = 12
    mask = reelin_control_point_mask(_bump_path_parameters(M, [0]))
    assert mask[0] and mask[1] and mask[11]
    assert not mask[6]


def test_full_cycle_angles_psi0_shifts_phase_and_preserves_periodicity():
    """psi0 offsets the figure phase without breaking exact periodicity.

    The curvature tuner in fit_periodic_cycle_config relies on psi0 to realign
    the reel-in fade with a fast part of the figure-eight; a constant phase
    offset must keep the curve 1-periodic (psi advances by 2*pi*n_loops over
    the period regardless of psi0) and reduce to the psi0=0 curve when zero.
    """
    s = np.linspace(0.0, 1.0, 400, endpoint=False)
    kwargs = dict(n_loops=3, reelout_fraction=0.65, beta0=0.35, beta_amp0=0.14,
                  az_amp0=0.36, beta_reelin_peak=1.2, az_reelin_amp=-0.5,
                  ramp_fraction=0.45, downloops=True)

    az_default, el_default = full_cycle_angles(s, **kwargs)
    az_zero, el_zero = full_cycle_angles(s, psi0=0.0, **kwargs)
    assert np.allclose(az_default, az_zero) and np.allclose(el_default, el_zero)

    az, el = full_cycle_angles(np.array([0.0, 1.0]), psi0=1.3, **kwargs)
    assert np.isclose(az[0], az[1]) and np.isclose(el[0], el[1])

    az_shift, _ = full_cycle_angles(s, psi0=1.3, **kwargs)
    assert not np.allclose(az_shift, az_zero)


def test_reelin_bump_center_moves_the_window_and_wraps_the_seam():
    """reelin_center relocates the window without changing its width; the
    default 0.5 is unchanged, and the window wraps across the periodic seam
    (any centre is valid, taken mod 1)."""
    s = np.linspace(0.0, 1.0, 4000, endpoint=False)
    f = 0.65
    h = 0.5 * (1.0 - f)

    b_default = reelin_bump(s, reelout_fraction=f, ramp_fraction=0.45)
    b_mid = reelin_bump(s, reelout_fraction=f, ramp_fraction=0.45, reelin_center=0.5)
    assert np.allclose(b_default, b_mid)

    for center in (0.0, 0.1, h, 0.35, 1.0 - h):  # incl. seam-wrapping windows
        b = reelin_bump(
            s, reelout_fraction=f, ramp_fraction=0.45, reelin_center=center
        )
        # circular centroid of the bump must sit at the requested centre
        ang = 2.0 * np.pi * s
        centroid = (
            np.arctan2((b * np.sin(ang)).sum(), (b * np.cos(ang)).sum())
            / (2.0 * np.pi)
        ) % 1.0
        d_centroid = abs(centroid - center)
        assert min(d_centroid, 1.0 - d_centroid) < 1e-2
        # zero outside the (circular) window keeps the curve periodic
        d = np.abs((s - center + 0.5) % 1.0 - 0.5)
        assert np.allclose(b[d > h + 1e-9], 0.0)
        # same window width regardless of the centre
        assert np.isclose(b.sum(), b_mid.sum(), rtol=1e-6)


def test_full_cycle_angles_reelin_center_shifts_arc_and_keeps_periodicity():
    """A shifted reel-in window moves the elevation arc in s while the curve
    stays exactly periodic at the seam; the default centre is unchanged."""
    s = np.linspace(0.0, 1.0, 2000, endpoint=False)
    f = 0.65
    h = 0.5 * (1.0 - f)
    kwargs = dict(n_loops=3, reelout_fraction=f, beta0=0.35, beta_amp0=0.14,
                  az_amp0=0.36, beta_reelin_peak=1.2, az_reelin_amp=-0.5,
                  ramp_fraction=0.45, downloops=True)

    az_default, el_default = full_cycle_angles(s, **kwargs)
    az_mid, el_mid = full_cycle_angles(s, reelin_center=0.5, **kwargs)
    assert np.allclose(az_default, az_mid) and np.allclose(el_default, el_mid)

    center = 1.0 - h  # reel-in ends at the seam -> s = 0 is the reel-out start
    _, el = full_cycle_angles(s, reelin_center=center, **kwargs)
    peak_mask = el > 0.9 * kwargs["beta_reelin_peak"]
    assert abs(float(s[peak_mask].mean()) - center) < 0.02
    # steady figure-eights around mid-period, where the window used to be
    assert np.all(el[np.abs(s - 0.5) < 0.05] < kwargs["beta0"] + 0.2)

    az_seam, el_seam = full_cycle_angles(
        np.array([0.0, 1.0]), reelin_center=center, **kwargs
    )
    assert np.isclose(az_seam[0], az_seam[1]) and np.isclose(el_seam[0], el_seam[1])

    # window centred ON the seam (wraps): s = 0 is the top of the reel-in and
    # the figure-eights fly at mid-period, where the window used to be
    _, el_wrap = full_cycle_angles(s, reelin_center=0.0, **kwargs)
    assert el_wrap[0] > 0.9 * kwargs["beta_reelin_peak"]
    assert np.all(el_wrap[np.abs(s - 0.5) < 0.05] < kwargs["beta0"] + 0.2)


def test_full_cycle_angles_pinned_handover_phases_and_descent_bow():
    """psi_entry/psi_exit pin the figure phase at the reel-in window edges
    (entry at the climbing centre crossing, re-entry heading the OTHER way,
    i.e. the first lobe lands on the other side), the curve stays exactly
    periodic, and the descent-only bow keeps the whole climb half of the
    reel-in on the zero-azimuth meridian."""
    s = np.linspace(0.0, 1.0, 3000, endpoint=False)
    f = 0.65
    h = 0.5 * (1.0 - f)
    c = 0.0  # window wraps the seam, like the generator's default
    kwargs = dict(n_loops=5, reelout_fraction=f, beta0=0.35, beta_amp0=0.14,
                  az_amp0=0.36, beta_reelin_peak=1.2, az_reelin_amp=-0.5,
                  ramp_fraction=0.45, reelin_center=c, downloops=True)

    # None/None + "sym" reproduces the legacy psi0 curve exactly
    az_legacy, el_legacy = full_cycle_angles(s, psi0=0.7, **kwargs)
    az_none, el_none = full_cycle_angles(
        s, psi0=0.7, psi_entry=None, psi_exit=None, bow_shape="sym", **kwargs
    )
    assert np.allclose(az_legacy, az_none) and np.allclose(el_legacy, el_none)

    az, el = full_cycle_angles(
        s, psi_entry=0.0, psi_exit=np.pi, bow_shape="descent", **kwargs
    )

    # exactly periodic at the seam (which sits INSIDE the wrapped window)
    az_seam, el_seam = full_cycle_angles(
        np.array([0.0, 1.0]), psi_entry=0.0, psi_exit=np.pi,
        bow_shape="descent", **kwargs
    )
    assert np.isclose(az_seam[0], az_seam[1], atol=1e-9)
    assert np.isclose(el_seam[0], el_seam[1], atol=1e-9)

    # entry edge (s = c - h): azimuth 0, moving toward +az and climbing;
    # exit edge (s = c + h): azimuth 0, moving toward -az -> other side
    ds = s[1] - s[0]
    i_in = int(round(((c - h) % 1.0) / ds))
    i_out = int(round(((c + h) % 1.0) / ds))
    d_az_in = (az[(i_in + 1) % s.size] - az[i_in - 1]) / (2 * ds)
    d_el_in = (el[(i_in + 1) % s.size] - el[i_in - 1]) / (2 * ds)
    d_az_out = (az[(i_out + 1) % s.size] - az[i_out - 1]) / (2 * ds)
    assert abs(az[i_in]) < 1e-6 and d_az_in > 0 and d_el_in > 0
    assert abs(az[i_out]) < 1e-6 and d_az_out < 0

    # one-sided bow: the reel-in azimuth stays on the az_reelin_amp side
    # (never crosses to the other), unlike the antisymmetric "sym" bow
    d = (s - c + 0.5) % 1.0 - 0.5
    r_s = kwargs["ramp_fraction"] * (1.0 - f)
    plateau = np.abs(d) < h - r_s
    assert plateau.any()
    assert az[plateau].max() < 1e-9  # negative az_reelin_amp -> -az side only
    assert az[plateau].min() < -0.1
    # ... and the path never stalls: the elevation plateau is flat, so the
    # bow must carry nonzero path speed everywhere (no cusp at the top)
    q = np.column_stack(
        (np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el))
    )
    speed = np.linalg.norm(np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0), axis=1)
    assert speed.min() > 0.1 * speed.mean()

    # pinning both phases needs the plateau to hide the phase correction
    with pytest.raises(ValueError, match="ramp_fraction"):
        full_cycle_angles(
            s, psi_entry=0.0, psi_exit=np.pi,
            **{**kwargs, "ramp_fraction": 0.5},
        )


def test_full_cycle_angles_lobe_bow_freezes_phase_and_lands_on_lobe():
    """bow_shape="lobe": the reel-in is one giant lobe. The figure phase
    freezes through the window (no wiggle: the plateau azimuth is pure bow,
    monotone), the climb sits on the zero meridian, the whole window stays on
    the az_reelin_amp side, the descent lands on the lobe extreme and the kite
    then flies the LOWER half of that lobe, the curve is exactly periodic,
    never stalls, and the shape needs both handover phases pinned."""
    s = np.linspace(0.0, 1.0, 4000, endpoint=False)
    f, rf, c = 0.65, 0.48, 0.0
    h = 0.5 * (1.0 - f)
    a, b, beta0 = 0.36, 0.14, 0.35
    kwargs = dict(n_loops=4, reelout_fraction=f, beta0=beta0, beta_amp0=b,
                  az_amp0=a, beta_reelin_peak=1.17, az_reelin_amp=-0.43,
                  ramp_fraction=rf, reelin_center=c, downloops=True,
                  psi_entry=2.0 * np.pi - 0.3, psi_exit=1.5 * np.pi,
                  bow_shape="lobe")
    az, el = full_cycle_angles(s, **kwargs)

    # exactly periodic at the seam (inside the wrapped window)
    az_seam, el_seam = full_cycle_angles(np.array([0.0, 1.0]), **kwargs)
    assert np.isclose(az_seam[0], az_seam[1], atol=1e-9)
    assert np.isclose(el_seam[0], el_seam[1], atol=1e-9)

    d = (s - c + 0.5) % 1.0 - 0.5
    window = np.abs(d) <= h
    r_s = rf * (1.0 - f)
    plateau = np.abs(d) < h - r_s
    assert plateau.any()
    # Frozen phase: in the plateau the figure term is exactly zero, so the
    # azimuth is the bow alone -- monotone toward the -az side, no wiggle.
    az_pl = az[plateau][np.argsort(d[plateau])]
    assert np.all(np.diff(az_pl) <= 1e-12)
    assert np.allclose(el[plateau], kwargs["beta_reelin_peak"])
    # Whole window on the az_reelin_amp (negative) side; descent reaches
    # the bow azimuth, and the climb starts on the meridian: where the
    # elevation lift is half way up the entry ramp the azimuth is already
    # within a quarter of the figure amplitude of zero.
    assert az[window].max() < 1e-9
    assert az[window].min() < -0.4
    entry_ramp = window & (d < 0) & (np.abs(d) > h - r_s)
    i_half = entry_ramp & (np.abs(el - 0.5 * (beta0 + kwargs["beta_reelin_peak"])) < 0.02)
    assert i_half.any() and np.all(np.abs(az[i_half]) < 0.25 * a)

    # Lands on the lobe: the descent heads DOWN onto the left extreme (az
    # near -az_amp0 at the lobe mid-height) and the figure then continues
    # through the lower half of the lobe -- right after the window exit the
    # elevation is below the base elevation and the kite heads toward +az.
    ds = s[1] - s[0]
    i_out = int(round(((c + h) % 1.0) / ds))
    i_after = i_out + int(0.01 / ds)
    assert el[i_after] < beta0
    assert az[i_after + 1] > az[i_after]
    # figure phase at the exit edge = psi_exit + LOBE_HANDOVER_PHASE
    psi_edge = kwargs["psi_exit"] + LOBE_HANDOVER_PHASE
    assert np.isclose(az[i_out], a * np.sin(psi_edge), atol=2e-3)
    assert np.isclose(el[i_out], beta0 + b * np.sin(2 * psi_edge), atol=2e-3)

    # never stalls (the bow carries speed across the flat top)
    q = np.column_stack(
        (np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el))
    )
    speed = np.linalg.norm(np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0), axis=1)
    assert speed.min() > 0.1 * speed.mean()

    # needs both handover phases pinned
    with pytest.raises(ValueError, match="lobe"):
        full_cycle_angles(s, **{**kwargs, "psi_exit": None})


def test_periodic_bspline_local_support_is_exact_on_its_interval_and_sparse():
    """``PeriodicBSpline.local_support(k)`` keeps only the 4 basis terms of
    knot interval k: exact (value, d/ds, d2/ds2) wherever
    ``knot_interval(s) == k`` -- including on the knots, where neighbouring
    truncations agree -- and structurally dependent on 4 coefficients
    instead of M (the sparse-Jacobian property the NLP relies on)."""
    import casadi as ca

    from awetrim.kinematics.parametrized_patterns import PeriodicBSpline

    M = 17
    rng = np.random.default_rng(3)
    C_phi = ca.DM(rng.normal(size=M))
    C_beta = ca.DM(rng.normal(size=M))
    for downloops in (True, False):
        full = PeriodicBSpline(M=M, C_phi=C_phi, C_beta=C_beta, s_init=0.0,
                               s_final=2.0, downloops=downloops)
        s_sym = ca.MX.sym("s")

        def _derivs(pat):
            az = pat.azimuth(0.0, s_sym)
            el = pat.elevation(0.0, s_sym)
            return ca.Function("f", [s_sym], [
                az, ca.gradient(az, s_sym), ca.hessian(az, s_sym)[0],
                el, ca.gradient(el, s_sym), ca.hessian(el, s_sym)[0],
            ])

        f_full = _derivs(full)
        knots = np.arange(M + 1) * (2.0 / M)  # s exactly on every knot
        samples = np.r_[rng.uniform(0.0, 2.0, 120), knots, [0.0, 2.0, 3.7, -0.4]]
        for s_val in samples:
            k = full.knot_interval(s_val)
            assert 0 <= k < M
            loc = full.local_support(k)
            assert loc.support_interval == k
            a = np.asarray(f_full(s_val)).ravel()
            b = np.asarray(_derivs(loc)(s_val)).ravel()
            assert np.allclose(a, b, atol=1e-9), (downloops, s_val, k)

    # vectorised knot lookup matches the scalar one
    ks = full.knot_interval(samples)
    assert ks.shape == samples.shape
    assert all(int(ks[i]) == full.knot_interval(samples[i]) for i in range(len(samples)))

    # structural sparsity: 4 coefficients per coordinate, wherever s is
    Csym = ca.MX.sym("C", M)
    loc = PeriodicBSpline(M=M, C_phi=Csym, C_beta=Csym, s_init=0.0, s_final=2.0,
                          support_interval=0)  # wraps: C[M-1], C[0], C[1], C[2]
    assert ca.jacobian(loc.azimuth(0.0, s_sym), Csym).nnz() == 4
    dense = PeriodicBSpline(M=M, C_phi=Csym, C_beta=Csym, s_init=0.0, s_final=2.0)
    assert ca.jacobian(dense.azimuth(0.0, s_sym), Csym).nnz() == M
    with pytest.raises(ValueError, match="support_interval"):
        PeriodicBSpline(M=M, C_phi=Csym, C_beta=Csym, s_init=0.0, s_final=2.0,
                        support_interval=M)


def test_create_pattern_from_dict_rejects_unsupported_type():
    """A type with no constructor (e.g. cst_helix, the cycle-config default)
    must raise a clear ValueError listing supported types, not a KeyError."""
    with pytest.raises(ValueError, match="Unknown or unsupported pattern type"):
        create_pattern_from_dict("cst_helix", {})


def test_create_pattern_from_dict_reports_missing_params():
    """A supported type with missing params reports them explicitly."""
    with pytest.raises(ValueError, match="Missing required parameters"):
        create_pattern_from_dict("spline_open", {"M": 6})


def test_named_curve_angles_support_lissajous_and_helix():
    s = np.linspace(0.0, 2.0 * np.pi, 5)

    phi_lissajous, beta_lissajous = named_curve_angles(
        s,
        curve_type="lissajous",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )
    phi_helix, beta_helix = named_curve_angles(
        s,
        curve_type="helix",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )

    assert phi_lissajous.shape == s.shape
    assert beta_lissajous.shape == s.shape
    assert phi_helix.shape == s.shape
    assert beta_helix.shape == s.shape
    assert not np.allclose(beta_lissajous, beta_helix)


def test_named_curve_angles_lemniscate_is_amplitude_matched_figure_eight():
    """The Bernoulli lemniscate is a drop-in for the lissajous eight: same
    azimuth/elevation amplitudes, starting at the centre crossing."""
    s = np.linspace(0.0, 2.0 * np.pi, 2001)
    az_amp0, beta0, beta_amp0 = 0.32, 0.3, 0.15

    phi, beta = named_curve_angles(
        s,
        curve_type="lemniscate",
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
    )

    assert np.isclose(np.max(np.abs(phi)), az_amp0, atol=1e-6)
    assert np.isclose(np.max(np.abs(beta - beta0)), beta_amp0, atol=1e-6)
    # starts at the centre crossing and closes periodically there
    assert np.isclose(phi[0], 0.0, atol=1e-9)
    assert np.isclose(beta[0], beta0, atol=1e-9)
    assert np.isclose(phi[-1], 0.0, atol=1e-6)

    phi_liss, _ = named_curve_angles(
        s, curve_type="lissajous", az_amp0=az_amp0, beta0=beta0, beta_amp0=beta_amp0
    )
    assert not np.allclose(phi, phi_liss)


def test_named_curve_angles_rejects_unknown_curve():
    with pytest.raises(ValueError, match="lissajous"):
        named_curve_angles(np.array([0.0, 1.0]), curve_type="spiral")


@pytest.mark.parametrize("curve_type", ["lissajous", "lemniscate"])
def test_periodic_uploop_fit_matches_runtime(curve_type):
    """A reversed (uploop) fit must reproduce the curve once rebuilt the way
    create_pattern_from_dict does at sim time. Regression for the singular
    reversed-grid (u in [-1, 0]) basis matrix and the missing ``downloops``
    passthrough that left uploops evaluated in the downloop sense."""
    az_amp0, beta0, beta_amp0 = 0.32, 0.3, 0.15
    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type="periodic",
        M=10,
        r0=230.0,
        s_init=0.0,
        s_final=2.0 * np.pi,
        n_fit=200,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=False,
    )
    assert path_parameters["downloops"] is False

    pattern = create_pattern_from_dict("spline_periodic", path_parameters)

    s = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=True)
    az_target, el_target = named_curve_angles(
        s,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=False,
    )
    az_fit = np.array([float(pattern.azimuth(230.0, sv)) for sv in s])
    el_fit = np.array([float(pattern.elevation(230.0, sv)) for sv in s])

    assert np.max(np.abs(az_fit - az_target)) < 5e-2
    assert np.max(np.abs(el_fit - el_target)) < 5e-2


def test_make_periodic_bspline_path_parameters_are_pattern_ready():
    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type="periodic",
        M=10,
        r0=230.0,
        s_init=0.0,
        s_final=2.0 * np.pi,
        n_fit=80,
        curve_type="helix",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )

    assert path_parameters["M"] == 10
    assert len(path_parameters["C_phi"]) == 10
    assert len(path_parameters["C_beta"]) == 10

    pattern = create_pattern_from_dict("spline_periodic", path_parameters)

    assert pattern.M == 10


def test_make_open_bspline_path_parameters_are_pattern_ready():
    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type="open",
        M=6,
        r0=230.0,
        s_init=0.0,
        s_final=1.0,
        n_fit=40,
        curve_type="lissajous",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )

    assert path_parameters["M"] == 6
    assert len(path_parameters["C_phi"]) == 6
    assert len(path_parameters["C_beta"]) == 6

    pattern = create_pattern_from_dict("spline_open", path_parameters)

    assert pattern.M == 6


def test_full_cycle_n_loops_for_half_figures_is_wysiwyg():
    """The user-facing loop count is half figure-eights (lobes) VISIBLE during
    reel-out: the helper inverts the pinned-phase snap (counting azimuth
    centre crossings in the reel-out confirms it), flips psi_entry by pi for
    odd counts (mirrored climb peel-off; the descent landing psi_exit is
    never touched), and the returned non-integer n_loops keeps the curve
    exactly periodic at the seam."""
    f = 0.65
    seed_entry = 2.0 * np.pi - 0.3
    psi_exit = 1.5 * np.pi
    shape = dict(reelout_fraction=f, beta0=0.35, beta_amp0=0.14, az_amp0=0.36,
                 beta_reelin_peak=1.1, az_reelin_amp=-0.36, ramp_fraction=0.45,
                 reelin_center=0.0, psi_exit=psi_exit, bow_shape="lobe",
                 downloops=True)
    s = np.linspace(0.0, 1.0, 6000, endpoint=False)
    h = 0.5 * (1.0 - f)
    reelout = (s > h) & (s < 1.0 - h)  # window is centred on the seam

    for n_halves in range(1, 11):
        n_loops, psi_entry = full_cycle_n_loops_for_half_figures(
            n_halves, reelout_fraction=f, psi_entry=seed_entry,
            psi_exit=psi_exit, bow_shape="lobe",
        )
        # parity: odd counts mirror the entry handover by pi, even keep it
        expected = (seed_entry + (np.pi if n_halves % 2 else 0.0)) % (
            2.0 * np.pi
        )
        assert np.isclose(psi_entry, expected)

        az, _ = full_cycle_angles(s, n_loops=n_loops, psi_entry=psi_entry,
                                  **shape)
        # each half-figure is bounded by azimuth centre crossings
        crossings = int(np.count_nonzero(np.diff(np.sign(az[reelout]))))
        assert abs(crossings - n_halves) <= 1

        az_seam, el_seam = full_cycle_angles(
            np.array([0.0, 1.0]), n_loops=n_loops, psi_entry=psi_entry,
            **shape,
        )
        assert np.isclose(az_seam[0], az_seam[1], atol=1e-9)
        assert np.isclose(el_seam[0], el_seam[1], atol=1e-9)

    # monotone: more requested halves never means fewer internal figures
    fitted = [
        full_cycle_n_loops_for_half_figures(
            n, reelout_fraction=f, psi_entry=seed_entry, psi_exit=psi_exit,
            bow_shape="lobe",
        )[0]
        for n in range(1, 11)
    ]
    assert np.all(np.diff(fitted) > 0)


def test_full_cycle_n_loops_for_half_figures_free_phase_and_validation():
    # free-phase fallback: integer n_loops (periodicity), psi_entry untouched
    n_loops, entry = full_cycle_n_loops_for_half_figures(6, reelout_fraction=0.7)
    assert n_loops == 4 and isinstance(n_loops, int) and entry is None
    n_loops, entry = full_cycle_n_loops_for_half_figures(
        1, reelout_fraction=0.7, psi_entry=0.3
    )
    assert n_loops == 1 and entry == 0.3

    with pytest.raises(ValueError, match="positive"):
        full_cycle_n_loops_for_half_figures(0, reelout_fraction=0.7)
