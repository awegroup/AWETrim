"""AWETrim's own VSM settings -- the ONE place they are stated.

VSM's constructor defaults change between versions, and some of them are
model choices rather than numerics: VSM develop flipped ``is_aoa_corrected``
to True (quarter-chord force directions) on 2026-09-17, added the
Kutta-Joukowski force on the attached trailed vortex segments
(``is_with_attached_trailed_vortex_force``, default True) and replaced the
fixed relaxation factor by an automatic one. A bare ``Solver()`` therefore
follows whichever VSM checkout is installed. Every AWETrim code path builds
its VSM objects through :func:`make_vsm_solver` / :func:`vsm_body_kwargs`
instead, so all of them run one stated model and changing a value here
changes it everywhere.

The model: the VSM with its circulation from the 3/4-chord control point
and its force DIRECTIONS from the flow at the quarter chord
(``is_aoa_corrected=True``, the LL-Gaunaa model of Gaunaa, Li & Pirrung,
TORQUE 2026), the chord-regularised curved bound vortex and the
attached-trailed vortex force, on the VSM commit pyproject.toml pins
(develop 1201f6f, 2026-10-28). On inviscid wings the quarter-chord directions
reproduce the Trefftz-plane induced drag (straight wings exactly, an elliptic
one within 2 %; the 3/4-chord directions overshoot it by 20-40 %): VSM
62c53af, and the curved-wing verification of VSM PR #165 (on-blade over
Trefftz induced drag 0.99-1.01 for arcs, swept wings and the V3). Default in
AWETrim since 2026-11-03; results before that date ran the 3/4-chord
directions (``is_aoa_corrected=False``: CD 11-16 % higher on the LEI-V3,
CL < 0.3 % different), and before 2026-10-28 VSM 8825a12 without the
attached-trailed force (polar and trim shifts of 1-4 %).
The CasADi trim runs either law through VSM's own force-law kernels
(``VSM.core.panel_kernels``, VSM PR #166), so it follows this setting too.

An option the installed VSM does not have is dropped when the requested
value is what that VSM does anyway (``_ABSENT_MEANS``), and refused
otherwise, so asking for a feature never silently runs without it.
"""

from __future__ import annotations

import inspect
from typing import Any

#: ``VSM.core.Solver.Solver`` keyword arguments.
VSM_SOLVER_DEFAULTS: dict[str, Any] = {
    # 3/4-chord control point; "LLT" evaluates on the quarter chord.
    "aerodynamic_model_type": "VSM",
    "core_radius_fraction": 0.05,
    "max_iterations": 5000,
    "allowed_error": 1e-6,
    # Fixed under-relaxation (VSM develop: None = 0.8 x its stability bound).
    "relaxation_factor": 0.01,
    "gamma_loop_type": "base",
    "gamma_initial_distribution_type": "zero",
    "is_with_viscous_drag_correction": False,
    "mu": 1.81e-5,
    "rho": 1.225,
    # Force directions from the quarter-chord flow (True) or the 3/4-chord flow (False).
    "is_aoa_corrected": True,
    # Kutta-Joukowski force on the attached (chordwise) trailed vortex legs.
    "is_with_attached_trailed_vortex_force": True,
    "is_with_artificial_viscosity": False,
    "artificial_viscosity_factor": 0.035,
    "anderson_max_iterations": 1000,
    "anderson_fallback_to_base": False,
    "polar_interpolation": "linear",
}

#: ``VSM.core.BodyAerodynamics.BodyAerodynamics.instantiate`` keyword arguments.
VSM_BODY_DEFAULTS: dict[str, Any] = {
    "aerodynamic_center_location": 0.25,
    "control_point_location": 0.75,
    # Direction of the attached trailed legs: along the rib.
    "leg_direction": "rib",
}

#: What a VSM without the option does: requesting this value of a missing
#: option is a no-op, any other value is refused.
_ABSENT_MEANS: dict[str, Any] = {
    "is_with_attached_trailed_vortex_force": False,
    "leg_direction": "rib",
}


def _supported(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(target).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    out = {}
    for key, value in kwargs.items():
        if key in params:
            out[key] = value
        elif key in _ABSENT_MEANS and value == _ABSENT_MEANS[key]:
            continue
        else:
            raise TypeError(
                f"The installed VSM has no option {key!r} (AWETrim asks for "
                f"{key}={value!r}); install the VSM version pyproject.toml pins."
            )
    return out


def vsm_solver_kwargs(**overrides: Any) -> dict[str, Any]:
    """AWETrim's ``Solver`` keyword arguments with ``overrides`` applied."""
    return {**VSM_SOLVER_DEFAULTS, **overrides}


def make_vsm_solver(**overrides: Any) -> Any:
    """A VSM ``Solver`` on AWETrim's settings, ``overrides`` taking precedence."""
    from VSM.core.Solver import Solver

    return Solver(**_supported(Solver.__init__, vsm_solver_kwargs(**overrides)))


def vsm_body_kwargs(**overrides: Any) -> dict[str, Any]:
    """Keyword arguments for ``BodyAerodynamics.instantiate``: AWETrim's body
    settings plus ``overrides``, reduced to what the installed VSM accepts."""
    from VSM.core.BodyAerodynamics import BodyAerodynamics

    return _supported(BodyAerodynamics.instantiate, {**VSM_BODY_DEFAULTS, **overrides})
