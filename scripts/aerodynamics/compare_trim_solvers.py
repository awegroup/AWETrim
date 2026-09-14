"""Compare the CasADi trim against the NumPy least-squares trims on one kite.

Solves the tetherless and the Williams-tether quasi-steady trim with
``solve_vsm_quasi_steady_trim`` / ``solve_vsm_qs_trim_with_williams_tether``
(production base loop and the casadi_newton inner loop) and with
``awetrim.aerodynamics.trim_casadi.CasadiTrim``, and prints the trim states,
coefficients, residuals and wall-clock times side by side, unsteered and with
an applied roll moment.

Example:
    python scripts/aerodynamics/compare_trim_solvers.py --n-panels 27
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    add_common_arguments,
    build_body,
    build_system_model,
    parsed_common,
)

logging.disable(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(ap)
    ap.add_argument("--roll-moment", type=float, default=150.0, help="Applied roll moment [N m] of the steered case.")
    ap.add_argument("--av", action="store_true", help="Artificial viscosity on (all solvers).")
    args = ap.parse_args()

    from awetrim.aerodynamics.kcu_drag import KcuDragModel
    from awetrim.aerodynamics.trim_casadi import CasadiTrim, CasadiTrimOptions
    from awetrim.aerodynamics.vsm_quasi_steady import (
        solve_vsm_qs_trim_with_williams_tether,
        solve_vsm_quasi_steady_trim,
    )
    from awetrim.system.williams_tether import WilliamsTether

    body, _ = build_body(args)
    values = parsed_common(args)
    system_model = build_system_model(args)
    kcu = KcuDragModel.from_system_model(system_model)
    has_williams = isinstance(getattr(system_model, "tether", None), WilliamsTether)
    common = dict(
        center_of_gravity=values["center_of_gravity"],
        reference_point=values["reference_point"],
        x_guess=values["x_guess"],
        bounds_lower=values["bounds_lower"],
        bounds_upper=values["bounds_upper"],
        include_gravity=args.include_gravity,
        moment_tolerance=args.moment_tolerance,
        is_with_artificial_viscosity=args.av,
        kcu_drag=kcu,
    )

    def numpy_trim(tether: bool, loop: str, tol: float, moment: float):
        kw = dict(common, gamma_loop=loop, gamma_tolerance=tol)
        if moment:
            kw["applied_moment_nm"] = np.array([moment, 0.0, 0.0])
        t0 = time.perf_counter()
        if tether:
            res, _ = solve_vsm_qs_trim_with_williams_tether(body_aero=body, system_model=system_model, **kw)
        else:
            res, _ = solve_vsm_quasi_steady_trim(body_aero=body, system_model=system_model, **kw)
        res["_time"] = time.perf_counter() - t0
        return res

    def casadi_trim(tether: bool, moment: float, bridle_rotates: bool = True):
        t0 = time.perf_counter()
        trim = CasadiTrim(
            body, system_model, values["center_of_gravity"], values["reference_point"], kcu_drag=kcu,
            applied_moment_nm=(np.array([moment, 0.0, 0.0]) if moment else None),
            bounds_lower=values["bounds_lower"], bounds_upper=values["bounds_upper"],
            options=CasadiTrimOptions(
                tether_model="williams" if tether else None, is_with_artificial_viscosity=args.av,
                include_gravity=args.include_gravity, moment_tolerance=args.moment_tolerance,
                bridle_rotates_with_body=bridle_rotates,
            ),
        )
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = trim.solve(values["x_guess"])
        res["_time"] = time.perf_counter() - t0
        res["_build"] = t_build
        return res

    def row(name, res, tether):
        x = np.asarray(res["opt_x"])
        extra = f" L_tether={res['williams_tether_length']:8.3f} m" if tether else ""
        diag = res.get("casadi_trim")
        its = f" newton_its={diag['iterations']:3d} passes={len(diag['passes'])} build={res['_build']:.2f}s" if diag else ""
        print(
            f"  {name:26s} t={res['_time']:6.2f}s  v={x[0]:9.5f} roll={x[1]:9.5f} pitch={x[2]:9.5f} "
            f"yaw={x[3]:9.5f} chi_dot={x[4]:10.3e}  CL={res['cl']:.5f} CD={res['cd']:.5f}  "
            f"|cm|max={np.max(np.abs(np.asarray(res['cm']))):.1e}{extra}{its}"
        )

    def report(ref, res, tether):
        dx = np.abs(np.asarray(res["opt_x"]) - np.asarray(ref["opt_x"]))
        print(
            f"      vs production: |dv|={dx[0]:.2e} m/s |droll|={dx[1]:.2e} |dpitch|={dx[2]:.2e} "
            f"|dyaw|={dx[3]:.2e} deg |dchi_dot|={dx[4]:.2e}, dCL={res['cl'] - ref['cl']:+.2e}, dCD={res['cd'] - ref['cd']:+.2e}"
        )
        if tether:
            print(
                f"      tether length: production {ref['williams_tether_length']:.4f} m, CasADi "
                f"{res['williams_tether_length']:.4f} m; ground miss CasADi {np.linalg.norm(res['williams_ground_residual']):.2e} m"
            )
        passes = res["casadi_trim"]["passes"]
        print("      wake passes (newton its, residual after rebuild):", [(p.get("iterations"), p.get("residual_after_wake_rebuild")) for p in passes])

    # The NumPy Williams trim has no applied-moment argument: steered case tetherless only.
    cases = [(False, 0.0), (False, args.roll_moment)] + ([(True, 0.0)] if has_williams else [])
    for tether, moment in cases:
        label = "Williams tether" if tether else "tetherless"
        print(f"\n=== {label} trim, applied roll moment {moment:g} N m, AV {'on' if args.av else 'off'} ===")
        ref = numpy_trim(tether, "base", 1e-6, moment)
        row("numpy base 1e-6 (production)", ref, tether)
        row("numpy casadi_newton 1e-10", numpy_trim(tether, "casadi_newton", 1e-10, moment), tether)
        res = casadi_trim(tether, moment)
        row("CasADi trim (rigid bridle)", res, tether)
        report(ref, res, tether)
        res = casadi_trim(tether, moment, bridle_rotates=False)
        row("CasADi trim (legacy bridle)", res, tether)
        report(ref, res, tether)


if __name__ == "__main__":
    main()
