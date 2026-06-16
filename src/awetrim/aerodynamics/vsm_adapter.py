# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

"""VSMAeroModelAdapter — KiteAeroModel implementation backed by the Vortex Step Method."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from awetrim.aerodynamics.vsm_quasi_steady import (
    DEFAULT_BOUNDS_LOWER,
    DEFAULT_BOUNDS_UPPER,
    DEFAULT_TRANSFORMATION_C_FROM_VSM,
    solve_vsm_quasi_steady_trim,
)
from awetrim.system.protocols import FlightCondition
from awetrim.system.state import State


class VSMAeroModelAdapter:
    """Wraps the VSM quasi-steady trim solver to implement ``KiteAeroModel``.

    Parameters
    ----------
    config_folder:
        Path to the kite data folder containing ``system.yaml``,
        ``aero_geometry.yaml``, and optionally ``struc_geometry.yaml``.
    n_panels:
        Number of VSM panels per half-span.
    center_of_gravity:
        3-vector [x, y, z] in the course frame [m].
    reference_point:
        3-vector [x, y, z] for VSM reference point [m].
    x_guess:
        Initial guess for the five VSM trim unknowns
        [speed_tangential, roll_deg, pitch_deg, yaw_deg, course_rate].
    spanwise_panel_distribution:
        Panel distribution type passed to ``BodyAerodynamics.instantiate``.
    """

    def __init__(
        self,
        config_folder: Path | str,
        *,
        n_panels: int = 18,
        center_of_gravity: Optional[np.ndarray] = None,
        reference_point: Optional[np.ndarray] = None,
        x_guess: Optional[np.ndarray] = None,
        spanwise_panel_distribution: str = "uniform",
    ):
        self.config_folder = Path(config_folder).expanduser().resolve()
        self.n_panels = n_panels
        self.center_of_gravity = (
            np.asarray(center_of_gravity, dtype=float)
            if center_of_gravity is not None
            else np.array([0.5, 0.0, 5.0], dtype=float)
        )
        self.reference_point = (
            np.asarray(reference_point, dtype=float)
            if reference_point is not None
            else np.array([0.0, 0.0, 0.0], dtype=float)
        )
        self.x_guess = (
            np.asarray(x_guess, dtype=float)
            if x_guess is not None
            else np.array([25.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        )
        self.spanwise_panel_distribution = spanwise_panel_distribution

        self.aero_geometry_path = self.config_folder / "aero_geometry.yaml"
        self.struc_geometry_path = self.config_folder / "struc_geometry.yaml"

        # Read mass from system.yaml via factory
        from awetrim.system.factory import create_system_model_from_yaml
        _sm = create_system_model_from_yaml(self.config_folder / "system.yaml")
        self.mass = float(_sm.kite.mass_wing)

    # ------------------------------------------------------------------
    # KiteAeroModel interface
    # ------------------------------------------------------------------

    def solve_quasi_steady(self, condition: FlightCondition) -> State:
        """Run a VSM trim solve for the given flight condition.

        Builds a lightweight ``SystemModel`` from ``condition``, instantiates
        the VSM body from ``aero_geometry.yaml``, calls
        ``solve_vsm_quasi_steady_trim``, and maps the result to a ``State``.
        """
        system = self._build_system_model(condition)
        body = self._build_body()

        result, _ = solve_vsm_quasi_steady_trim(
            body_aero=body,
            center_of_gravity=self.center_of_gravity,
            reference_point=self.reference_point,
            system_model=system,
            x_guess=self.x_guess,
            bounds_lower=DEFAULT_BOUNDS_LOWER,
            bounds_upper=DEFAULT_BOUNDS_UPPER,
            transformation_c_from_vsm=DEFAULT_TRANSFORMATION_C_FROM_VSM,
        )

        opt_x = np.asarray(result["opt_x"], dtype=float)

        gamma_dist = result.get("gamma_distribution")
        alpha_dist = result.get("alpha_at_ac")
        f_dist = result.get("F_distribution")

        return State(
            distance_radial=condition.distance_radial,
            angle_elevation=condition.angle_elevation,
            angle_azimuth=condition.angle_azimuth,
            angle_course=condition.angle_course,
            speed_radial=condition.speed_radial,
            speed_tangential=float(opt_x[0]),
            input_depower=condition.input_depower,
            input_steering=condition.input_steering,
            tension_tether_ground=float(result["tether_force"]),
            angle_roll=float(np.deg2rad(opt_x[1])),
            angle_pitch=float(np.deg2rad(opt_x[2])),
            angle_yaw=float(np.deg2rad(opt_x[3])),
            angle_of_attack=float(np.deg2rad(result["aoa_deg"])),
            lift_coefficient=float(result["cl"]),
            drag_coefficient=float(result["cd"]),
            side_force_coefficient=float(result.get("cfy", 0.0)),
            speed_apparent_wind=float(result["Umag"]),
            lift_distribution=np.asarray(f_dist) if f_dist is not None else None,
            circulation_distribution=(
                np.asarray(gamma_dist) if gamma_dist is not None else None
            ),
            angle_of_attack_sections=(
                np.asarray(alpha_dist) if alpha_dist is not None else None
            ),
        )

    def get_aero_coefficients(self, state: State) -> Dict[str, float]:
        """Return {'CL': ..., 'CD': ..., 'CS': ...} from a solved ``State``."""
        return {
            "CL": state.lift_coefficient,
            "CD": state.drag_coefficient,
            "CS": state.side_force_coefficient or 0.0,
        }

    def compute_forces(self, state: State) -> Dict[str, Any]:
        """Return tether tension and mechanical power from a solved ``State``."""
        tension = state.tension_tether_ground or 0.0
        return {
            "tension_tether_ground": tension,
            "mechanical_power": tension * (state.speed_radial or 0.0),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_model(self, condition: FlightCondition):
        """Build a SystemModel configured for the given flight condition."""
        from awetrim.system.system_model import SystemModel
        from awetrim.system.tether import RigidLumpedTether

        system = SystemModel(tether=RigidLumpedTether(diameter=0.0))
        system.kite.mass_wing = self.mass
        system.angle_elevation = condition.angle_elevation
        system.angle_azimuth = condition.angle_azimuth
        system.angle_course = condition.angle_course
        system.speed_radial = condition.speed_radial
        system.distance_radial = condition.distance_radial
        system.wind.speed_wind_ref = condition.wind_speed
        system.timeder_speed_tangential = 0.0
        system.timeder_speed_radial = 0.0
        system.timeder_angle_course = 0.0
        return system

    def _build_body(self):
        """Instantiate a VSM ``BodyAerodynamics`` from ``aero_geometry.yaml``."""
        from VSM.core.BodyAerodynamics import BodyAerodynamics

        with open(self.aero_geometry_path, "r", encoding="utf-8") as f:
            body_config = yaml.safe_load(f)

        self._resolve_csv_paths(body_config)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(body_config, tmp)
            tmp_path = tmp.name

        try:
            bridle = (
                str(self.struc_geometry_path)
                if self.struc_geometry_path.exists()
                else None
            )
            body = BodyAerodynamics.instantiate(
                n_panels=self.n_panels,
                file_path=tmp_path,
                spanwise_panel_distribution=self.spanwise_panel_distribution,
                bridle_path=bridle,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return body

    def _resolve_csv_paths(self, body_config: dict) -> None:
        """Resolve relative CSV polar paths to absolute paths in-place."""
        airfoils = body_config.get("wing_airfoils", {})
        for row in airfoils.get("data", []):
            if (
                len(row) >= 3
                and isinstance(row[2], dict)
                and "csv_file_path" in row[2]
            ):
                p = row[2]["csv_file_path"]
                if not Path(p).is_absolute():
                    row[2]["csv_file_path"] = str(
                        (self.aero_geometry_path.parent / p).resolve()
                    )
