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

"""Named symbolic outputs for SystemModel.

These expressions are derived from the model context and are intended for
post-processing, constraints, and CasADi function extraction.
"""


def build_expression_registry(model):
    return {
        "force_aerodynamic": lambda: model.kite.force_aerodynamic(model),
        "force_gravity": lambda: model.kite.force_gravity_for(model),
        "force_gravity_wing": lambda: model.kite.force_gravity_wing_for(model),
        "force_gravity_kcu": lambda: model.kite.force_gravity_kcu_for(model),
        "force_tether_at_kite": lambda: model.tether.force_tether_at_kite_for(model),
        "drag_tether_at_kite": lambda: model.tether.drag_tether_at_kite_for(model),
        "force_gravity_tether_at_kite": lambda: (
            model.tether.force_gravity_tether_at_kite_for(model)
        ),
        "mass_tether": lambda: model.tether.mass_tether_for(model),
        "tension_kite": lambda: model.tether.tension_kite_for(model),
        "angle_of_attack": lambda: model.kite.angle_of_attack_for(model),
        "pitch_bridle": lambda: model.kite.pitch_bridle_for(model),
        "roll_bridle": lambda: model.kite.roll_bridle_for(model),
        "angle_roll_aerodynamic": lambda: model.kite.angle_roll_aerodynamic_for(model),
        "lift_coefficient": lambda: model.kite.lift_coefficient_for(model),
        "drag_coefficient": lambda: model.kite.drag_coefficient_for(model),
    }
