"""Re-export of :mod:`awetrim.aerostructural.case` under its old script name.

The helpers moved into the package on 2026-10-07. Scripts that put
``scripts/aerostructural`` on ``sys.path`` and ``import common`` keep working;
new code imports ``awetrim.aerostructural.case`` directly.
"""

from awetrim.aerostructural.case import *  # noqa: F401,F403
from awetrim.aerostructural.case import (  # noqa: F401
    CONFIG_DEFAULTS,
    DEFAULT_KITE_NAME,
    DEFAULT_STRUC_GEOMETRY_FILENAME,
    build_actuation_case_folder,
    build_system_model,
    configure_system_model_from_config,
    format_length_tag,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
    resolve_starting_rest_lengths,
    resolve_starting_struc_nodes,
)
