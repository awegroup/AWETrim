# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Write ``scripts/server/openapi.yaml`` from the FastAPI app.

FastAPI only documents the happy path plus the 422 of request validation, so
this script also declares the error responses the endpoints actually raise
(see the ``HTTPException``s in ``awetrim/server/app.py``) and splits the
``/step`` reply into 200 (``wait=true``) and 202 (``wait=false``).

Run from the project root after changing ``app.py`` or ``schemas.py``:

    python scripts/server/generate_openapi.py
"""

from pathlib import Path

import yaml

from awetrim.server.app import create_app

OUTPUT = Path(__file__).with_name("openapi.yaml")

HEADER = """\
# Generated file — do not edit by hand.
# Regenerate with: python scripts/server/generate_openapi.py
"""

# The bodies of every HTTPException raised in app.py.
ERROR_SCHEMA = {
    "title": "ErrorDetail",
    "type": "object",
    "properties": {
        "detail": {"title": "Detail", "type": "string"},
    },
    "required": ["detail"],
}

# path -> status code -> description, mirroring the except-clauses of app.py.
ERRORS = {
    "/init": {
        "400": "Bad configuration: unknown config path, or a value the "
        "session model rejects.",
        "409": "A solve is still running — wait for /status to leave "
        "'solving'.",
    },
    "/step": {
        "400": "A value the session model rejects, e.g. winch mode 'reelin'.",
        "409": "Called before /init, or while a solve is running.",
    },
    "/trajectory": {
        "404": "No successful solve yet.",
        "409": "A solve is running — the re-simulation is rejected meanwhile.",
        "500": "The optional forward re-simulation failed; retry with "
        "resimulate=false for the plain guidance table.",
    },
    "/reset": {
        "409": "A solve is running and cannot be interrupted.",
    },
}

ERROR_RESPONSE = {
    "content": {
        "application/json": {
            "schema": {"$ref": "#/components/schemas/ErrorDetail"}
        }
    }
}


def build_spec() -> dict:
    app_spec = create_app().openapi()
    spec = {
        "openapi": app_spec["openapi"],
        "info": app_spec["info"],
        "servers": [
            {
                "url": "http://127.0.0.1:8000",
                "description": "Local server as started by "
                "scripts/server/run_reelout_server.py",
            }
        ],
        "paths": app_spec["paths"],
        "components": app_spec["components"],
    }
    spec["components"]["schemas"]["ErrorDetail"] = ERROR_SCHEMA

    for path, errors in ERRORS.items():
        responses = next(iter(spec["paths"][path].values()))["responses"]
        for code, description in errors.items():
            responses[code] = {"description": description, **ERROR_RESPONSE}

    # /step answers 200 with a StepReply (wait=true) or 202 with a
    # StepAccepted (wait=false); the Union response_model collapses both onto
    # 200, which no client can act on.
    step = spec["paths"]["/step"]["post"]["responses"]
    step["200"] = {
        "description": "Solve finished (wait=true): the optimized "
        "StepParams struct.",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/StepReply"}
            }
        },
    }
    # Two different bodies share the 422 here: the request-validation error of
    # FastAPI and the plain detail string of an infeasible solve.
    step["422"] = {
        "description": "Request validation failed, or the optimization was "
        "infeasible / did not converge. The previous trajectory stays "
        "available from /trajectory.",
        "content": {
            "application/json": {
                "schema": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/HTTPValidationError"},
                        {"$ref": "#/components/schemas/ErrorDetail"},
                    ]
                }
            }
        },
    }
    step["202"] = {
        "description": "Solve accepted (wait=false): poll /status, then "
        "fetch /trajectory.",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/StepAccepted"}
            }
        },
    }
    for path_item in spec["paths"].values():
        for operation in path_item.values():
            codes = operation["responses"]
            operation["responses"] = {c: codes[c] for c in sorted(codes)}
    return spec


class _Dumper(yaml.SafeDumper):
    """Emit the multi-line descriptions as readable block scalars."""

    # The shared error-response dicts must be written out in full: OpenAPI
    # tooling does not have to understand YAML anchors.
    def ignore_aliases(self, data) -> bool:
        return True


def _str_representer(dumper: yaml.Dumper, data: str):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_representer)


def main() -> None:
    body = yaml.dump(build_spec(), Dumper=_Dumper, sort_keys=False, width=88,
                     allow_unicode=True)
    OUTPUT.write_text(HEADER + body, encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
