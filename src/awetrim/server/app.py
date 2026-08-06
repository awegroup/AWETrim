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

"""FastAPI application exposing the reelout trajectory optimization.

Endpoints (see ``schemas.py`` for units and reference-frame conventions):

- ``POST /init`` — build the session (system model, pattern, inflow
  conditions, initial guess). Synchronous, does not solve.
- ``POST /step`` — one warm-started re-optimization (optionally with updated
  inflow conditions and/or tether length). Blocking by default; with
  ``wait=false`` it returns 202 immediately and ``/status`` is polled.
- ``GET /status`` — session state machine + last solve metrics.
- ``GET /trajectory`` — dense guidance table from the last successful solve
  (kept available while a new solve runs and after a failed one).
- ``POST /reset``, ``GET /health``.

Run with a single worker process — the session lives in process memory.
"""

from typing import Callable, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from awetrim.server.schemas import (
    InitReply,
    InitRequest,
    StatusResponse,
    StepAccepted,
    StepReply,
    StepRequest,
    TrajectoryResponse,
)
from awetrim.server.session import (
    NoTrajectoryError,
    ReeloutSession,
    SessionBusyError,
    SessionNotInitializedError,
    SolveFailedError,
)


def create_app(
    session_factory: Callable[[], ReeloutSession] = ReeloutSession,
) -> FastAPI:
    app = FastAPI(
        title="AWETrim reelout optimization server",
        description=__doc__,
        version="0.1.0",
    )
    app.state.session = session_factory()

    def session() -> ReeloutSession:
        return app.state.session

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/init", response_model=InitReply)
    def init(request: InitRequest) -> InitReply:
        """Set up the session. The reply contains the InitParams struct
        (name, max_time, winch_params, inflow_conditions, trajectory); the
        trajectory is the fitted STARTING path — call /step to optimize it."""
        try:
            session().init(request.model_dump())
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return InitReply(**session().init_reply())

    @app.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        return StatusResponse(**session().status())

    @app.post("/step", response_model=Union[StepReply, StepAccepted])
    def step(request: StepRequest):
        """One warm-started re-optimization.

        Default (wait=true): blocks until the solve finishes and the reply
        contains the StepParams struct (winch_params + OPTIMIZED trajectory).
        With wait=false: returns 202 immediately; poll /status and fetch
        /trajectory. An infeasible request returns 422 with the solver
        message; the previous trajectory stays available."""
        kwargs = dict(
            inflow_conditions=request.inflow_conditions.model_dump()
            if request.inflow_conditions
            else None,
            wind=request.wind.model_dump() if request.wind else None,
            winch_params=request.winch_params.model_dump()
            if request.winch_params
            else None,
            trajectory=request.trajectory.model_dump()
            if request.trajectory
            else None,
            distance_radial=request.distance_radial,
            max_iter=request.max_iter,
        )
        try:
            if request.wait:
                return StepReply(**session().step_blocking(**kwargs))
            step_index = session().step(**kwargs)
        except (SessionNotInitializedError, SessionBusyError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except SolveFailedError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"optimization infeasible or not converged: {exc}",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return JSONResponse(
            status_code=202,
            content=StepAccepted(step_index=step_index).model_dump(),
        )

    @app.get("/trajectory", response_model=TrajectoryResponse)
    def trajectory(
        resimulate: bool = Query(
            default=False,
            description="Also re-fly the optimized pattern for a full "
            "timeseries (x/y/z, power, ...). Slower; rejected while solving.",
        ),
    ) -> TrajectoryResponse:
        try:
            payload = session().trajectory(resimulate=resimulate)
        except NoTrajectoryError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:
            # The optional forward re-simulation can fail independently of
            # the NLP; the plain (resimulate=false) response stays available.
            raise HTTPException(
                status_code=500, detail=f"re-simulation failed: {exc}"
            )
        return TrajectoryResponse(**payload)

    @app.post("/reset", response_model=StatusResponse)
    def reset() -> StatusResponse:
        try:
            session().reset()
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return StatusResponse(**session().status())

    return app
