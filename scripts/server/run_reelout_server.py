"""Launch the AWETrim reelout-optimization REST server.

Run from the project root:

    python scripts/server/run_reelout_server.py --port 8000

Requires the server extra:  pip install -e .[server]

The server holds a single in-process optimization session, so it must run
with exactly one worker process (enforced below). Interactive docs are at
http://<host>:<port>/docs once running.

Typical client flow (e.g. from a Julia kite simulator):
    POST /init        {"wind": {"model_type": "logarithmic", "U_ref": 8}}
    POST /step        {}                          -> 202, solve in background
    GET  /status      poll until "converged"
    GET  /trajectory  dense guidance table (t, azimuth, elevation, rates, r, r_dot)
    POST /step        {"wind": {...}, "distance_radial": 220}   -> refresh
"""

# Force a non-interactive matplotlib backend BEFORE any awetrim import:
# awetrim modules import pyplot at module level, and the solver runs in a
# background thread where a GUI backend would crash or hang on Windows.
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AWETrim reelout REST server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    from awetrim.server.app import create_app

    # workers=1 is required: the optimization session lives in process memory.
    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
