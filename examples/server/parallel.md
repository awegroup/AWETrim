# Multiple parallel sessions — feasibility notes

The reelout server currently serves **one** session: `create_app` builds a single
`ReeloutSession` and stores it in `app.state.session`, and every endpoint talks to
that one object ("Run with a single worker process — the session lives in process
memory", `app.py` module docstring). This note records what it would take to serve
several sessions at once, and what was measured about running solves concurrently.

Two separate questions, with very different answers:

1. **Several sessions coexisting** (each client keeps its own model, pattern and
   last trajectory) — easy, the code is already shaped for it.
2. **Several solves running at the same time** — this is the hard part, and
   thread-based concurrency measurably does not work.

## 1. Session state — the easy half (~half a day)

`session.py` is already written as if it were one of many:

- all mutable state is instance state on `ReeloutSession` (set in `_clear`), no
  module-level globals;
- it carries its own `threading.Lock`;
- no file writes from a solve — `phase._plot_failed_iterate` is stubbed out in
  `init()` precisely so a worker never writes plots;
- it has no FastAPI/pydantic dependency: plain dicts in, plain dicts out;
- `session_id` is already a field of `InitReply` and `StatusResponse`
  (`schemas.py`), currently pinned to the class constant `ReeloutSession.session_id
  = "default"`.

So the work is confined to the app layer and the clients:

- `app.py`: replace `app.state.session` with a `{session_id: ReeloutSession}`
  registry plus a lock guarding it. `POST /init` mints an id (uuid4, or the
  request's `name`) and returns it — the reply field already exists.
- Routing: read the id from an `X-Session-Id` header (or a query parameter) and
  fall back to `"default"` when absent. That keeps the four example clients, the
  README flow and `openapi.yaml`'s paths working unchanged. A path prefix
  (`/sessions/{id}/step`) is cleaner REST but breaks all of them.
- `session_id` becomes an instance attribute instead of a class constant.
- Lifecycle, not optional: `DELETE /session`, a cap on live sessions (reject with
  409/429 beyond it), and idle-TTL eviction. Without it every `/init` leaks a full
  `SystemModel` + `Phase` + expanded NLP.
- Clients (`client_example.{py,jl,m}`, `client.py`) store the id from `/init` and
  send it on `/step`, `/status`, `/trajectory`. Regenerate `openapi.yaml`, update
  this folder's README, extend `tests/server/test_app.py` (it already parameterizes
  construction via `create_app(session_factory=...)`).

## 2. Concurrent solves — the hard half

Endpoints are sync `def`, so FastAPI runs them in the anyio threadpool: parallel
blocking `/step` calls **would** genuinely overlap. Today nothing overlaps only
because there is a single session (a second `/step` gets 409). That is the risk to
understand before allowing N sessions.

### Measurements

Synthetic NLP (1200 variables, `sin`/`diff` objective, 2 equality constraints)
solved through the same stack and options `phase.py` uses — CasADi `nlpsol`, IPOPT,
`linear_solver: "mumps"`, `hessian_approximation: "limited-memory"`. Linux, CPython
3.12, venv of this repo:

```
1 solve, nothing else running                 :  1.41 s
1 solve + 1 busy python thread (default 5 ms) : 15.50 s   (11.0x slower)
1 solve + 1 busy python thread (0.5 ms)       :  3.09 s   ( 2.2x slower)

2 solves sequential                           :  4.22 s
2 solves in threads, one process              :  4.53 s   (0.93x — no gain)
2 solves in separate processes                :  2.94 s   (1.43x)
```

A separate probe: a pure-Python counting loop in another thread runs at **93 %** of
its solo rate while a solve is in progress.

### What that means

CasADi *does* release the GIL during the solve (hence the 93 %), but it re-acquires
it often enough that a competing Python thread starves the solve — the classic
convoy effect, where the native thread pays up to one `sys.setswitchinterval`
(default 5 ms) on each re-acquisition. Consequences:

- Thread-based parallelism buys nothing: two concurrent solves take as long as
  running them one after the other (0.93x).
- Worse, it makes each individual solve slower and less predictable. A second
  client merely *polling* `/status` is Python work competing with a running solve.
  A 10–20 s solve degrading by even 2x is a co-simulation problem, since the
  simulator is waiting on it.
- `sys.setswitchinterval(0.0005)` cuts the damage from 11x to 2.2x, so it is a
  mitigation, not a fix (and it slows all other Python work in the process).
- Separate processes scale as expected.

### Secondary risk: MUMPS reentrancy

IPOPT with MUMPS is not generally safe to drive from several threads of one
process. Two and three concurrent solves did not crash in these runs, but a handful
of runs is not evidence of safety under sustained load. Process isolation removes
the question entirely.

## 3. Options

**A. Registry + one global solve semaphore (~half a day).** Do section 1, then wrap
the solve in a process-wide `threading.Semaphore(1)`. Multiple sessions each keep
their own model, pattern and last trajectory; solves queue instead of overlapping,
so solve time stays 10–20 s rather than degrading. Blocking `/step` simply waits
longer; the existing "previous trajectory stays available" contract is unaffected.
If the requirement is *"several simulators each keep their own path"* rather than
*"solves literally run at the same time"*, this is the right answer.

**B. Process per session (~1–2 days).** `/init` spawns a long-lived worker; the
parent registry holds a pipe per session and forwards `(method, kwargs)` → dict
replies. Use the `spawn` start method — `fork` with CasADi and threads already in
the parent is asking for trouble — which costs ~1 s of re-import per `/init`.
The CasADi `Opti`/`MX` objects cannot be pickled, so the session must *live* in the
worker; that is exactly what this design does, and because `session.py` is already
dict-in/dict-out it needs no changes inside it. Gives real parallelism plus
isolation from MUMPS reentrancy. Costs N× the NLP memory, and needs worker
supervision (crash → session marked failed, not a hung parent).

**C. One server process per client, different `--port`.** Zero code. Already
supported by `bin/run_server --port`.

### B versus C

Both isolate the solves in separate processes, so they give the **same**
parallelism and the same protection against MUMPS reentrancy. Everything B adds is
operational:

| | B (process per session) | C (server per client) |
|---|---|---|
| client configuration | one fixed address, session from `/init` | a port per client, mapping kept by hand |
| remote clients | one firewall port | one firewall port per client |
| session lifecycle | `/init` creates, `DELETE`/idle-TTL destroys | a human starts and stops processes |
| abandoned client | evicted, memory reclaimed | server keeps the NLP resident until noticed |
| crashed session | reported as `failed` on the surviving parent | dead port that still looks alive in the mapping |
| load limits | cap on live sessions and concurrent solves in one place | nothing enforces one; N users oversubscribe the cores |
| visibility | one log, one `/health`, `GET /sessions` | N terminals |
| `/init` latency | ~1 s worker spawn (a warm pool hides it) | none |
| new failure modes | pipe protocol, worker supervision, zombie reaping | none |
| debugging | multiplexed logs, indirection through the pipe | each server owns its terminal |
| restart / upgrade | restarts every session | per client, independently |

So B is worth building when sessions come and go programmatically, or when this
becomes a shared service used by people who should not have to start their own
process. At the scale of a handful of known simulators started by hand, B's entire
benefit is administration that nobody is doing anyway — C wins on value.

Recommendation: **A** unless concurrent solves are genuinely required. If they are,
**C** while the set of clients is small and human-managed, **B** once session
creation is programmatic or the server is shared.

## Reproducing the benchmark

```python
"""Threads vs processes for concurrent ipopt+mumps solves, plus the effect of
sys.setswitchinterval on the 'native thread starved by a python thread' convoy."""
import multiprocessing as mp
import sys
import threading
import time

import casadi as ca
import numpy as np

N = 1200
stop = threading.Event()


def spinner():
    while not stop.is_set():
        pass


def solve(k=1):
    x = ca.MX.sym("x", N)
    f = ca.sumsqr(ca.sin(x) - 0.5) + ca.sumsqr(ca.diff(x)) ** 2
    g = ca.vertcat(ca.sum1(x) - 1.0, ca.sumsqr(x) - 10.0)
    s = ca.nlpsol("s", "ipopt", {"x": x, "f": f, "g": g},
                  {"ipopt.print_level": 0, "print_time": 0,
                   "ipopt.linear_solver": "mumps",
                   "ipopt.hessian_approximation": "limited-memory",
                   "ipopt.max_iter": 3000})
    s(x0=np.linspace(0, 1, N) + 0.01 * k, lbg=[0, 0], ubg=[0, 0])


def timed(fn, *a):
    t0 = time.time(); fn(*a); return time.time() - t0


if __name__ == "__main__":
    solve(0)  # warm up
    alone = timed(solve, 1)
    print(f"1 solve, nothing else running          : {alone:.2f}s")

    for interval in (0.005, 0.0005):
        sys.setswitchinterval(interval)
        stop.clear(); th = threading.Thread(target=spinner); th.start()
        dt = timed(solve, 1)
        stop.set(); th.join()
        print(f"1 solve + 1 python thread (switch={interval*1000:g} ms): "
              f"{dt:.2f}s  ({dt/alone:.1f}x slower)")
    sys.setswitchinterval(0.005)

    seq = timed(lambda: [solve(k) for k in (1, 2)])
    t0 = time.time()
    ts = [threading.Thread(target=solve, args=(k,)) for k in (1, 2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    thr = time.time() - t0
    t0 = time.time()
    ps = [mp.Process(target=solve, args=(k,)) for k in (1, 2)]
    [p.start() for p in ps]; [p.join() for p in ps]
    proc = time.time() - t0
    print(f"\n2 solves sequential                    : {seq:.2f}s")
    print(f"2 solves in threads (one process)      : {thr:.2f}s  ({seq/thr:.2f}x)")
    print(f"2 solves in separate processes         : {proc:.2f}s  ({seq/proc:.2f}x)")
```

Beware when writing variants of this: comparing *different* problems sequentially
vs. threaded (different `x0` → different iteration counts) produced an apparent
4x "speedup" that was pure noise. Both measurements must run the identical set of
problems.
