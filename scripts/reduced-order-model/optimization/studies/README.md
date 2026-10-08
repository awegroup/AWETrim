# Optimisation studies

One-off studies built on the full-cycle optimiser in `../cycle/`. They are
additional options, not replacements for the production cycle scripts. Run
everything from the repo root; outputs go to `results/<kite>/optimization/`.

## Up-loop figure eight (`run_uploop_eight_opti.py`, `plot_uploop_eight_comparison.py`)

Question: can ONE figure eight per period carry both the reel-out and the
reel-in, with fewer steering reversals than the normal cycle (figure eights
plus a separate reel-in over the top), at medium wind?

Formulation: periodic B-spline path (`C_phi`, `C_beta`, `r0` free),
`winch_mode: free_speed` (v_r is a free control, its sign decides where the
kite reels in), per-node depower, free period, `periodic_wrap` (the seam
interval is flown and the cycle closes there), max mean power. Staged S0
fixed path / S1 shape (repeated while a step box is active) / S2 polish.
All shapes share the same bounds: height band 40-400 m (z = r sin beta),
turn radius >= 25 m, 2 deg stall margin, ROM depower band, r0 <= 1000 m
(the 415 m tether still caps r). Settings: `uploop_eight:` and
`symmetric_eight:` in `data/LEI-V3-KITE/cycle_profile.yaml`.

```
# slanted up-loop eight + the normal cycle on the same formulation
python scripts/reduced-order-model/optimization/studies/run_uploop_eight_opti.py --kite LEI-V3-KITE --yes --compare --baseline-seed
# symmetric eight climbing at both sides, mirror symmetry imposed
python scripts/reduced-order-model/optimization/studies/run_uploop_eight_opti.py --kite LEI-V3-KITE --shape symmetric --symmetric --yes --compare
# trajectories (2-D and 3-D) and time series of the three
python scripts/reduced-order-model/optimization/studies/plot_uploop_eight_comparison.py
```

Result on the LEI-V3 at 10 m/s @ 100 m (log, z0 0.03), 2026-10-08:

| cycle | mean power | steering reversals | r0 |
|---|---|---|---|
| normal (figure eights + reel-in) | 6.3 kW | 8 | 242 m |
| slanted up-loop eight | 6.0 kW | 4 | 339 m |
| symmetric eight (mirror) | 4.7 kW | 6 | 359 m |

- The slanted eight comes within ~3 % of the normal cycle with half the
  steering reversals, but its reel-in arc climbs to ~78 deg: it converges to
  the normal over-the-top reel-in, with fewer turns.
- The side-climb basin (reel in over both side tops) exists only under the
  imposed mirror symmetry; free, it drifts to a one-sided reel-in. Reeling in
  twice per cycle costs ~1.5 kW.
- Both eights want a longer tether than the normal cycle (r0 interior at
  339/359 m vs 242 m).
- Not fully converged: the shape step boxes, the turn-radius floor and the
  s_dot floor still bind, and the eights' low lobes fold into extra loops in
  the last passes (7 and 5 self-crossings); the last single-crossing slanted
  eight gave 5.6 kW with 2 reversals.
