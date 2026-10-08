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
fixed path / S1 shape (re-centred while any step box is active, at most 6
passes, each saved as `_stage1_passK`) / S2 polish. Power, duration and
reel-out fraction use the NLP's own left rule over all N intervals,
`dt_i = (s_{i+1} - s_i) / s_dot_i`, seam included. With `--compare` the
table adds a "last clean pass" column: the last pass whose self-crossings
and lobe senses still match the seed (a line is printed when the final
topology differs).
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

| cycle | mean power | steering reversals | cycle time | r0 |
|---|---|---|---|---|
| normal (figure eights + reel-in) | 6.25 kW | 8 | 69.9 s | 242 m |
| slanted up-loop eight | 6.07 kW | 4 | 74.3 s | 339 m |
| symmetric eight (mirror) | 4.77 kW | 6 | 78.6 s | 359 m |

(Recomputed 2026-10-08 from the stored optima with the left rule above; the
first write-up used a right-rule time axis and read 6.3 / 6.0 / 4.7 kW.)

- The slanted eight comes within ~3 % of the normal cycle with half the
  steering reversals, but its reel-in arc climbs to ~78 deg: it converges to
  the normal over-the-top reel-in, with fewer turns.
- The side-climb basin (reel in over both side tops) exists only under the
  imposed mirror symmetry; free, it drifts to a one-sided reel-in. The
  mirror eight makes 4.8 kW: reeling in twice per cycle costs ~1.3 kW
  against the slanted eight (~1.5 kW against the normal cycle).
- Both eights want a longer tether than the normal cycle (r0 interior at
  339/359 m vs 242 m).
- Not fully converged: the shape step boxes, the turn-radius floor and the
  s_dot floor still bind, and the eights' low lobes fold into extra loops in
  the last passes (7 and 5 self-crossings); the last single-crossing slanted
  eight (S1 pass 3) gave ~5.6 kW with 2 reversals -- a log value of that run,
  under the earlier right-rule time axis (the correction is ~1 %), since the
  per-pass results were not saved then; a re-run now stores them.
- The mirror rows make the problem LICQ-degenerate (every active node bound
  is active at both mirrored nodes); IPOPT copes with it, but the clean
  formulation is elimination of the second half (open follow-up).
