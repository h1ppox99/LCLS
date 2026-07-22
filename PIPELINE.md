# XTC_Agent pipeline — npy-based two-agent workflow

Rewritten workflow. (The old Docker/manifest compton setup — `compton/`,
`agent/.claude/`, `agent/Dockerfile` — has been removed; its `qa` skill was ported
as-is to `skills/qa/` and is not yet wired into the pipeline.)

```
XTC (5 streams, raw)
   │  step 0 · deterministic code, run once
   │  pipeline/step0_xtc_to_npy.py
   ▼
npy/frames_raw.npy  (N,2,512,1024) uint16 memmap      npy/shot_table.npz (per-shot scalars)
   │
   │  step 1 · REDUCTION AGENT  (Claude Agent SDK, phase "reduction")
   │  reads skills/selection/01a,01b + skills/normalization/*,
   │  gathers evidence from shot_table, autonomously decides:
   │    - low-ipm exclusion  run/skip + threshold
   │    - high-ipm exclusion run/skip + percentile
   │    - normalization      run/skip + form + REFERENCE PARAMETER, chosen
   │      from the monitor menu (skills/normalization/01-04: ipm2 |
   │      upstream GMD/BMMON | in-hutch diode boxes | self)
   │  → writes outputs/<run>/decisions.json, then executes step 2
   ▼
   │  step 2 · deterministic code (agent-invoked)
   │  pipeline/step2_accumulate.py: keep-mask + weights → calibrate frames
   │  ((ADC−ped[mode])/gain[mode]) → (weighted) sum
   ▼
outputs/<run>/sum_calib.npy · sum_assembled.npy · sum.png · accumulate_log.json
   │
   │  step 3 · MASK AGENT  (SDK, phase "mask")
   │  reads skills/masking + the summed image, draws layers
   │  → writes mask_assembled.npy + mask_rationale.md, executes
   │  pipeline/step3_apply_mask.py
   ▼
outputs/<run>/masked_sum.npy + masked_sum.png        ← PIPELINE ENDPOINT
   │
   │  step 4 · deterministic code (verifier-invoked)
   │  pipeline/step4_iq.py: azimuthal I(q) + iq_metrics.json
   ▼
   │  step 5 · VERIFIER AGENT  (SDK, phase "verify")
   │  reads skills/verification/*.md (one criterion per file, machine
   │  thresholds in a ```json block), judges iq_metrics.json
   │  → writes verify_report.json  (schema in skills/verification/README.md)
   ▼
 PASS → done.   FAIL → feedback.message is injected into the prompt of
 feedback.target_phase ("reduction" or "mask") and that phase re-runs
 (then everything downstream), up to --max-iters times. Each iteration's
 small artifacts are archived under outputs/<run>/history/iter_NN/.
```

## How to run

```bash
# once: deterministic conversion (~1-2 min, writes ~6.7 GB npy/)
python3 pipeline/step0_xtc_to_npy.py

# once, after step 0 (~3 s): decode ALL remaining per-shot monitor channels
# (gas detectors, e-beam/photon energy, phase cavity, 4 IPM diode boxes,
# BMMON positions) and merge them into npy/shot_table.npz (10 -> 71 columns)
python3 pipeline/step0b_extend_shot_table.py

# full agentic run (needs ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL /
# ANTHROPIC_MODEL in env or ./.env — Stanford gateway). NOTE: if your shell
# exports its own ANTHROPIC_BASE_URL, force the .env one first:
#   set -a; source .env; set +a
python3 agent/entrypoint.py --run-id run475

# with the verify-feedback loop (verifier FAIL → feedback → re-run, ≤3 rounds)
python3 agent/entrypoint.py --run-id run475 --max-iters 3

# verify an existing endpoint only
python3 agent/entrypoint.py --run-id run475 --phase verify

# mechanics-only baseline (no API calls, rule-based decisions + mask + verify)
python3 agent/entrypoint.py --run-id smoke --no-llm

# stability test: K repeated runs + cross-trial comparison report
python3 test_stability.py --trials 3 --no-llm     # mechanical reproducibility
python3 test_stability.py --trials 3              # agent-decision stability
```

## File contracts

- `decisions.json` — schema documented at the top of `pipeline/step2_accumulate.py`.
  Every decision carries a `rationale`. Skipping is allowed; silent skipping is not.
  `normalization.monitor` may name any shot_table column (e.g. `ipmfex22_sum`);
  step 2 derives that monitor's own zero offset (`monitor_offset`, default auto).
- `mask_assembled.npy` — (1064, 1030) bool, True = masked, assembled geometry
  (calib/ix.npy, calib/iy.npy map panel pixels → assembled image).
- calib constants in `calib/` (ped/gain/status per gain mode, from dark run 362 +
  detector-group gain files; ix/iy from the smalldata UserDataCfg snapshot).
- `verify_report.json` — verifier verdict, schema in `skills/verification/README.md`.
  `iq.npy` / `iq_metrics.json` — deterministic I(q) + metrics from `step4_iq.py`.
  Exit code 5 = pipeline ran but final verdict is FAIL.
- Agent skill sources: `skills/selection/01a_low_ipm_exclusion.md`,
  `skills/selection/01b_high_ipm_exclusion.md`,
  `skills/normalization/01_per_shot_flux_ipm2.md` +
  `02_per_shot_flux_upstream.md` / `03_per_shot_flux_alternatives.md` /
  `04_per_shot_flux_self.md` (one file per normalization reference — ipm2 /
  upstream GMD+BMMON / in-hutch diode boxes / detector self; 02-04 need the
  step0b extended shot_table), `skills/masking/*`,
  `skills/verification/*` (verifier criteria — one md per criterion).

## Stability criteria (test_stability.py)

STABLE = across trials: identical non-rationale decision fields, identical kept-shot
counts, pairwise mask IoU > 0.99. The report also tracks endpoint image differences.
`--no-llm` isolates pipeline mechanics (must be bit-stable); LLM mode measures how
reproducible the *agent's judgments* are given the same evidence and skills.
