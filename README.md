# XTC Agent

XTC Agent is a research pipeline for reducing LCLS Jungfrau1M scattering data
directly from XTC files. Deterministic Python stages handle decoding, calibration,
accumulation, center estimation, masking, and azimuthal integration. Agent phases
use the versioned method files under `skills/` to make and explain the decisions
between those stages.

The repository is grounded in `xppl1016922` Run0475 (LaB6). Raw XTC files and
generated run directories are intentionally not committed.

## Repository layout

| Path | Purpose |
|---|---|
| `agent/` | Orchestrator, workflow memory, and skill-usage reporting |
| `pipeline/` | Deterministic data-reduction and validation stages |
| `skills/` | Versioned selection, normalization, center, masking, verification, and QA methods |
| `memory/` | Measured campaign findings injected into agent prompts |
| `tests/` | Fast workflow and center-semantics tests |
| `calib/` | Run-specific calibration and detector-geometry arrays |
| `npy/` | Shot-table metadata; the large decoded frame array remains local |
| `papers/` | Literature index; source PDFs remain local |
| `outputs/` | Local runtime products and historical tracked validation examples |

See [PIPELINE.md](PIPELINE.md) for the full dataflow, file contracts, phase
behavior, and run commands. See [skills/README.md](skills/README.md) for the skill
schema and method catalogue.

## Setup

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For an agent-backed run, configure `ANTHROPIC_AUTH_TOKEN`,
`ANTHROPIC_BASE_URL`, and `ANTHROPIC_MODEL` in the environment or in a local
`.env` file. The `.env` file is ignored by Git.

## Run

```bash
# Convert local XTC streams once, then extend the shot table.
python pipeline/step0_xtc_to_npy.py
python pipeline/step0b_extend_shot_table.py

# Full agentic workflow.
python agent/entrypoint.py --run-id run475 --max-iters 3

# Deterministic mechanics-only workflow with no API calls.
python agent/entrypoint.py --run-id smoke --no-llm --image-center 992 35
```

The installed editable environment also exposes `xtc-agent` as an alias for
`python agent/entrypoint.py`.

## Verify

```bash
python -m pytest -q
ruff check agent pipeline skills tests test_stability.py
PYTHONPYCACHEPREFIX=/tmp/xtc-agent-pyc python -m compileall -q \
  agent pipeline skills tests test_stability.py
```

The tests do not read the multi-gigabyte raw frame array. Full end-to-end runs do.
