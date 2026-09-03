# automask: the masking library

`automask` is a plain Python library. There is no command-line tool: humans and
Slurm jobs use it from a script by holding the objects directly, and the agent
reaches the same operations through a per-run stdio MCP server (see
`.claude/skills/automask/SKILL.md` and `lcls_agent/`). Activate the Python 3.11
psana environment first:

```bash
source psana_env.sh
python -m pip install -e . --no-deps
```

## Capability discovery

`automask.interface.catalog.capability_catalog()` returns the registered statistics,
regularizers, selection operators, reductions, default parameters, and the
field-native parameter shapes that `ShotSelection` and `Pipeline` accept. It is
pure introspection — prefer it over any cached copy, which may reflect older code.

```python
from automask.interface.catalog import capability_catalog

cat = capability_catalog()
```

## Scripting the workflow

Inspect a run once, then reuse its `RunProfile` for selection, preview, masking,
and validation. `ProfileStore` caches profiles on disk by run — the same
compute-or-load pattern `ImageStore` uses for reductions — so a later script
skips the expensive psana pass.

```python
import numpy as np
from automask.profiling.run_inspection import inspect_run
from automask.profiling.profile_store import ProfileStore
from automask.selection.shot_selection import ShotSelection, Condition, PercentileTrim
from automask.sample import Sample
from automask.sample.image_store import ImageStore
from automask.mask import production_pipeline
from automask.profiling.utils import configure_psana_environment

configure_psana_environment()

# 1. Inspect once, cache the profile by run (the one expensive psana read).
profiles = ProfileStore()  # pass cache_dir= to relocate
profile = profiles.try_load(475)
if profile is None:
    report = inspect_run("xppl1016922", 475, ...)  # see inspect_run signature
    profile = report.profile
    profiles.save(profile)

# 2. Select shots (field-native objects; conditions are ANDed).
selection = ShotSelection(
    where=(Condition("ai/ch03", "<=", 2.0),),
    trim=PercentileTrim("diodeU/channels[0]", low=0.03, high=0.03),
    n_shots=800,
)
print(selection.describe(profile))  # stage-by-stage counts

# 3. Build a mask. Keep the geometry + status_as_mask floor channels.
pipeline = production_pipeline()
store = ImageStore(run_profile=profile)  # pass cache_dir= to reuse reductions
sample = Sample.from_store(profile.run, selection, pipeline.needs(), store=store)
mask = pipeline.run(sample)  # boolean; True == masked
np.save("mask.npy", mask)
```

Reductions (`mean`, `std`, `mad`) are computed through `ImageStore`;
give it a `cache_dir` to reuse the content-addressed cache across runs. For
sensitivity analysis, `automask.evaluation.validate_mask` runs declared
parameter/fold perturbations and returns a descriptive stability report plus a
recommended pipeline — stability is robustness to the tested variations, not
agreement with ground truth.

## Conventions

- Masks are boolean and `True` means excluded/masked.
- Reduced images are in assembled detector space; calibration constants stay in
  native panel space until the consuming statistic interprets them.
- Run-level masking requires the registered `geometry` and `status_as_mask` floor
  channels; `automask.interface.recipes.require_run_floor` checks a pipeline for them.
- A profile built with `max_events` is development evidence, not a full run.

## Parameters as dicts

`automask.interface.recipes` converts between these objects and field-native dicts
(`selection_to_dict`/`selection_from_dict`, `pipeline_to_dict`/`pipeline_from_dict`,
`validation_design_to_dict`/`validation_design_from_dict`). The agent tools accept
those dicts inline; a script can pass the objects directly and never touch them.
