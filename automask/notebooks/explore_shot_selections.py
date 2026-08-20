# %% [markdown]
# # Explore shot selections
#
# Visually compare selected-shot images produced by different `ShotSelection`
# recipes using `automask.viz`. Each image is a reduction over the
# shots a `ShotSelection` keeps. A recipe consists of arbitrary `Condition`s
# over canonical profile fields, an optional `PercentileTrim`, a shot cap,
# and an optional normalization field.
#
# **Physics.** There is no laser in this experiment: one x-ray beam is split into
# the **CC** and **VCC** branches, each with its own shutter (`ai/ch02`/`ai/ch03`
# thresholded at 2 V). EVR code 137 records whether the machine delivered
# x-rays. These are experiment meanings supplied below, not selector concepts.
#
# **Branch availability.** CC is open on 100% of shots in both local runs, so
# a CC-open comparison is redundant. VCC is open on 90% of run 389 but **0%
# of run 475**; a VCC-open condition on run 475 returns no shots.
#
# **Kernel.** Select the **`Python (ana-psana)`** kernel (top-right). It's the
# `ana-4.0.62` conda env — the only one with both `automask` *and* psana, and it
# has the `SIT_*` data vars baked in, so both cached and uncached selections work.
# Any kernel lacking `automask` fails at the import cell; one with `automask` but
# no psana can only render selections already in the cache.
#
# **Cache note.** A selection already in the ImageStore cache renders instantly
# (numpy only). A *new* selection triggers a raw-XTC pass (needs the psana kernel
# above; first render is slow, then it's cached). Changing a condition changes
# the provenance key, so expect an XTC pass for each new recipe.
#
# Present runs: **389** and **475**.

# %%
import os
from pathlib import Path

import automask

OUTPUT_DIR = (
    Path(automask.__file__).resolve().parent
    / "outputs"
    / "notebooks"
    / "explore_shot_selections"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved {path}")
    return path


os.environ.setdefault("SIT_PSDM_DATA", "/home/groups/darve/hippowal/psdm")
os.environ.setdefault("SIT_ROOT", "/home/groups/darve/hippowal/psdm/sit_root")
os.environ.setdefault("SIT_DATA", "/home/groups/darve/hippowal/psdm/data")

# %%
import matplotlib.pyplot as plt

from automask import viz
from automask.shot_selection import Condition, PercentileTrim, ShotSelection

RUN = 389  # or 475
BEAM = "DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[137]"
VCC = "ai/ch03"
I0 = "diodeU/channels[0]"
BEAM_ON = (Condition(BEAM, "==", 1),)
BEAM_OFF = (Condition(BEAM, "==", 0),)

# The production selection is prewarmed by `python -m automask.producers.build_images`.

# %% [markdown]
# ## 1. A single selected-shot image
#
# `show_image` accepts a run, a `ShotSelection`, and a reduction (default
# `'mean'`). Dead/zero pixels are shown neutral; scale is a robust 1–99th pct.

# %%
reductions = ["mean", "std", "mad"]

selection = ShotSelection(
    where=BEAM_ON,
    trim=PercentileTrim(I0, low=0.03, high=0.03),
    n_shots=100,
)

# %%
viz.show_image(RUN, selection, reduction="mean")
save_figure(plt.gcf(), f"run_{RUN:04d}_selected_mean.png")
plt.show()

# %%
viz.show_image(RUN, ShotSelection(where=BEAM_OFF), reduction="mean")
save_figure(plt.gcf(), f"run_{RUN:04d}_beam_off_mean.png")
plt.show()

# %%
viz.show_image(RUN, selection, reduction="std")
save_figure(plt.gcf(), f"run_{RUN:04d}_selected_std.png")
plt.show()

# %% [markdown]
# ## 2. Effect of a shot selector: beam-on vs i0-normalized vs dark
#
# `compare_selections` renders one panel per selection on a **shared** color scale
# and colorbar, so brightness differences between recipes are real (not per-panel
# autoscaled). Here: plain beam-on mean vs the same with per-shot `sample_diode`
# normalization vs the beam-off dark.
#
# Note `sample_diode` (the lab's own normalizer) is *downstream* of the CC/VCC
# split, so unlike ipm2 it tracks the flux actually reaching the detector.

# %%
viz.compare_selections(
    RUN,
    [
        ShotSelection(where=BEAM_ON),
        ShotSelection(where=BEAM_ON, normalization=I0),
        ShotSelection(where=BEAM_OFF),
    ],
    reduction="mean",
)
save_figure(plt.gcf(), f"run_{RUN:04d}_beam_normalization_comparison.png")
plt.show()

# %% [markdown]
# ## 3. Same idea on the std reduction
#
# The beam-on per-pixel std is where scattering contrast lives. Compare
# plain vs `sample_diode`-normalized.

# %%
viz.compare_selections(
    RUN,
    [
        ShotSelection(where=BEAM_ON),
        ShotSelection(where=BEAM_ON, normalization=I0),
    ],
    reduction="std",
)
save_figure(plt.gcf(), f"run_{RUN:04d}_beam_std_comparison.png")
plt.show()

# %% [markdown]
# ## Investigate from here
#
# Swap in your own field conditions below. Remember an **uncached** recipe
# needs the psana env (it does a raw-XTC
# pass on first render, then caches). Ideas to try:
#
# - intensity trim: `PercentileTrim(I0, low=..., high=...)` — how much
#   do the brightest/dimmest shots move the mean?
# - branch state: `Condition(VCC, '>', 2)` vs `'<='` on **run 389** (run 475 has no
#   VCC-open shots). This is the real physical knob — the two branches deposit
#   visibly different flux on the detector.
# - trim or normalization field — compare a downstream monitor (`diodeU`,
#   `lombpm`) against the upstream `ipm2`, which is blind to the branch state.
# - `reduction='std'` vs `'mean'`.
#
# `show`, `show_image`, and `compare_selections` all take an optional `ax=` and
# return their artist/figure — pass `out='foo.png'` to save.

# %%
viz.compare_selections(
    389,  # run 475 has zero VCC-open shots
    [
        ShotSelection(where=BEAM_ON + (Condition(VCC, ">", 2.0),), n_shots=800),
        ShotSelection(where=BEAM_ON + (Condition(VCC, "<=", 2.0),), n_shots=800),
    ],
    reduction="mean",
)
save_figure(plt.gcf(), f"run_{RUN:04d}_vcc_branch_comparison.png")
plt.show()
