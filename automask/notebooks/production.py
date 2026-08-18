# %% [markdown]
# # Production notebook building masking like an agent would do it

# %% [markdown]
# The goal of this notebook is to go through what an agent should do when producing a mask for a given run, and use this exploration to identify simplifications and clarifications necessary in the codebase.

# %% [markdown]
# **Plan** (input = RUN):
# 1. Explore content of the run + associated calibration data
# ```python
# list_contents(RUN)
# # Should list number of shots, different properties, number of shots available per property, calibration data available
# # Geometry of the detector etc ...
# ```
# 2. Shot selection : based on content, decide on several shot selections and run them
# ```python
# select_shots(...) # Selection 1
# select_shots(...) # Selection 2
# select_shots(...) # Selection 3
# display_shots() # Displays shots together for visual analysis and refine/discard some if necessary
# ```
# 3. Masking : based on observed shots, decide what masking tools to use and apply them (always apply geometry + calib first)
# ```python
# detector(...) # Masking : includes stats, regularization etc ...
# display_masks() # Display each mask obtained and refine parameters or tools if necessary
# ```
# 4. Validation : based on obtained masks, apply validation methods to refine previous behavior if necessary
# ```python
# raise(NotImplemented)
# ```

# %%
from pathlib import Path

import automask
import matplotlib.pyplot as plt

from automask.utils import configure_psana_environment

OUTPUT_DIR = (
    Path(automask.__file__).resolve().parent / "outputs" / "notebooks" / "production"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved {path}")
    return path


PSANA_ENVIRONMENT = configure_psana_environment()

# %% [markdown]
# ## 1. Run inspection
#
# The inspection resolves the run's XTC streams and applicable detector calibration files, then makes one XTC pass to inventory raw payloads and build the event-aligned `RunProfile`. It also reads detector geometry. The resulting report is the concise context handed to the agent; the full arrays remain available through `inspection_report.profile`.

# %% [markdown]
# Inputs to clarify before starting:
# 1. Experiment name
# 2. Run number
# 3. Detector alias, psana source, and calibration type

# %%
from automask.io.read_xtc import JUNGFRAU_NAME
from automask.run_inspection import inspect_run

EXPERIMENT_NAME = "xppl1016922"
RUN = 396
DETECTOR_NAME = JUNGFRAU_NAME
DETECTOR_SOURCE = "XppEndstation.0:Jungfrau.0"
DETECTOR_CALIB_TYPE = "Jungfrau::CalibV1"

# %%
inspection_report = inspect_run(
    EXPERIMENT_NAME,
    RUN,
    DETECTOR_NAME,
    DETECTOR_SOURCE,
    DETECTOR_CALIB_TYPE,
)
inspection_report.display()
inspection_report.save(
    OUTPUT_DIR / f"run_{RUN:04d}_inspection",
    overwrite=True,
)
run_profile = inspection_report.profile

# %% [markdown]
# > [!warning]
# >
# > Some key information is sometimes not available from the configuration files. Examples include sample-detector distance, ?? (#TODO include others). The agent should be able to identify this and either
# > 1. Retrieve if from log files or other sources
# > 2. Ask the user for this information
# > 3. Proceed to calibrate the distance itself from the data (if possible)

# %% [markdown]
# ## 2. Shot selection

# %% [markdown]
# The profiler is an inventory, not an automatic selector. The practitioner supplies the small amount of experimental meaning that the files cannot provide: which fields and values define the scientifically relevant population. The agent writes those choices as ordinary comparisons over the profiled field names, checks the resulting counts, and only then decodes detector frames.
#
# For this run the context is simple: EVR 137 is the genuine beam flag; `ai/ch02` (CC) is constant open and therefore adds no information; `ai/ch03` (VCC) is constant closed and selects the requested branch state; and `sample_diode` is downstream and suitable for trimming lit shots. Beam-off shots do not need an intensity field because no percentile trim or normalization is requested.
#
# `ShotSelection` has no built-in beam or branch concepts. Another experiment can use completely different fields and values without changing the selection library. Conditions support `==`, `!=`, `<`, `<=`, `>`, `>=`, `between`, `in`, `not in`, `finite`, and `nonzero`; all conditions are combined with AND.

# %% [markdown]
# ### Practitioner-defined selection recipe

# %%
from automask.shot_selection import Condition, PercentileTrim, ShotSelection

BEAM_ON_FIELD = "DetInfo(NoDetector.0:Evr.0)/EvrData.DataV4/eventCode[137]"
VCC_FIELD = "ai/ch03"
INTENSITY_FIELD = "diodeU/channels[0]"

LIT_SELECTION = ShotSelection(
    where=(
        Condition(BEAM_ON_FIELD, "==", 1),
        Condition(VCC_FIELD, "<=", 2.0),
    ),
    trim=PercentileTrim(INTENSITY_FIELD, low=0.03, high=0.03),
    n_shots=800,
)

DARK_SELECTION = ShotSelection(
    where=(Condition(BEAM_ON_FIELD, "==", 0),),
)

# %%
from automask.image_store import ImageStore
from automask.viz import show

requests = {
    "Lit calibrated mean": (LIT_SELECTION, "mean"),
}
store = ImageStore(run_profile=run_profile)
selected_images = {
    name: store.reduce(RUN, selection, reduction)
    for name, (selection, reduction) in requests.items()
}

for name, image in selected_images.items():
    artist = show(image, title=f"Run {RUN:04d}: {name.lower()}")
    filename = name.lower().replace(" ", "_")
    save_figure(artist.axes.figure, f"run_{RUN:04d}_{filename}.png")
    plt.show()

# %% [markdown]
# ## 3. Masking

# %% [markdown]
# When entering the masking stage, the agent has already profiled the run and selected a set of shots. The next step is to apply masking tools to the selected shots, starting with geometry and calibration masks.

# %% [markdown]
# ### Practitioner-defined masking recipe
#
# A `Pipeline` is one list of `Channel`s and a combiner. A channel is one statistic plus the stages around it, and whether it belongs to the intensity-free floor is read from the statistic's registered kind — not declared a second time. So `geometry` and `status_as_mask` are configured exactly like `blackhat`, and their knobs are reachable the same way.
#
# `status_as_mask` is psana's per-run pixel status, read through `psana.Detector` at the run being masked. Nothing here loads a frozen array.
#
# `asic_polish` targets a different defect class: dark-current and offset defects that live in the *pedestal calibration constant* rather than in the scattering image, and that `status_as_mask` does not flag. It works in native panel space, removing each 256×256 ASIC's row/column structure by median polish and scaling the residual by its own MAD, then projects back to assembled space. Its per-pixel z sits below the chip's own noise floor, so it is useless alone — the `blob_scale` field regularizer aggregates the spatially coherent evidence first, which is what makes `k=15` separable. `fill_holes` and `area_gate` then clean the resulting mask.
#
# `Sample.from_store` then materializes exactly the arrays `MASKING_PIPELINE.needs()` asks for: names that are reductions are computed over `LIT_SELECTION`'s shots (already cached from the cell above), and every other name is passed to psana as a calibration accessor. Adding `asic_polish` is what puts `pedestals` in that list — a shot selection never sees it, because a pedestal is not a shot.

# %%
from automask.masking import Channel, Pipeline
from automask.regularization.area_gate import AreaGateParams
from automask.regularization.blob_scale import BlobScaleParams
from automask.regularization.fill_holes import FillHolesParams
from automask.sample import Sample
from automask.stats.asic_polish import AsicPolishParams
from automask.stats.geometry import GeometryParams
from automask.stats.status_as_mask import StatusAsMaskParams
from automask.stats.variance import VarianceParams
from automask.viz import show_mask

MASKING_PIPELINE = Pipeline(
    channels=[
        Channel("geometry", GeometryParams(pad=2, frac=0.4), field_reg=None),
        Channel("status_as_mask", StatusAsMaskParams(pad=2), field_reg=None),
        Channel("variance", VarianceParams(k=3.5, mode="low"), field_reg="tv"),
        Channel(
            "asic_polish",
            AsicPolishParams(asic=256, n_iter=3, k=15.0, mode="high"),
            field_reg=["blob_scale"],
            field_reg_params=[BlobScaleParams()],
            mask_reg=["fill_holes", "area_gate"],
            mask_reg_params=[FillHolesParams(), AreaGateParams()],
        ),
    ],
    combiner="union",
)

print("needs:", MASKING_PIPELINE.needs())
sample = Sample.from_store(RUN, LIT_SELECTION, MASKING_PIPELINE.needs(), store=store)
computed_mask = MASKING_PIPELINE.run(sample)
mask_ax = show_mask(
    computed_mask, title=f"Run {RUN:04d}: {computed_mask.mean():.2%} masked"
)
save_figure(mask_ax.figure, f"run_{RUN:04d}_computed_mask.png")
plt.show()

# %% [markdown]
# ## 4. Validation
#
# %%
from automask.evaluation import MaskValidationDesign, ParameterSweep, validate_mask

validation_report = validate_mask(
    MASKING_PIPELINE,
    RUN,
    selection=LIT_SELECTION,
    store=store,
    design=MaskValidationDesign(
        sweeps=(
            ParameterSweep(
                "variance.params.k",
                (3.0, 4.0),
                "Check the declared +/-0.5 threshold tolerance",
            ),
            ParameterSweep(
                "variance.field_reg.tv.weight",
                (2.0, 8.0),
                "Check nearby smoothing strengths",
            ),
        ),
        n_folds=10,
    ),
)
validation_report.display()
MASKING_PIPELINE = validation_report.recommended_pipeline
