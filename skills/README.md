---
name: xray-agentic-skillset
description: Skill set for an agent doing serial X-ray / Jungfrau1M scattering analysis directly from XTC (no psana). Three skill categories — masking, normalization, selection — each holding at least 3 concrete, tested methods grounded in the Run0475 (xppl1016922) workflow.
---

# X-ray Agentic Analysis — Skill Set

A self-contained skill set for an agent that reduces area-detector scattering data
(Jungfrau1M, `(2, 512, 1024)`, LCLS `xppl1016922` Run0475) into clean 1D/2D results,
working **directly from raw XTC** without `psana`.

The reduction always factors into three orthogonal decisions. Keep them separate —
mixing them is the most common source of wrong results. **Each category is a folder;
each method inside it is its own file:**

| Category (folder) | Question it answers | Acts on | Methods |
|---|---|---|---|
| [masking/](masking/README.md) | *Which pixels are untrustworthy?* | pixels (spatial) | 5 files |
| [normalization/](normalization/README.md) | *How do I put shots / pixels on a common scale?* | intensities (scale) | 4 files |
| [selection/](selection/README.md) | *Which shots / events / regions do I keep?* | shots + events + q-range | 5 files |

Each category folder has a `README.md` index and one MD file **per method**. Every
method file gives: the principle, the hyperparameters and why, when to reach for it,
its trade-offs, and the concrete artifact/script in this repo that implements it.

```
agent_skills/
├── README.md                     ← you are here (skill-set index)
├── masking/
│   ├── README.md                 ← masking index
│   ├── 00_status_baseline.md
│   ├── 01_geometry_gap.md
│   ├── 02_pyfai_azimuthal_sigmaclip.md
│   ├── 03_rmm_dark_based.md
│   └── 04_rmm_feature6_light.md
├── normalization/
│   ├── README.md                 ← monitor menu: one file per reference parameter
│   ├── 01_per_shot_flux_ipm2.md
│   ├── 02_per_shot_flux_upstream.md
│   ├── 03_per_shot_flux_alternatives.md
│   └── 04_per_shot_flux_self.md
└── selection/
    ├── README.md
    ├── 01a_low_ipm_exclusion.md   (agent decides)
    ├── 01b_high_ipm_exclusion.md  (agent decides)
    ├── 02_event_temporal_alignment.md
    ├── 03_good_pixel.md
    ├── 04_q_range.md
    └── 05_correlation_gate.md
```

## Framework evolution: from "one bundled file" to "one file per method"

This skill set went through two structural versions. The core change is **granularity** —
from "one big file per category" to "one file per method".

### v1 (before): all methods bundled into a single md

```
agent_skills/
├── masking.md         ← all 4 masking methods crammed into this one file
├── normalization.md   ← all 4 normalization methods crammed into this one file
└── selection.md       ← all 5 selection methods crammed into this one file
```

- **One category = one file**, methods separated only by headings (`## Method 1`,
  `## Method 2`, …).
- To read a single method (say pyFAI sigma-clip) you open the whole `masking.md` and scroll
  through hundreds of lines.
- One `name`/`description` frontmatter covers the entire file — so **the whole category can
  only be referenced as a single skill**; you cannot point at one method on its own.

### v2 (now): one folder per category, one md per method

```
masking/
├── README.md                       ← category index (method table + when to use which)
├── 00_status_baseline.md           ← each method is its own file,
├── 01_geometry_gap.md                 each with its own name/description frontmatter
├── 02_pyfai_azimuthal_sigmaclip.md
├── 03_rmm_dark_based.md
└── 04_rmm_feature6_light.md
```

- **One category = one folder; one method = one file**, plus a `README.md` index inside the
  folder.
- Every method file has **its own frontmatter**, so it can be **invoked/retrieved as a
  standalone skill** (granularity down to the method).
- Methods **cross-link** by relative path (e.g. selection's flux-quality →
  normalization's per-shot-flux).

### Side-by-side comparison

| Dimension | v1 bundled single file | v2 split per method |
|---|---|---|
| Structure | 3 files, category-level | 3 folders + 14 method files + 4 READMEs |
| Smallest referenceable unit | **whole category** | **single method** |
| Frontmatter granularity | one per category | one per method (registrable as its own skill) |
| Reading one method | open big file, scroll | open the matching small file directly |
| Adding/editing one method | edit a shared big file, easy to disturb neighbors | touch one file only, others untouched |
| Navigation | in-file headings | category README index + cross-links between methods |
| Cost | fewer files, but bloated | more files, index READMEs to maintain |

### Why the finer split is better (for an agent)

1. **Addressability** — the agent can load exactly one method instead of pulling the whole
   category into context: fewer tokens, tighter focus.
2. **Composability** — each method is a standalone unit that can be referenced, swapped, and
   versioned on its own; when `selection` cites one `normalization` method, the link lands on
   that one file.
3. **Maintainability** — tuning pyFAI's `thres` or adding a new masking method touches only
   the relevant small file, never the others.
4. **Bounded cost** — the added "more files, needs an index" overhead is absorbed by each
   folder's `README.md` index table.

> In one line: v1 was "three lecture handouts"; v2 is "a tabbed binder with a table of
> contents" — same content, but every page can be pulled out and used on its own.

## How the categories compose

```
raw XTC ──▶ [SELECTION: event alignment]        ── sort L1Accepts to acquisition order
        ──▶ [NORMALIZATION: gain/pedestal calib] ── raw ADU → calibrated frame
        ──▶ [SELECTION: shot quality]            ── drop dropouts/spikes/bad flux
        ──▶ [NORMALIZATION: per-shot flux]       ── divide each shot by its own I0
        ──▶ [MASKING: pixel masks]               ── exclude bad/outlier pixels
        ──▶ [NORMALIZATION: geometric]           ── solid-angle / polarization on integration
        ──▶ [SELECTION: q-range]                 ── restrict radial integration window
        ──▶ clean S(q) / S(q,χ)
```

## Golden rules (apply across all three skills)

1. **Normalize per-shot before summing.** Summing then dividing once is only valid
   when the incident flux is constant. See
   [normalization/01_per_shot_flux_ipm2.md](normalization/01_per_shot_flux_ipm2.md).
2. **A mask is not a selection.** [Masking](masking/README.md) removes *pixels* on a frame;
   [selection](selection/README.md) removes *whole shots/events*. Never fold flux dropouts
   into the pixel mask or vice-versa.
3. **Always run an end-to-end alignment check.** Per-shot total scattering vs `ipm2`
   should correlate strongly (≈0.87 here); a near-zero correlation means the event
   ordering is wrong and every downstream weight lands on the wrong frame.
4. **Report what each method removed.** Log pixel/shot counts and percentages; silent
   truncation reads as "kept everything" when it did not.

## Repo pointers

- `xtc_explore.py`, `accumulate_calib.py` — pure-Python XTC parse + calibrated accumulate
- `weighted_sum_v2.py` — per-shot flux weighting + shot selection
- `masking_gap_geometry/`, `masking_pyFAI_sigmaclip/`, `masking_RMM_darkbased/`,
  `masking_RMM_feature6_light/` — the four masking methods, each with its own README
- `Normalization_method.md`, `Normalization_方法与对比.md` — normalization write-ups
