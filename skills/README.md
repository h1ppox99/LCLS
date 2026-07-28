---
name: xray-agentic-skillset
description: Skill set for an agent doing serial X-ray / Jungfrau1M scattering analysis directly from XTC (no psana). Five skill categories — masking, normalization, selection, verification, qa — each a folder with a README index and one md file per method, all following the same v3 method-file template. Grounded in the Run0475 (xppl1016922) workflow.
category: index
role: skill-set-index
---

# X-ray Agentic Analysis — Skill Set

A self-contained skill set for an agent that reduces area-detector scattering data
(Jungfrau1M, `(2, 512, 1024)`, LCLS `xppl1016922` Run0475) into clean 1D/2D results,
working **directly from raw XTC** without `psana`, and then judges the results.

**Each category is a folder; each method inside it is its own file**, and every file
follows the same template (see [Method-file template](#method-file-template-v3) below):

| Category (folder) | Question it answers | Acts on | Methods | Pipeline status |
|---|---|---|---|---|
| [masking/](masking/README.md) | *Which pixels are untrustworthy?* | pixels (spatial) | 7 files | wired (mask agent) |
| [normalization/](normalization/README.md) | *How do I put shots on a common scale?* | intensities (scale) | 4 files | wired (reduction agent) |
| [selection/](selection/README.md) | *Which shots do I keep?* | whole shots | 2 files | wired (reduction agent) |
| [verification/](verification/README.md) | *Is the endpoint scientifically usable?* | I(q) metrics | 1 criterion | wired (verifier agent) |
| [qa/](qa/SKILL.md) | *Are the 1D curves physically reasonable?* | curves + frames | 10 methods | **not wired** (ported from the old manifest workflow) |

```
skills/
├── README.md                     ← you are here (skill-set index + template spec)
├── masking/                      ← README + 00–06 (7 methods)
├── normalization/                ← README + 01–04 (monitor menu, 4 reference classes)
├── selection/                    ← README + 01a/01b (2 agent-decided cuts)
├── verification/                 ← README + 01 (endpoint criteria, machine thresholds)
└── qa/                           ← index + methods/ (10 checks) + scripts/
```

## Method-file template (v3)

Every method file has YAML frontmatter with exactly these fields:

```yaml
---
name: xray-<category>-<method-slug>   # globally unique; registrable as a standalone skill
description: one line — what it does + when to use it + the key calibrated fact
category: masking | normalization | selection | verification | qa
role: <role within the category>      # e.g. signal-independent, gate, agent-decided cut
gate: <when it runs>                  # "always", or the concrete triggering condition
status: wired | not-wired             # is it reachable from agent/entrypoint.py today
---
```

Body sections, in this **fixed order** (omit a section only when it has no content;
`###` subsections are free):

| # | Section | Holds |
|---|---|---|
| 0 | *(intro text after the H1)* | 1–3 lines of relationships / context, no heading |
| 1 | `## The decision the agent owns` | *agent-decided methods only* — the blockquoted question |
| 2 | `## Principle` | what it does and why (the physics) |
| 3 | `## Parameters` | knobs, defaults, and why — as a table where possible |
| 4 | `## Decision rules` | run/skip conditions, verdict logic, evidence to gather |
| 5 | `## Implementation` | reference code, when the file carries it |
| 6 | `## Evidence (Run0475)` | calibrated results, worked examples, validation |
| 7 | `## When to use` | applicability |
| 8 | `## Trade-offs` | limitations, failure modes, escalation |
| 9 | `## Do not` | hard guardrails, when they exist |
| 10 | `## Outputs` | what it writes/logs (report fields, layers, decisions.json keys) + repo artifacts |
| 11 | `## Machine block` | machine-readable thresholds as a ```json block (parsed by code — exactly one per file) |
| 12 | `## Links` | Part of / Companion / Complements / Feeds cross-links |

The H1 is `# <Category> · <NN> — <Title>` where `<NN>` matches the filename prefix.
Category `README.md` files carry frontmatter (`role: category-index`) and the sections
**Goal → `## Methods` (table) → `## Choosing` → `## Golden rules`**, plus
category-specific extras (comparison tables, output contracts).

Rules the template encodes:

- **Numbers live in files, not in agents** — thresholds change only by editing the
  method file (a human-reviewed act), never inside a run.
- **Machine-readable where code reads it**: anything parsed by `agent/entrypoint.py`
  (e.g. `verification/01_iq_quality.md`) keeps its single ```json block intact.
- **Skipping is allowed; silent skipping is not** — every gated method states its
  `gate:` and its skip-reporting expectation.

## Framework evolution: v1 → v2 → v3

The skill set went through three structural versions.

### v1: all methods bundled into a single md per category

```
agent_skills/
├── masking.md         ← all masking methods crammed into one file
├── normalization.md   ← all normalization methods crammed into one file
└── selection.md       ← all selection methods crammed into one file
```

One `name`/`description` frontmatter covered an entire category — the whole category
could only be referenced as a single skill, and reading one method meant scrolling
through hundreds of unrelated lines.

### v2: one folder per category, one md per method

Every method got its own file and its own frontmatter (registrable as a standalone
skill), with a `README.md` index per folder and cross-links by relative path.

| Dimension | v1 bundled | v2 split per method |
|---|---|---|
| Smallest referenceable unit | whole category | single method |
| Reading one method | open big file, scroll | open the matching small file |
| Editing one method | disturbs neighbors | touches one file only |
| Navigation | in-file headings | README index + cross-links |

### v3: one template across all categories (current)

v2 solved granularity but let each category grow its own dialect — different
frontmatter fields, different section names for the same concept (`Result` /
`Evidence` / `verdict` / `Validated feasibility`), different index-file conventions.
v3 fixes the **format**: one frontmatter schema, one section vocabulary and order
(the table above), numbered filenames reflecting execution order, machine thresholds
always in a ```json block. Same content, uniform shape — an agent (or a human) can
now parse any method file the same way.

## How the categories compose

```
raw XTC ──▶ [SELECTION: shot quality]            ── drop dropouts / bright tail (01a/01b)
        ──▶ [NORMALIZATION: per-shot flux]       ── divide each shot by its own I0
        ──▶ [MASKING: pixel masks]               ── exclude bad/outlier pixels
        ──▶ azimuthal integration                ── I(q) endpoint
        ──▶ [VERIFICATION: endpoint criteria]    ── pass / fail + feedback loop
        ──▶ [QA: curve + detector checks]        ── physical-reasonableness report (not wired yet)
```

## Golden rules (apply across all categories)

1. **Normalize per-shot before summing.** Summing then dividing once is only valid
   when the incident flux is constant. See
   [normalization/01](normalization/01_per_shot_flux_ipm2.md).
2. **A mask is not a selection.** [Masking](masking/README.md) removes *pixels* on a frame;
   [selection](selection/README.md) removes *whole shots/events*. Never fold flux dropouts
   into the pixel mask or vice-versa.
3. **Always run an end-to-end alignment check.** Per-shot total scattering vs `ipm2`
   should correlate strongly (≈0.87 here); a near-zero correlation means the event
   ordering is wrong and every downstream weight lands on the wrong frame.
4. **Report what each method removed.** Log pixel/shot counts and percentages; silent
   truncation reads as "kept everything" when it did not.

## Repo pointers

In **this** repo the executable counterparts live in `pipeline/`
(`step0_xtc_to_npy.py`, `step0b_extend_shot_table.py`, `step2_accumulate.py`,
`step3_apply_mask.py`, `step4_iq.py`) and `agent/entrypoint.py`; validated runs are
under `outputs/`. The `Repo`/provenance pointers inside method files
(`masking_gap_geometry/`, `weighted_sum_v2.py`, `accumulate_calib.py`, …) refer to the
**parent research folder** `Summer2026/X-ray/X_ray_Agentic_Analysis/`, where the
methods were first developed and validated.
