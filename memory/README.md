# Workflow memory

Durable, **measured** findings from past runs of this pipeline. Each entry is one
decision the agents should not have to rediscover — and, in the cases recorded here,
one they got wrong or nearly got wrong once already.

`decisions.jsonl` is the source of truth (one JSON object per line).
`agent/memory.py` renders the entries for a phase and **`agent/entrypoint.py` injects
them verbatim into that phase's prompt**, so an agent sees them before it reads any
skill file. That injection matters: the skill-usage traces
(`outputs/<run>/skill_usage.md`) show the agents read skill files only in a single
burst in the first seconds of a phase and never reopen them, so a memory file that
merely *exists* would not be consulted.

## Entries

| id | phase | what it settles |
|---|---|---|
| **MEM-001** | mask | On this campaign, azimuthal sigma-clipping must mask **negative outliers only** — positives are the LaB6 Bragg rings. Masking all outliers drops ring 847 px contrast 1.40 → 0.32 and ring 1245 px 3.33 → 1.03, both below the verifier's C2 minima, i.e. the run FAILS. |
| **MEM-002** | verify | The nominal 190 mm sample–detector distance is **+3.5 % off** (D_eff = 196.6 mm, two rings agreeing to 0.13 %). Pixel positions are sound; the printed q axis is ~3.4 % high. |

## Fields

| field | meaning |
|---|---|
| `id`, `date`, `phase` | identity, and which agent gets it injected |
| `status` | `active` (injected) or `retired` (kept for history, not injected) |
| `confidence` | `measured` — backed by numbers in `evidence`; nothing else is injected |
| `rule` | the imperative the agent must follow |
| `evidence` | the measurement, with the numbers, that established the rule |
| `why` | the physics/statistics, so the agent can tell when the rule stops applying |
| `scope` | the conditions under which it holds — **read this before generalizing** |
| `conflict_resolved` | when the entry settles a disagreement between two skill files |
| `source` | files and runs to go back to |

## Rules for writing an entry

1. **Measured only.** An entry carries numbers from a real run. A hunch is not memory.
2. **State the scope.** Every rule here is campaign-specific until proven otherwise;
   MEM-001 is true for a LaB6 calibration and false for an amorphous pump–probe product.
   An entry without a scope line will be misapplied.
3. **Memory does not override a skill's thresholds.** It records which of several
   admissible choices this campaign measured to be right. Changing a threshold is still
   a human-reviewed edit to the criterion file.
4. **Retire, don't delete.** If a run disproves an entry, set `status: retired`, add the
   contradicting evidence, and write the replacement as a new id.
