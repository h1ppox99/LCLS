"""Machine-readable introspection of the registered automask capabilities.

The catalog is pure library introspection -- registered statistics, regularizers,
selection operators, reductions, default parameters, and the field-native
parameter shapes that ``ShotSelection``/``Pipeline`` accept. It carries no agent
policy and no I/O; the agent exposes it through the ``automask_catalog`` tool and
a human can call :func:`capability_catalog` directly from a script.
"""

from __future__ import annotations

from automask.image_store import REDUCTIONS
from automask.masking import production_pipeline
from automask.recipes import (
    pipeline_to_dict,
    selection_to_dict,
    to_plain,
    validation_design_to_dict,
)
from automask.regularization.base import REGULARIZERS
from automask.shot_selection import CONDITION_OPERATORS, ShotSelection
from automask.stats.base import STATS


def _registry_entry(spec, **metadata) -> dict:
    return {
        **metadata,
        "default_params": to_plain(spec.params()),
        "documentation": spec.doc,
    }


def capability_catalog() -> dict:
    from automask.evaluation import MaskValidationDesign

    return {
        "schema_version": 1,
        "kind": "automask_capability_catalog",
        "conventions": {
            "mask": "boolean ndarray; True means masked",
            "reductions": "assembled detector space",
            "calibration_constants": "native panel space",
            "run_floor": sorted(
                name for name, spec in STATS.items() if spec.kind == "floor"
            ),
            "protected_output_trees": ["xtc", "calib", "hdf5", "xpp_sharing"],
        },
        "statistics": {
            name: _registry_entry(
                spec,
                kind=spec.kind,
                emits_mask=spec.emits_mask,
                needs=list(spec.needs),
                default_mode=spec.mode,
            )
            for name, spec in sorted(STATS.items())
        },
        "regularizers": {
            name: _registry_entry(spec, kind=spec.kind)
            for name, spec in sorted(REGULARIZERS.items())
        },
        "selection": {
            "operators": list(CONDITION_OPERATORS),
            "conditions": "all conditions are ANDed",
            "normalization": "excludes non-finite and zero values",
        },
        "reductions": sorted(REDUCTIONS),
        "parameter_examples": {
            "selection": selection_to_dict(ShotSelection()),
            "pipeline": pipeline_to_dict(production_pipeline()),
            "validation": validation_design_to_dict(MaskValidationDesign()),
        },
    }
