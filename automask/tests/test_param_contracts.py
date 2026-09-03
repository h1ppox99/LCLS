import pytest

from automask.interface.catalog import capability_catalog
from automask.evaluation import MaskValidationDesign, ParameterSweep
from automask.mask import Pipeline, production_pipeline
from automask.interface.recipes import (
    pipeline_from_dict,
    pipeline_to_dict,
    require_run_floor,
    selection_from_dict,
    selection_to_dict,
    validation_design_from_dict,
    validation_design_to_dict,
)
from automask.selection.shot_selection import (
    CONDITION_OPERATORS,
    Condition,
    PercentileTrim,
    ShotSelection,
)


# -- inline param dict <-> object conversion ------------------------------
def test_selection_dict_round_trip():
    original = ShotSelection(
        where=(
            Condition("state", "in", ("lit", "sample")),
            Condition("intensity", "finite"),
        ),
        trim=PercentileTrim("intensity", low=0.1, high=0.2),
        n_shots=3,
        normalization="intensity",
    )

    restored = selection_from_dict(selection_to_dict(original))

    assert restored == original


def test_pipeline_dict_round_trip():
    original = production_pipeline()

    restored = pipeline_from_dict(pipeline_to_dict(original))

    assert pipeline_to_dict(restored) == pipeline_to_dict(original)
    require_run_floor(restored)


def test_pipeline_from_dict_rejects_unknown_params_and_missing_floor():
    payload = pipeline_to_dict(production_pipeline())
    payload["channels"][0]["params"]["not_a_parameter"] = 1
    with pytest.raises(ValueError, match="unknown params"):
        pipeline_from_dict(payload)

    with pytest.raises(ValueError, match="missing"):
        require_run_floor(Pipeline([]))


def test_validation_dict_round_trip():
    original = MaskValidationDesign(
        sweeps=(ParameterSweep("variance.params.k", (3.0, 4.0), "nearby tolerance"),),
        n_folds=4,
        strategies=("round_robin",),
        prune_redundant=False,
    )

    restored = validation_design_from_dict(validation_design_to_dict(original))

    assert restored == original


# -- catalog (pure library introspection) ---------------------------------
def test_capability_catalog_exposes_registry_and_parameter_examples():
    catalog = capability_catalog()

    assert catalog["kind"] == "automask_capability_catalog"
    assert catalog["conventions"]["run_floor"] == ["geometry", "status_as_mask"]
    assert catalog["statistics"]["variance"]["needs"] == ["std"]
    assert catalog["regularizers"]["tv"]["kind"] == "field"
    assert catalog["selection"]["operators"] == list(CONDITION_OPERATORS)
    assert "channels" in catalog["parameter_examples"]["pipeline"]
    assert "where" in catalog["parameter_examples"]["selection"]
