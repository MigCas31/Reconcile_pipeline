from pathlib import Path

import pytest

from reconcile_tiers.build_internals.envelope_pieces import _building_dict, _roof_dict
from reconcile_tiers.classify.tiers import classify_building
from reconcile_tiers.extract.building import extract_building_model
from reconcile_tiers.roof.roof import build_roof_model


@pytest.mark.parametrize(
    ("uuid", "legacy_flat_count"),
    [
        ("c72ad855-9e52-46f1-886d-a9f37911521f", 6),
        ("f40dcc9f-b97b-4bef-8b40-ba011aabf0bd", 10),
        ("2ea3b759-e047-424c-8034-f8ee5b811fb4", 11),
    ],
)
def test_flat_count_matches_legacy_cohort_within_one(uuid: str, legacy_flat_count: int):
    model = extract_building_model(uuid, Path("pipeline-outputs"), Path(".scan-cache"))

    roof = build_roof_model(model)

    assert abs(len(roof.flat) - legacy_flat_count) <= 1


def test_oblique_count_after_priors_for_c72():
    """The architectural-prior filter (`roof.priors.filter_obliques_by_roof_type`)
    drops obliques that don't fit the classified building type. For
    `c72ad855` (gable), the south-facing low-pitch surface that has no
    mirror partner gets dropped, leaving the 2 main gable slopes.
    """
    model = extract_building_model(
        "c72ad855-9e52-46f1-886d-a9f37911521f",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    roof = build_roof_model(model)

    assert len(roof.oblique) == 2


def test_complex_roof_drops_stranded_oblique_inside_partner_field():
    """A perpendicular single face over an existing gable pair is scan noise, not
    a separate complex roof wing — the priors filter must drop it.

    Phase 5 per-wing fan-out reorders source_index across wings, so we
    assert the *count* is 2 (one gable pair survived; the stranded one
    dropped) rather than pinning specific indices. The behavioural
    invariant is the same: post-priors, exactly two obliques remain.
    """
    model = extract_building_model(
        "893d4535-3169-4907-a93a-b3ab8f66ec1c",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    roof = build_roof_model(model)

    assert len(roof.oblique) == 2


def test_arrangement_splits_c72_obliques_by_room_cells():
    model = extract_building_model(
        "c72ad855-9e52-46f1-886d-a9f37911521f",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    roof = build_roof_model(model)

    # Two slopes split across many rooms — should still produce many cells.
    assert len(roof.oblique_split) >= 10
    assert len(roof.oblique_split) > len(roof.oblique)


def test_legacy_tier7_gable_pair_survives_new_roof_pipeline():
    model = extract_building_model(
        "019e1376-9762-42d6-8520-b664b8c752df",
        Path("pipeline-outputs"),
        Path(".scan-cache"),
    )

    result = classify_building(
        _building_dict(model), _roof_dict(build_roof_model(model))
    )

    assert result["signals"]["has_gable"] is True
    assert result["tier"] == 7


def test_roof_layer_does_not_import_legacy_graph_or_hypothesis_stages():
    import importlib
    import pkgutil

    import reconcile_tiers.roof

    forbidden = (
        "graph",
        "hypothesis",
        "evidence",
        "coverage",
        "cell_complex",
        "building_part",
        "flat_role",
    )
    for module_info in pkgutil.walk_packages(
        reconcile_tiers.roof.__path__, "reconcile_tiers.roof."
    ):
        module = importlib.import_module(module_info.name)
        module_text = " ".join(str(value) for value in module.__dict__.values())
        assert not any(token in module_text for token in forbidden), module_info.name
