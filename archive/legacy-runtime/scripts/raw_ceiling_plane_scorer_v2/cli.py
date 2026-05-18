from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import prototype_raw_ceiling_plane_scorer as legacy

from .config import GlobalSelectionConfig, ScorerV2Config
from .reporting import write_outputs
from .runner import score_corpus_v2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relation-first raw ceiling plane scorer (V2)."
    )
    parser.add_argument("--uuid", help="Score a single building UUID")
    parser.add_argument("--buildings", type=Path, default=legacy.BUILDINGS_PATH)
    parser.add_argument("--roof-results", type=Path, default=legacy.ROOF_RESULTS_PATH)
    parser.add_argument("--v3-results", type=Path, default=legacy.V3_RESULTS_PATH)
    parser.add_argument(
        "--ridge-eave-scores", type=Path, default=legacy.RIDGE_EAVE_SCORES_PATH
    )
    parser.add_argument("--out-dir", type=Path, default=legacy.OUT_DIR)
    parser.add_argument(
        "--enable-global-selection-ilp",
        action="store_true",
        help="Enable global XY ILP selection for oblique candidate/committed faces.",
    )
    args = parser.parse_args()

    with args.buildings.open() as handle:
        buildings = json.load(handle)
    with args.roof_results.open() as handle:
        roof_results = json.load(handle)

    ridge_eave_scores_by_uuid: dict[str, dict] = {}
    if args.ridge_eave_scores.exists():
        with args.ridge_eave_scores.open() as handle:
            ridge_payload = json.load(handle)
        ridge_eave_scores_by_uuid = {
            str(entry.get("building_uuid")): entry
            for entry in (ridge_payload.get("buildings") or [])
            if entry.get("building_uuid")
        }

    v3_results_by_uuid: dict[str, dict] = {}
    if args.v3_results.exists():
        with args.v3_results.open() as handle:
            v3_payload = json.load(handle)
        v3_results_by_uuid = {
            str(entry.get("building_uuid")): entry
            for entry in v3_payload
            if entry.get("building_uuid")
        }

    if args.uuid:
        buildings = [
            building for building in buildings if str(building.get("uuid")) == args.uuid
        ]
        if not buildings:
            raise SystemExit(f"No building found for UUID {args.uuid}")

    config = ScorerV2Config(
        global_selection=GlobalSelectionConfig(
            enabled=bool(args.enable_global_selection_ilp)
        )
    )
    output = score_corpus_v2(
        buildings,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_eave_scores_by_uuid,
        v3_results_by_uuid=v3_results_by_uuid,
        config=config,
    )
    write_outputs(args.out_dir, output)

    print(f"Targets scored: {len(output.per_target_rows)}")
    print(f"Stories summarized: {len(output.per_story_rows)}")
    print(f"Split pieces: {len(output.split_piece_rows)}")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
