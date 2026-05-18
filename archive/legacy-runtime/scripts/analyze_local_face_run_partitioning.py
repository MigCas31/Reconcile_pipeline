#!/usr/bin/env python3
"""Analyze local face-run partition signals for multi-partner ridge/eave cases.

This script combines:

* the selected ridge/eave pair graph
* the multi-partner node analysis
* the raw eave-supported split-piece outputs

to answer a specific modeling question:

When one selected plane-group pairs with more than one selected partner,
does the local face-run partition already exist in the supported split pieces?

We classify each multi-partner node into one of four practical buckets:

* ``already_partitioned_by_plane_group``:
  every involved plane-group has exactly one supported chain-signature, and
  those signatures are disjoint across plane-groups.

* ``intra_plane_only``:
  at least one involved plane-group has multiple supported chain-signatures,
  but there is no signature shared across plane-groups.

* ``cross_target_overlap_only``:
  every involved plane-group has a single supported chain-signature, but one
  or more signatures are shared across plane-groups.

* ``intra_plane_and_cross_target_overlap``:
  at least one plane-group spans multiple chain-signatures AND at least one
  signature is shared across plane-groups.

The last bucket is the strongest evidence that we need two stages:

1. split a plane-group into local face runs using supported chain families
2. resolve overlap/ownership between same-side runs locally
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_multi_partner_plane_groups import (
    analyze_multi_partner_plane_groups,
)
from scripts.prototype_raw_ceiling_plane_scorer import (
    BUILDINGS_PATH,
    RIDGE_EAVE_SCORES_PATH,
    ROOF_RESULTS_PATH,
    V3_RESULTS_PATH,
    score_buildings,
)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / ".context" / "local_face_run_partitioning_analysis.json"


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _plane_group_token(element_id: str) -> str:
    if "::plane-group::" not in element_id:
        return element_id
    return element_id.split("::plane-group::", 1)[1]


def _index_supported_piece_signatures(
    split_piece_rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    indexed: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in split_piece_rows:
        target_element_id = str(row.get("target_element_id") or "")
        if "::ridge-eave-candidate::plane-group::" not in target_element_id:
            continue
        if row.get("piece_role") != "supported":
            continue
        target_token = _plane_group_token(target_element_id)
        if not target_token:
            continue
        indexed[str(row["uuid"])][target_token].append(row)
    return indexed


def _summarize_target_signatures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature_area: dict[tuple[str, ...], float] = defaultdict(float)
    signature_piece_ids: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in rows:
        signature = tuple(
            sorted(str(chain_id) for chain_id in (row.get("chain_ids") or []))
        )
        if not signature:
            continue
        signature_area[signature] += float(row.get("area_xz_m2") or 0.0)
        signature_piece_ids[signature].append(str(row.get("piece_id") or ""))
    return [
        {
            "chain_signature": list(signature),
            "area_xz_m2": round(area, 6),
            "piece_ids": sorted(piece_ids),
        }
        for signature, area, piece_ids in sorted(
            (
                (signature, signature_area[signature], signature_piece_ids[signature])
                for signature in signature_area
            ),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]


def _classify_node_partition_case(
    target_summaries: list[dict[str, Any]],
) -> tuple[str, dict[str, list[str]]]:
    multi_signature_targets = [
        str(target["plane_group_id"])
        for target in target_summaries
        if len(target["supported_chain_signatures"]) > 1
    ]
    signature_targets: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for target in target_summaries:
        plane_group_id = str(target["plane_group_id"])
        for signature in target["supported_chain_signatures"]:
            signature_targets[tuple(signature["chain_signature"])].add(plane_group_id)
    shared_signatures = [
        {
            "chain_signature": list(signature),
            "plane_group_ids": sorted(plane_group_ids),
        }
        for signature, plane_group_ids in sorted(signature_targets.items())
        if len(plane_group_ids) > 1
    ]

    if multi_signature_targets and shared_signatures:
        case = "intra_plane_and_cross_target_overlap"
    elif multi_signature_targets:
        case = "intra_plane_only"
    elif shared_signatures:
        case = "cross_target_overlap_only"
    else:
        case = "already_partitioned_by_plane_group"
    return case, {
        "multi_signature_plane_group_ids": sorted(multi_signature_targets),
        "shared_signatures": shared_signatures,
    }


def _signature_relation(a: set[str], b: set[str]) -> str:
    if a == b:
        return "equal"
    if a.issubset(b):
        return "subset"
    if b.issubset(a):
        return "superset"
    if a.isdisjoint(b):
        return "disjoint"
    return "partial_overlap"


def analyze_local_face_run_partitioning(
    *,
    buildings: list[dict[str, Any]],
    roof_results: dict[str, Any],
    ridge_eave_scores: dict[str, Any],
    v3_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    multi_partner = analyze_multi_partner_plane_groups(
        buildings, roof_results, ridge_eave_scores
    )
    cohort_uuids = {str(row["uuid"]) for row in multi_partner["buildings"]}
    buildings_subset = [
        building for building in buildings if str(building.get("uuid")) in cohort_uuids
    ]
    ridge_by_uuid = {
        str(entry.get("building_uuid")): entry
        for entry in (ridge_eave_scores.get("buildings") or [])
        if entry.get("building_uuid")
    }
    v3_by_uuid = {
        str(entry.get("building_uuid")): entry
        for entry in (v3_results or [])
        if isinstance(entry, dict) and entry.get("building_uuid")
    }

    (
        _rows,
        _per_story_rows,
        _chain_rows,
        _plane_chain_rows,
        split_piece_rows,
        _ownership_rows,
        split_summary,
    ) = score_buildings(
        buildings_subset,
        roof_results,
        ridge_eave_scores_by_uuid=ridge_by_uuid,
        v3_results_by_uuid=v3_by_uuid,
    )
    split_index = _index_supported_piece_signatures(split_piece_rows)

    case_counts: Counter[str] = Counter()
    case_counts_by_arch_context: dict[str, Counter[str]] = defaultdict(Counter)
    within_target_signature_relation_counts: Counter[str] = Counter()
    across_target_signature_relation_counts: Counter[str] = Counter()
    subpart_vs_signature_counts: Counter[str] = Counter()
    node_rows: list[dict[str, Any]] = []

    for building_row in multi_partner["buildings"]:
        uuid = str(building_row["uuid"])
        subpart_count = len(
            ((roof_results.get(uuid) or {}).get("roof_coverage_graph") or {}).get(
                "subparts"
            )
            or []
        )
        for node in building_row["multi_partner_plane_groups"]:
            involved_plane_group_ids = [
                str(node["plane_group_id"]),
                *[
                    str(plane_group_id)
                    for plane_group_id in node["neighbor_plane_group_ids"]
                ],
            ]
            target_summaries: list[dict[str, Any]] = []
            for plane_group_id in involved_plane_group_ids:
                target_token = _plane_group_token(plane_group_id)
                signature_rows = _summarize_target_signatures(
                    split_index[uuid].get(target_token, [])
                )
                target_summaries.append(
                    {
                        "plane_group_id": plane_group_id,
                        "supported_chain_signatures": signature_rows,
                    }
                )
            distinct_signatures = {
                tuple(signature["chain_signature"])
                for target in target_summaries
                for signature in target["supported_chain_signatures"]
            }
            if len(distinct_signatures) < subpart_count:
                subpart_vs_signature_counts["signatures_lt_subparts"] += 1
            elif len(distinct_signatures) > subpart_count:
                subpart_vs_signature_counts["signatures_gt_subparts"] += 1
            else:
                subpart_vs_signature_counts["signatures_eq_subparts"] += 1

            for target in target_summaries:
                signatures = [
                    set(signature["chain_signature"])
                    for signature in target["supported_chain_signatures"]
                ]
                for idx, sig_a in enumerate(signatures):
                    for sig_b in signatures[idx + 1 :]:
                        within_target_signature_relation_counts[
                            _signature_relation(sig_a, sig_b)
                        ] += 1
            for idx, target_a in enumerate(target_summaries):
                signatures_a = [
                    set(signature["chain_signature"])
                    for signature in target_a["supported_chain_signatures"]
                ]
                for target_b in target_summaries[idx + 1 :]:
                    signatures_b = [
                        set(signature["chain_signature"])
                        for signature in target_b["supported_chain_signatures"]
                    ]
                    for sig_a in signatures_a:
                        for sig_b in signatures_b:
                            across_target_signature_relation_counts[
                                _signature_relation(sig_a, sig_b)
                            ] += 1

            case, details = _classify_node_partition_case(target_summaries)
            case_counts[case] += 1
            case_counts_by_arch_context[str(building_row["architectural_context"])][
                case
            ] += 1
            node_rows.append(
                {
                    "uuid": uuid,
                    "architectural_context": str(building_row["architectural_context"]),
                    "node_plane_group_id": str(node["plane_group_id"]),
                    "degree": int(node["degree"]),
                    "node_pattern": str(node["pattern"]),
                    "neighbor_plane_group_ids": [
                        str(plane_group_id)
                        for plane_group_id in node["neighbor_plane_group_ids"]
                    ],
                    "partition_case": case,
                    "distinct_signature_count": len(distinct_signatures),
                    "roof_coverage_subpart_count": subpart_count,
                    **details,
                    "targets": target_summaries,
                }
            )

    examples_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(
        node_rows,
        key=lambda item: (
            item["partition_case"],
            -item["degree"],
            item["uuid"],
            item["node_plane_group_id"],
        ),
    ):
        case = str(row["partition_case"])
        if len(examples_by_case[case]) >= 8:
            continue
        examples_by_case[case].append(
            {
                "uuid": row["uuid"],
                "architectural_context": row["architectural_context"],
                "node_plane_group_id": row["node_plane_group_id"],
                "degree": row["degree"],
                "node_pattern": row["node_pattern"],
                "neighbor_plane_group_ids": row["neighbor_plane_group_ids"],
                "multi_signature_plane_group_ids": row[
                    "multi_signature_plane_group_ids"
                ],
                "shared_signatures": row["shared_signatures"],
            }
        )

    return {
        "summary": {
            "n_multi_partner_buildings": len(cohort_uuids),
            "n_multi_partner_nodes": len(node_rows),
            "partition_case_counts": dict(case_counts),
            "partition_case_counts_by_architectural_context": {
                context: dict(counter)
                for context, counter in sorted(case_counts_by_arch_context.items())
            },
            "within_target_signature_relation_counts": dict(
                within_target_signature_relation_counts
            ),
            "across_target_signature_relation_counts": dict(
                across_target_signature_relation_counts
            ),
            "subpart_vs_signature_counts": dict(subpart_vs_signature_counts),
            "split_piece_summary": split_summary,
        },
        "examples_by_case": dict(examples_by_case),
        "nodes": node_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--roof-results", type=Path, default=ROOF_RESULTS_PATH)
    parser.add_argument(
        "--ridge-eave-scores", type=Path, default=RIDGE_EAVE_SCORES_PATH
    )
    parser.add_argument("--v3-results", type=Path, default=V3_RESULTS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    buildings = _load_json(args.buildings)
    roof_results = _load_json(args.roof_results)
    ridge_eave_scores = _load_json(args.ridge_eave_scores)
    v3_results = _load_json(args.v3_results) if args.v3_results.exists() else []

    payload = analyze_local_face_run_partitioning(
        buildings=buildings,
        roof_results=roof_results,
        ridge_eave_scores=ridge_eave_scores,
        v3_results=v3_results,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
