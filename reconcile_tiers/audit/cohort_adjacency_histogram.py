"""Dump the per-building adjacency-tag histogram across the corpus.

Reads every `pipeline-outputs/*/tier_payload.json`, counts adjacency tags
on every envelope element, and prints a CSV row per building. Use this to
sanity-check the tagger after a change:

    python -m reconcile_tiers.audit.cohort_adjacency_histogram > cohort_adj.csv

The script also prints an `--assertions` block to stderr that flags the
cases the plan documents as physically impossible (no `EXTERNAL_AIR` walls
on any building, `UNHEATED_ATTIC` on a tier-1 flat-roof building, etc.).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from reconcile_tiers.payload.schema import AdjacencyKind

ENVELOPE_TAGS = [kind.value for kind in AdjacencyKind]


def _count_payload(data: dict) -> Counter[str]:
    counts: Counter[str] = Counter()
    for room in data.get("rooms", []):
        floor = room.get("floor")
        # Schema migration: room.floor used to be a single HorizontalLid; now
        # it is list[HorizontalLid]. Tolerate both shapes.
        floor_pieces = (
            floor
            if isinstance(floor, list)
            else [floor]
            if isinstance(floor, dict)
            else []
        )
        for piece in floor_pieces:
            if isinstance(piece, dict) and "adjacency" in piece:
                counts[piece["adjacency"]] += 1
        for wall in room.get("walls", []):
            if "adjacency" in wall:
                counts[wall["adjacency"]] += 1
    for collection in (
        "ceiling",
        "knee_walls",
        "dormer_faces",
        "gable_closures",
        "gaps",
    ):
        for item in data.get(collection, []):
            if "adjacency" in item:
                counts[item["adjacency"]] += 1
    return counts


def _flag_anomalies(
    uuid: str, tier: int, has_basement: bool, counts: Counter[str]
) -> list[str]:
    flags: list[str] = []
    external_walls = counts.get(AdjacencyKind.EXTERNAL_AIR.value, 0)
    if external_walls == 0:
        flags.append("no_external_air_elements")
    if tier == 1 and counts.get(AdjacencyKind.UNHEATED_ATTIC.value, 0) > 0:
        flags.append("unheated_attic_on_flat_roof_tier_1")
    basement_wall_count = counts.get(
        AdjacencyKind.BASEMENT_WALL_GROUND_SHALLOW.value, 0
    ) + counts.get(AdjacencyKind.BASEMENT_WALL_GROUND_DEEP.value, 0)
    if has_basement and basement_wall_count == 0:
        flags.append("has_basement_but_no_basement_walls")
    if not has_basement and basement_wall_count > 0:
        flags.append("no_basement_but_basement_walls_present")
    return flags


def _has_basement(data: dict) -> bool:
    labels = data.get("story_labels", []) or []
    return any(isinstance(s, str) and s.lower().startswith("kælder") for s in labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", default="pipeline-outputs", type=Path)
    parser.add_argument(
        "--out", default="-", type=str, help="CSV output path or '-' for stdout"
    )
    args = parser.parse_args()

    rows = []
    anomalies: list[tuple[str, list[str]]] = []
    for entry in sorted(args.pipeline_dir.iterdir()):
        payload_path = entry / "tier_payload.json"
        if not payload_path.exists():
            continue
        try:
            data = json.loads(payload_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        uuid = data.get("uuid", entry.name)
        tier = int(data.get("classification", {}).get("tier", 0) or 0)
        has_basement = _has_basement(data)
        counts = _count_payload(data)
        row = {
            "uuid": uuid,
            "tier": tier,
            "has_basement": int(has_basement),
            **{tag: counts.get(tag, 0) for tag in ENVELOPE_TAGS},
        }
        rows.append(row)
        flagged = _flag_anomalies(uuid, tier, has_basement, counts)
        if flagged:
            anomalies.append((uuid, flagged))

    if not rows:
        print("no payloads found", file=sys.stderr)
        return 1

    fieldnames = ["uuid", "tier", "has_basement", *ENVELOPE_TAGS]
    out_stream = sys.stdout if args.out == "-" else open(args.out, "w")
    try:
        writer = csv.DictWriter(out_stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if out_stream is not sys.stdout:
            out_stream.close()

    print(
        f"\n--- assertions ({len(anomalies)} flagged of {len(rows)} buildings) ---",
        file=sys.stderr,
    )
    for uuid, flags in anomalies:
        print(f"{uuid}: {', '.join(flags)}", file=sys.stderr)
    return 0 if not anomalies else 2


if __name__ == "__main__":
    sys.exit(main())
