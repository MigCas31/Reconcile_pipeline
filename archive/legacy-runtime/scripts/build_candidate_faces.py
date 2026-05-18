#!/usr/bin/env python3
"""Phase A driver — run candidate-face generation over the full corpus.

Loads each building from ``reconcile/reconcile_v3_results.json``,
derives the scan footprint via ``build_building_footprint`` on the
live building dict (same code path the roof pipeline uses), then calls
``reconcile_v3.reconstruction.candidate_faces.build_candidate_faces``.

Outputs
-------
- ``<out>/candidates.json``  per-building candidate face sets (compact)
- ``<out>/summary.md``       aggregate stats and distribution

Usage
-----
    python scripts/build_candidate_faces.py \
        --results reconcile/reconcile_v3_results.json \
        --pipeline-dir pipeline-outputs \
        --out-dir reports/candidate_faces_20260419
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reconcile.extract3d import extract_building
from reconcile.roof_algorithms_py.ceiling_plane_generation import collect_exposed_rooms
from reconcile.roof_algorithms_py.footprint_derivation import build_building_footprint
from reconcile.roof_algorithms_py.simple_slant import identify_simple_slant_rooms
from reconcile.roof_algorithms_py.story_index import (
    build_story_index,
    build_story_stats,
)
from reconcile_v3.reconstruction.candidate_faces import build_candidate_faces
from reconcile_v3.reconstruction.zones import derive_authoritative_zones


def _load_building_and_scan_footprint(
    uuid: str,
    pipeline_dir: Path,
    scan_cache: Path | None,
):
    """Return ``(building, scan_footprint_xz)`` where the footprint is a list
    of ``(x, z)`` pairs or ``None``.
    """
    bldg = extract_building(uuid, pipeline_dir, scan_cache)
    if not bldg:
        return None, None
    story_index = build_story_index(bldg)
    story_stats = build_story_stats(bldg)
    simple_slant = identify_simple_slant_rooms(
        bldg, story_index["has_floor_above"], story_index["all_stories"]
    )
    exclude_rooms = simple_slant["simple_slant_rooms"]
    exposed = collect_exposed_rooms(
        bldg, story_index["has_floor_above"], exclude_room_indices=exclude_rooms
    )
    info = build_building_footprint(exposed, story_stats["story_floor_polys"])
    return bldg, info.get("building_footprint")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", default="reconcile/reconcile_v3_results.json", type=Path
    )
    ap.add_argument("--pipeline-dir", default="pipeline-outputs", type=Path)
    ap.add_argument("--scan-cache", default=".scan-cache", type=Path)
    ap.add_argument(
        "--roof-results", default="reconcile/roof_algorithms_py_results.json", type=Path
    )
    ap.add_argument("--out-dir", default="reports/candidate_faces", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.results}...")
    with args.results.open() as handle:
        corpus = json.load(handle)
    roof_results_by_uuid: dict[str, dict] = {}
    if args.roof_results.exists():
        print(f"Loading {args.roof_results}...")
        with args.roof_results.open() as handle:
            roof_results_by_uuid = json.load(handle)
    if args.limit:
        corpus = corpus[: args.limit]
    scan_cache_root = args.scan_cache if args.scan_cache.exists() else None

    per_building: list[dict] = []
    errors: list[tuple[str, str]] = []
    t0 = time.time()

    for i, building in enumerate(corpus):
        uuid = building["building_uuid"]
        merged = building.get("merged_roof_segments") or []
        bldg = None
        try:
            bldg, fp = _load_building_and_scan_footprint(
                uuid, args.pipeline_dir, scan_cache_root
            )
        except Exception as exc:
            errors.append((uuid, f"footprint: {type(exc).__name__}: {exc}"))
            fp = None
        zones = derive_authoritative_zones(
            building_uuid=uuid,
            roof_building=roof_results_by_uuid.get(uuid) or {},
            bldg=bldg,
        )

        try:
            faces = build_candidate_faces(
                uuid,
                merged,
                fp,
                building_zones=zones,
                building_gaps=building.get("gaps") or [],
            )
        except Exception as exc:
            errors.append((uuid, f"candidates: {type(exc).__name__}: {exc}"))
            traceback.print_exc()
            faces = []

        per_building.append(
            {
                "building_uuid": uuid,
                "n_segments": len(merged),
                "n_candidates": len(faces),
                "n_extended": sum(1 for f in faces if f.extended),
                "n_zones": len(zones),
                "has_scan_footprint": fp is not None,
                "scan_footprint_xz": (
                    [[float(x), float(z)] for (x, z) in fp] if fp else None
                ),
                "zones": zones,
                "candidates": [asdict(f) for f in faces],
            }
        )

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(corpus)} ({time.time() - t0:.1f}s elapsed)")

    elapsed = time.time() - t0

    # Write candidates.json (compact)
    candidates_path = args.out_dir / "candidates.json"
    with candidates_path.open("w") as handle:
        json.dump(per_building, handle)

    # Aggregate stats
    n_total_segments = sum(b["n_segments"] for b in per_building)
    n_total_candidates = sum(b["n_candidates"] for b in per_building)
    n_total_extended = sum(b["n_extended"] for b in per_building)
    n_total_zones = sum(b["n_zones"] for b in per_building)
    n_with_fp = sum(1 for b in per_building if b["has_scan_footprint"])
    n_empty_candidates = sum(1 for b in per_building if b["n_candidates"] == 0)
    cand_counts = [b["n_candidates"] for b in per_building if b["n_candidates"] > 0]

    def _pct(p: int, total: int) -> str:
        return f"{100 * p / total:.1f}%" if total else "—"

    md = [
        "# Phase A — candidate face generation summary",
        "",
        f"Corpus: `{args.results}` — {len(per_building)} buildings "
        f"(elapsed {elapsed:.1f}s).",
        "",
        "## Aggregates",
        "",
        f"- Buildings with scan footprint: **{n_with_fp}** "
        f"({_pct(n_with_fp, len(per_building))})",
        f"- Total merged segments in: {n_total_segments}",
        f"- Total candidate faces out: **{n_total_candidates}** "
        f"({_pct(n_total_candidates, n_total_segments)} vs. segments)",
        f"- Extended (ridge-extrapolated) candidates: **{n_total_extended}** "
        f"({_pct(n_total_extended, n_total_candidates)})",
        f"- Total authoritative zones: **{n_total_zones}**",
        f"- Buildings with zero candidates: {n_empty_candidates}",
        "",
        "## Candidate count distribution (non-empty buildings)",
        "",
    ]
    if cand_counts:
        md += [
            f"- min: {min(cand_counts)}",
            f"- median: {statistics.median(cand_counts):.0f}",
            f"- mean: {statistics.mean(cand_counts):.1f}",
            f"- p90: {statistics.quantiles(cand_counts, n=10)[-1]:.0f}",
            f"- max: {max(cand_counts)}",
        ]

    md += [
        "",
        "## Errors",
        "",
    ]
    if errors:
        for uuid, msg in errors[:30]:
            md.append(f"- `{uuid}` — {msg}")
        if len(errors) > 30:
            md.append(f"- ... and {len(errors) - 30} more")
    else:
        md.append("- none")

    md += [
        "",
        "## Phase B handoff",
        "",
        "Each building's ``candidates`` list feeds directly into the BIP",
        "solver (`reconcile_v3/reconstruction/solver.py`, not yet written).",
        "Per the revised plan, each candidate face carries: plane, clipped",
        "footprint (XZ), area, azimuth/inclination, neighbour IDs",
        "(topology constraint), support (data-term weight), and — once",
        "D1 calibration runs — a GBM prior to be multiplied into the",
        "objective.",
    ]

    md_path = args.out_dir / "summary.md"
    md_path.write_text("\n".join(md))

    print(f"\nWrote {candidates_path}")
    print(f"Wrote {md_path}")
    print(
        f"{n_total_candidates} candidate faces across "
        f"{len(per_building)} buildings "
        f"({n_total_extended} extended)."
    )
    if errors:
        print(f"{len(errors)} errors; see summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
