"""Simulate promoting `committed_residual` pieces that live entirely inside
a target's `source_part_ids` at the same story.

Scratch script for the debug-element skill — counts how many pieces the
fix would repair / leave unchanged / regress (overlap with other final
pieces) without modifying any pipeline code.

Inputs:
  .context/raw_ceiling_plane_scorer_v2_full/plane_extent_splits.json

The proposed fix (see skill output 2026-04-23): in
``scripts/raw_ceiling_plane_scorer_v2/layer_policy.py`` around line 630-635,
before demoting a ``committed_oblique`` piece whose ``piece_role != 'supported'``
to ``committed_residual``, check whether the piece sits entirely inside a
building-part that already carries eave-chain support for this target. If so,
keep it as final.

Promotion predicate (layer_policy level):
    target_kind == 'committed_oblique'
  AND piece_role == 'residual'
  AND final_layer_reason == 'committed_residual'
  AND set(piece_part_ids_story) <= set(source_part_ids)  # non-empty both sides
  AND area_xz_m2 >= MIN_AREA  # skip slivers

Regression check: compare the promoted polygon against every other
``final_layer=True`` piece in the same building. If the promoted polygon's
intersection area with another final piece exceeds OVERLAP_MIN_M2, count it
as a regression (risk of double-rendered roof / z-fight).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon

DEFAULT_SIDECAR = Path(
    ".context/raw_ceiling_plane_scorer_v2_full/plane_extent_splits.json"
)

MIN_AREA_M2 = 0.5  # cohort cutoff — skip sub-half-m² slivers
OVERLAP_MIN_M2 = 0.25  # intersection-area threshold to flag regression


def _poly_from_row(row: dict[str, Any]) -> Polygon | None:
    corners = row.get("corners") or []
    if not corners or len(corners) < 3:
        return None
    shell = [(float(c[0]), float(c[2])) for c in corners]
    holes_field = row.get("holes") or []
    hole_rings: list[list[tuple[float, float]]] = []
    for hole in holes_field:
        if not hole or len(hole) < 3:
            continue
        hole_rings.append([(float(c[0]), float(c[2])) for c in hole])
    poly = Polygon(shell, holes=hole_rings)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    return poly


def _is_cohort_residual(row: dict[str, Any]) -> bool:
    return (
        row.get("target_kind") == "committed_oblique"
        and row.get("piece_role") == "residual"
        and row.get("final_layer_reason") == "committed_residual"
        and (row.get("area_xz_m2") or 0.0) >= MIN_AREA_M2
    )


def _would_promote(row: dict[str, Any]) -> bool:
    """Apply the proposed promotion predicate."""
    if not _is_cohort_residual(row):
        return False
    source_parts = set(row.get("source_part_ids") or [])
    piece_parts_story = set(row.get("piece_part_ids_story") or [])
    if not source_parts or not piece_parts_story:
        return False
    return piece_parts_story <= source_parts


def simulate(sidecar_path: Path) -> dict[str, Any]:
    data = json.loads(sidecar_path.read_text())
    buildings = data["buildings"]

    summary: dict[str, Any] = {
        "n_buildings": len(buildings),
        "cohort_pieces": 0,
        "repaired": 0,
        "unchanged_part_guard": 0,
        "unchanged_coplanar_guard": 0,
        "regressed": 0,
        "repaired_area_m2": 0.0,
        "coplanar_guard_area_m2": 0.0,
        "regressed_area_m2": 0.0,
        "per_building": [],
        "repaired_examples": [],
        "regressed_examples": [],
        "unchanged_examples": [],
        "coplanar_guard_examples": [],
    }

    for uuid, pieces in buildings.items():
        # Build final polygons for overlap check. Keep the plane coefficients so
        # we can later distinguish coplanar overlap (real z-fight) from ridge
        # meetings between non-parallel planes (false positive in 2D).
        final_polys: list[
            tuple[str, Polygon, tuple[float, float, float, float] | None]
        ] = []
        for p in pieces:
            if not p.get("final_layer"):
                continue
            poly = _poly_from_row(p)
            if poly is None or poly.area < 1e-6:
                continue
            coeffs = p.get("target_plane_coeffs")
            peer_plane = tuple(float(c) for c in coeffs) if coeffs else None
            final_polys.append((p.get("piece_id", ""), poly, peer_plane))

        b_row = {
            "uuid": uuid,
            "cohort": 0,
            "repaired": 0,
            "unchanged_part_guard": 0,
            "unchanged_coplanar_guard": 0,
            "regressed": 0,
        }

        for p in pieces:
            if not _is_cohort_residual(p):
                continue
            b_row["cohort"] += 1
            summary["cohort_pieces"] += 1

            area = float(p.get("area_xz_m2") or 0.0)
            piece_id = p.get("piece_id", "")

            if not _would_promote(p):
                b_row["unchanged_part_guard"] += 1
                summary["unchanged_part_guard"] += 1
                if len(summary["unchanged_examples"]) < 12:
                    summary["unchanged_examples"].append(
                        {
                            "uuid": uuid,
                            "piece_id": piece_id,
                            "area": round(area, 3),
                            "source_part_ids": p.get("source_part_ids"),
                            "piece_part_ids_story": p.get("piece_part_ids_story"),
                        }
                    )
                continue

            candidate_poly = _poly_from_row(p)
            if candidate_poly is None or candidate_poly.area < 1e-6:
                b_row["unchanged_part_guard"] += 1
                summary["unchanged_part_guard"] += 1
                continue

            # Regression: promoted polygon overlaps a COPLANAR final piece.
            # Non-parallel planes that meet at a ridge/valley project to the
            # same XZ region without actually sharing 3D surface — ignore those.
            cand_coeffs = p.get("target_plane_coeffs")
            cand_plane = tuple(float(c) for c in cand_coeffs) if cand_coeffs else None
            max_overlap_area = 0.0
            max_overlap_peer = None
            for peer_id, peer_poly, peer_plane in final_polys:
                if peer_id == piece_id:
                    continue
                if cand_plane is not None and peer_plane is not None:
                    # Compare unit normals' dot product.
                    import math

                    a1, b1, c1, _ = cand_plane
                    a2, b2, c2, _ = peer_plane
                    n1 = math.sqrt(a1 * a1 + b1 * b1 + c1 * c1)
                    n2 = math.sqrt(a2 * a2 + b2 * b2 + c2 * c2)
                    if n1 > 0 and n2 > 0:
                        cosang = (a1 * a2 + b1 * b2 + c1 * c2) / (n1 * n2)
                        # Skip peers that are > 10° off parallel (or anti-parallel)
                        if abs(cosang) < math.cos(math.radians(10.0)):
                            continue
                try:
                    inter = candidate_poly.intersection(peer_poly)
                except Exception:
                    continue
                ia = float(getattr(inter, "area", 0.0) or 0.0)
                if ia > max_overlap_area:
                    max_overlap_area = ia
                    max_overlap_peer = peer_id

            # Refined predicate: skip promotion if the residual would overlap a
            # coplanar final piece (same target, coplanar ceiling-oblique, ridge
            # mirror). These stay as `committed_residual` under the proposed fix.
            if max_overlap_area >= OVERLAP_MIN_M2:
                b_row["unchanged_coplanar_guard"] += 1
                summary["unchanged_coplanar_guard"] += 1
                summary["coplanar_guard_area_m2"] += area
                if len(summary["coplanar_guard_examples"]) < 12:
                    summary["coplanar_guard_examples"].append(
                        {
                            "uuid": uuid,
                            "piece_id": piece_id,
                            "area": round(area, 3),
                            "overlap_area": round(max_overlap_area, 3),
                            "overlap_peer": max_overlap_peer,
                        }
                    )
            else:
                b_row["repaired"] += 1
                summary["repaired"] += 1
                summary["repaired_area_m2"] += area
                if len(summary["repaired_examples"]) < 12:
                    summary["repaired_examples"].append(
                        {
                            "uuid": uuid,
                            "piece_id": piece_id,
                            "area": round(area, 3),
                            "overlap_area": round(max_overlap_area, 3),
                        }
                    )

        if b_row["cohort"] > 0:
            summary["per_building"].append(b_row)

    return summary


def _print_report(s: dict[str, Any]) -> None:
    print(f"Buildings scanned:         {s['n_buildings']}")
    print(f"Cohort pieces (≥{MIN_AREA_M2} m²):    {s['cohort_pieces']}")
    print(
        f"  REPAIRED (now final):              {s['repaired']}   total "
        f"{s['repaired_area_m2']:.1f} m²"
    )
    print(f"  UNCHANGED · part-id guard:         {s['unchanged_part_guard']}")
    print(
        f"  UNCHANGED · coplanar overlap guard:{s['unchanged_coplanar_guard']}   total "
        f"{s['coplanar_guard_area_m2']:.1f} m²"
    )
    print(
        f"  REGRESSED (residual):              {s['regressed']}   total "
        f"{s['regressed_area_m2']:.1f} m²"
    )
    print()

    buildings_with_cohort = len(s["per_building"])
    print(f"Buildings with ≥1 cohort piece: {buildings_with_cohort}")
    print()

    print("Top repaired examples:")
    for ex in s["repaired_examples"][:10]:
        print(
            f"  {ex['uuid'][:8]}… piece={ex['piece_id'].split('::')[-1]}  "
            f"area={ex['area']:.2f} m²  max_overlap={ex['overlap_area']:.2f} m²"
        )
    print()
    print("Top coplanar-guard examples (predicate refinement skipped promotion):")
    for ex in s["coplanar_guard_examples"][:10]:
        print(
            f"  {ex['uuid'][:8]}… piece={ex['piece_id'].split('::')[-1]}  "
            f"area={ex['area']:.2f} m²  overlap={ex['overlap_area']:.2f} m²  "
            f"peer={ex['overlap_peer']}"
        )
    print()
    print("Top regressed examples:")
    for ex in s["regressed_examples"][:10]:
        print(
            f"  {ex['uuid'][:8]}… piece={ex['piece_id'].split('::')[-1]}  "
            f"area={ex['area']:.2f} m²  overlap={ex['overlap_area']:.2f} m²  "
            f"peer={ex['overlap_peer']}"
        )
    print()
    print("Top part-guard unchanged examples (predicate did not fire):")
    for ex in s["unchanged_examples"][:8]:
        print(
            f"  {ex['uuid'][:8]}… piece={ex['piece_id'].split('::')[-1]}  "
            f"area={ex['area']:.2f} m²"
            f"  source_parts={ex['source_part_ids']}"
            f"  piece_parts_story={ex['piece_part_ids_story']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".context/sim_committed_residual_promotion.json"),
    )
    args = parser.parse_args(argv)

    if not args.sidecar.exists():
        print(f"sidecar not found: {args.sidecar}", file=sys.stderr)
        return 1

    s = simulate(args.sidecar)
    _print_report(s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(s, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
