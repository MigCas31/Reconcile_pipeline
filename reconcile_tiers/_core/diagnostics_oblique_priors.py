"""Diagnose how many obliques per building would be dropped by an
architectural-prior filter, evaluated from the tier payload.

For each `pipeline-outputs/<uuid>/tier_payload[_v2].json`:
  - reads `payload.classification.roof_type` (gable, hip, mansard, …)
  - groups `payload.ceiling[]` pieces with source `roof_arrangement*` (v1) or
    `computed_oblique` (v2) by `oblique_idx` parsed from `arrangement_cell_id`
  - reconstructs one oblique per idx (azimuth, inclination, total xz area,
    representative plane and corner set) by unioning the per-room fragments
  - applies two prior filters and reports drop counts:
      A. gable-partner filter: only enforced when roof_type == GABLE; each
         oblique must have a `_is_gable_partner_pair` partner
      B. per-type filter: applies the appropriate constraint for each classified
         roof_type (gable, hip, cross_gable, mansard, pyramid, shed, flat,
         complex)

Run:
  python -m reconcile_tiers._core.diagnostics_oblique_priors \\
    [--root pipeline-outputs] \\
    [--out pipeline-outputs/_diagnostics/oblique_priors.json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon
from shapely.ops import unary_union

from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.classify.roof_type import (
    APEX_TOL_M,
    PITCH_BAND_GAP_DEG,
)
from reconcile_tiers.classify.roof_type import (
    AZIMUTH_TOL_DEG as CLASSIFY_AZIMUTH_TOL_DEG,
)
from reconcile_tiers.classify.roof_type import (
    INCL_TOL_DEG as CLASSIFY_INCL_TOL_DEG,
)

GABLE_PARTNER_AZIMUTH_TOL_DEG = 40.0  # build.py:95
GABLE_PARTNER_INCL_TOL_DEG = 10.0  # build.py:96
SHED_MIN_INCL_DEG = 5.0

ARRANGEMENT_SOURCES = {
    "roof_arrangement",
    "roof_arrangement_attic",
    "computed_oblique",  # v2 relabel of roof_arrangement
}
ARRANGEMENT_CELL_RE = re.compile(r"^cell:(\d+)(?::room:\d+:\d+)?$")
ATTIC_FULL_LOC_RE = re.compile(r"::tier-ceiling-roof-arrangement-attic-full::(\d+)$")


def _angle_diff_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


def _axis_diff_deg(a: float, b: float) -> float:
    return _angle_diff_deg(a % 180.0, b % 180.0)


def _azimuth_inclination_from_plane(plane: dict[str, float]) -> tuple[float, float]:
    a = float(plane["a"])
    b = float(plane["b"])
    c = float(plane["c"])
    norm = math.sqrt(a * a + b * b + c * c) or 1.0
    a /= norm
    b /= norm
    c /= norm
    if b < 0:
        a, b, c = -a, -b, -c
    incl_deg = math.degrees(math.acos(max(-1.0, min(1.0, b))))
    azimuth_deg = math.degrees(math.atan2(a, c)) % 360.0
    return azimuth_deg, incl_deg


def _xz_polygon(corners: list[dict[str, float]]) -> Polygon | None:
    if len(corners) < 3:
        return None
    try:
        poly = make_valid(Polygon([(float(c["x"]), float(c["z"])) for c in corners]))
    except Exception:
        return None
    if not isinstance(poly, Polygon) or poly.is_empty or poly.area <= 1e-6:
        return None
    return poly


def _oblique_idx_from_piece(piece: dict[str, Any]) -> int | None:
    cell = piece.get("arrangement_cell_id")
    if cell:
        match = ARRANGEMENT_CELL_RE.match(cell)
        if match:
            return int(match.group(1))
    locator = piece.get("locator_id", "")
    match = ATTIC_FULL_LOC_RE.search(locator)
    if match:
        # attic-full keys collide with arrangement-cell keys; offset to a
        # disjoint range so we don't accidentally union them.
        return 100_000 + int(match.group(1))
    return None


def _reconstruct_obliques(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct one entry per source oblique surface from the payload's
    arrangement-fragmented pieces. Returns dicts with azimuth_deg, incl_deg,
    plane, total_xz_area, fragment_count, max_y, min_y, idx, sample_corners.
    """
    by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for piece in payload.get("ceiling", []):
        if piece.get("source") not in ARRANGEMENT_SOURCES:
            continue
        idx = _oblique_idx_from_piece(piece)
        if idx is None:
            continue
        by_idx[idx].append(piece)

    obliques: list[dict[str, Any]] = []
    for idx, pieces in sorted(by_idx.items()):
        polys: list[Polygon] = []
        for piece in pieces:
            poly = _xz_polygon(piece["corners"])
            if poly is not None:
                polys.append(poly)
        if not polys:
            continue
        xz_union = make_valid(unary_union(polys))
        if xz_union.is_empty:
            continue
        rep = max(pieces, key=lambda p: len(p["corners"]))
        azimuth, incl = _azimuth_inclination_from_plane(rep["plane"])
        ys = [float(c["y"]) for piece in pieces for c in piece["corners"]]
        obliques.append(
            {
                "idx": idx,
                "azimuth_deg": azimuth,
                "incl_deg": incl,
                "plane": rep["plane"],
                "xz_area_m2": float(xz_union.area),
                "fragment_count": len(pieces),
                "min_y": min(ys) if ys else 0.0,
                "max_y": max(ys) if ys else 0.0,
                "locator_id": rep.get("locator_id"),
            }
        )
    return obliques


# ----- prior-fit logic --------------------------------------------------------


def _gable_partner_pair(a: dict[str, Any], b: dict[str, Any]) -> bool:
    az_diff = _angle_diff_deg(a["azimuth_deg"], b["azimuth_deg"])
    if (
        not (180.0 - GABLE_PARTNER_AZIMUTH_TOL_DEG)
        <= az_diff
        <= (180.0 + GABLE_PARTNER_AZIMUTH_TOL_DEG)
    ):
        return False
    if abs(a["incl_deg"] - b["incl_deg"]) > GABLE_PARTNER_INCL_TOL_DEG:
        return False
    return True


def _has_partner(target: dict[str, Any], obliques: list[dict[str, Any]]) -> bool:
    return any(
        _gable_partner_pair(target, other) for other in obliques if other is not target
    )


def _pair_axis(o: dict[str, Any]) -> float:
    return o["azimuth_deg"] % 180.0


def _opposing_pair_indices(obliques: list[dict[str, Any]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(obliques):
        for j, b in enumerate(obliques[i + 1 :], start=i + 1):
            az_diff = _angle_diff_deg(a["azimuth_deg"], b["azimuth_deg"])
            if abs(az_diff - 180.0) > CLASSIFY_AZIMUTH_TOL_DEG:
                continue
            if abs(a["incl_deg"] - b["incl_deg"]) > CLASSIFY_INCL_TOL_DEG:
                continue
            pairs.append((i, j))
    return pairs


def _hip_perpendicular_partners(obliques: list[dict[str, Any]]) -> set[int]:
    """Return indices of obliques that participate in at least one
    perpendicular pair-of-pairs (i.e. fit the hip / cross-gable topology).
    """
    pairs = _opposing_pair_indices(obliques)
    if len(pairs) < 2:
        return set()
    accepted: set[int] = set()
    for idx_a, pair_a in enumerate(pairs):
        for pair_b in pairs[idx_a + 1 :]:
            axis_a = _pair_axis(obliques[pair_a[0]])
            axis_b = _pair_axis(obliques[pair_b[0]])
            if abs(_axis_diff_deg(axis_a, axis_b) - 90.0) <= CLASSIFY_AZIMUTH_TOL_DEG:
                accepted.update(pair_a)
                accepted.update(pair_b)
    return accepted


def _mansard_tier_partners(obliques: list[dict[str, Any]]) -> set[int]:
    """Mansard-style: obliques cluster into two pitch tiers. Drop those that
    don't fall in either tier *and* lack a partner inside their tier.
    """
    if len(obliques) < 2:
        return set()
    inclinations = sorted([(o["incl_deg"], i) for i, o in enumerate(obliques)])
    incls = [val for val, _ in inclinations]
    if not incls:
        return set()
    gaps = [(incls[i + 1] - incls[i], i) for i in range(len(incls) - 1)]
    if not gaps:
        return set()
    max_gap, split_idx = max(gaps)
    if max_gap < PITCH_BAND_GAP_DEG:
        # Single tier; fall back to gable-partner test
        return {i for i, o in enumerate(obliques) if _has_partner(o, obliques)}
    lower = {idx for _, idx in inclinations[: split_idx + 1]}
    upper = {idx for _, idx in inclinations[split_idx + 1 :]}
    accepted: set[int] = set()
    for tier in (lower, upper):
        tier_obs = [obliques[i] for i in tier]
        for i in tier:
            if _has_partner(obliques[i], tier_obs):
                accepted.add(i)
    return accepted


def _pyramid_partners(obliques: list[dict[str, Any]]) -> set[int]:
    """Pyramid: obliques converge to a shared apex point. We only check that
    each oblique's max-y point is within APEX_TOL_M of the median apex.
    """
    if len(obliques) < 3:
        return set()
    apexes = []
    for o in obliques:
        apexes.append((o["max_y"], o["idx"]))
    if not apexes:
        return set()
    median_y = sorted(y for y, _ in apexes)[len(apexes) // 2]
    return {
        i
        for i, o in enumerate(obliques)
        if abs(o["max_y"] - median_y) <= APEX_TOL_M * 4.0
    }


def filter_a_gable_only(
    obliques: list[dict[str, Any]], roof_type: str
) -> tuple[set[int], set[int]]:
    """Filter A — partner check enforced only for buildings classified as
    GABLE. Returns (kept_indices, dropped_indices). Same never-strand
    safeguard as filter B.
    """
    n = len(obliques)
    all_idx = set(range(n))
    if (roof_type or "").lower() != "gable":
        return all_idx, set()
    if n < 2:
        return all_idx, set()
    kept = {i for i, o in enumerate(obliques) if _has_partner(o, obliques)}
    if not kept and n > 0:
        return all_idx, set()
    return kept, all_idx - kept


def _min_obliques_for_type(roof_type: str) -> int:
    """How many obliques the classified type structurally requires before
    the prior is even meaningful. Below this, the building is more likely
    misclassified than genuinely violating the prior."""
    return {
        "gable": 2,
        "hip": 4,
        "cross_gable": 4,
        "mansard": 4,
        "pyramid": 4,
    }.get((roof_type or "").lower(), 0)


def filter_b_per_type(
    obliques: list[dict[str, Any]], roof_type: str
) -> tuple[set[int], set[int]]:
    """Filter B — per-classified-type prior. Returns (kept, dropped).

    Safeguards:
      1. Skip the filter when the building has fewer obliques than the type
         structurally requires (likely misclassification, not a violation).
      2. Never strand: if applying the prior would keep zero obliques, fall
         back to keeping everything.
    """
    rt = (roof_type or "").lower()
    n = len(obliques)
    all_idx = set(range(n))
    if n < _min_obliques_for_type(rt):
        return all_idx, set()
    if rt in ("none", "flat"):
        return set(), all_idx
    if rt == "shed":
        if not obliques:
            return set(), set()
        kept = {max(range(n), key=lambda i: obliques[i]["xz_area_m2"])}
    elif rt == "gable":
        kept = {i for i, o in enumerate(obliques) if _has_partner(o, obliques)}
    elif rt in ("hip", "cross_gable"):
        kept = _hip_perpendicular_partners(obliques)
        for i, o in enumerate(obliques):
            if _has_partner(o, obliques) and o["xz_area_m2"] > 1.0:
                kept.add(i)
    elif rt == "mansard":
        kept = _mansard_tier_partners(obliques)
    elif rt == "pyramid":
        kept = _pyramid_partners(obliques)
    elif rt == "complex":
        kept = set()
        for i, o in enumerate(obliques):
            if _has_partner(o, obliques) or o["xz_area_m2"] >= 4.0:
                kept.add(i)
    else:
        return all_idx, set()
    # Never-strand safeguard.
    if not kept and n > 0:
        return all_idx, set()
    return kept, all_idx - kept


# ----- aggregation ------------------------------------------------------------


def analyse_payload(payload: dict[str, Any]) -> dict[str, Any]:
    classification = payload.get("classification", {}) or {}
    roof_type = (classification.get("roof_type") or "").lower()
    obliques = _reconstruct_obliques(payload)

    kept_a, dropped_a = filter_a_gable_only(obliques, roof_type)
    kept_b, dropped_b = filter_b_per_type(obliques, roof_type)

    return {
        "uuid": payload.get("uuid"),
        "roof_type": roof_type,
        "n_oblique": len(obliques),
        "obliques": [
            {
                "idx": o["idx"],
                "azimuth_deg": round(o["azimuth_deg"], 1),
                "incl_deg": round(o["incl_deg"], 1),
                "xz_area_m2": round(o["xz_area_m2"], 2),
                "fragment_count": o["fragment_count"],
                "min_y": round(o["min_y"], 2),
                "max_y": round(o["max_y"], 2),
                "filter_a_keep": (i in kept_a),
                "filter_b_keep": (i in kept_b),
                "locator_id": o["locator_id"],
            }
            for i, o in enumerate(obliques)
        ],
        "filter_a": {"kept": len(kept_a), "dropped": len(dropped_a)},
        "filter_b": {"kept": len(kept_b), "dropped": len(dropped_b)},
        "all_pass_a": len(dropped_a) == 0,
        "all_pass_b": len(dropped_b) == 0,
    }


def aggregate(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_type_a: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_type_b: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    drop_dist_a: dict[int, int] = defaultdict(int)
    drop_dist_b: dict[int, int] = defaultdict(int)
    n_buildings_with_a_drops = 0
    n_buildings_with_b_drops = 0
    most_b_drops: list[tuple[int, str, str]] = []
    for record in records:
        rt = record["roof_type"] or "unknown"
        a = record["filter_a"]
        b = record["filter_b"]
        by_type_a[rt]["buildings"] += 1
        by_type_a[rt]["obliques_total"] += a["kept"] + a["dropped"]
        by_type_a[rt]["obliques_kept"] += a["kept"]
        by_type_a[rt]["obliques_dropped"] += a["dropped"]
        by_type_b[rt]["buildings"] += 1
        by_type_b[rt]["obliques_total"] += b["kept"] + b["dropped"]
        by_type_b[rt]["obliques_kept"] += b["kept"]
        by_type_b[rt]["obliques_dropped"] += b["dropped"]
        drop_dist_a[a["dropped"]] += 1
        drop_dist_b[b["dropped"]] += 1
        if a["dropped"] > 0:
            n_buildings_with_a_drops += 1
        if b["dropped"] > 0:
            n_buildings_with_b_drops += 1
            most_b_drops.append(
                (b["dropped"], record.get("uuid_dir") or record["uuid"], rt)
            )
    most_b_drops.sort(reverse=True)
    return {
        "by_type_filter_a": {k: dict(v) for k, v in by_type_a.items()},
        "by_type_filter_b": {k: dict(v) for k, v in by_type_b.items()},
        "drop_distribution_a": dict(drop_dist_a),
        "drop_distribution_b": dict(drop_dist_b),
        "n_buildings_with_a_drops": n_buildings_with_a_drops,
        "n_buildings_with_b_drops": n_buildings_with_b_drops,
        "top_b_drops": most_b_drops[:15],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("pipeline-outputs"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pipeline-outputs/_diagnostics/oblique_priors.json"),
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for uuid_dir in sorted(args.root.iterdir()):
        if not uuid_dir.is_dir() or uuid_dir.name.startswith("_"):
            continue
        payload_path = uuid_dir / "tier_payload.json"
        if not payload_path.is_file():
            continue
        try:
            payload = json.loads(payload_path.read_text())
        except Exception as exc:
            skipped.append(f"{payload_path}: {exc}")
            continue
        record = analyse_payload(payload)
        record["uuid_dir"] = uuid_dir.name
        records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "buildings_examined": len(records),
        "aggregate": aggregate(records),
        "per_building": records,
    }
    if skipped:
        summary["skipped"] = skipped
    args.out.write_text(json.dumps(summary, indent=2, default=float))

    agg = summary["aggregate"]
    print(f"\n=== {summary['buildings_examined']} buildings ===")
    print(f"  buildings with A-drops (gable-only): {agg['n_buildings_with_a_drops']}")
    print(f"  buildings with B-drops (per-type):   {agg['n_buildings_with_b_drops']}")
    print("  by roof_type — filter A (gable-only):")
    for rt, stats in sorted(agg["by_type_filter_a"].items()):
        print(
            f"    {rt:>15s}: {stats['buildings']:>4d} buildings, "
            f"{stats['obliques_dropped']:>4d}/{stats['obliques_total']:>4d} "
            f"obliques dropped"
        )
    print("  by roof_type — filter B (per-type):")
    for rt, stats in sorted(agg["by_type_filter_b"].items()):
        print(
            f"    {rt:>15s}: {stats['buildings']:>4d} buildings, "
            f"{stats['obliques_dropped']:>4d}/{stats['obliques_total']:>4d} "
            f"obliques dropped"
        )
    print("  drop-count distribution — filter B:")
    for k in sorted(agg["drop_distribution_b"].keys()):
        print(f"    {k} dropped: {agg['drop_distribution_b'][k]} buildings")
    print("  top 15 buildings by B-drops:")
    for n, uuid_dir, rt in agg["top_b_drops"]:
        print(f"    {uuid_dir} ({rt}): {n} dropped")
    if skipped:
        print(f"\n  skipped {len(skipped)} payloads (see {args.out} 'skipped')")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
