"""Phase 0 cohort audits for reconcile_tiers planning.

Reads the existing producer artefacts (buildings_3d.json +
roof_algorithms_py_results.json)
and the existing classifier (reconcile.complexity_tiers.classify_building) to gate
Phase D/E/F decisions.

Outputs a JSON report at <repo-root>/.context/phase0_audit_results.json plus a
printed summary. (`.context/` is gitignored — the JSON is for local review;
re-running the script regenerates it from the committed artefacts.)

Audits:
  1. Story-index disagreement: V1 stories_found vs. max(oblique.dominant_story) + 1
     (a proxy for roof_pipeline's has_floor_above story view).
  2. oblique_split coverage: of buildings with any obliques, fraction with
  oblique_split[].
  3. Loftrum thermal-cap incidence: count buildings with ceiling.thermal[*].kind ==
  "thermal-cap".
  4. Tier-8 cohort: enumerate buildings the current classifier puts in tier 8 with their
     (n_oblique, n_flat, surface azimuths/inclinations) so we can hand-pick HIP/MANSARD
     samples.

Usage:
    cd <repo-root>
    python -m reconcile_tiers.scripts.phase0_audits

Findings from the 2026-04-26 run are recorded in:
  - tracking_progress.md (entry 2026-04-26)
  - reconcile_tiers/TRACKING.md (Decision log)
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDINGS_PATH = REPO_ROOT / "reconcile" / "buildings_3d.json"
ROOF_PATH = REPO_ROOT / "reconcile" / "roof_algorithms_py_results.json"
OUTPUT_PATH = REPO_ROOT / ".context" / "phase0_audit_results.json"

sys.path.insert(0, str(REPO_ROOT))


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _load() -> tuple[dict[str, dict], dict[str, dict]]:
    print(f"Loading {BUILDINGS_PATH} ({BUILDINGS_PATH.stat().st_size / 1e6:.1f} MB)...")
    with BUILDINGS_PATH.open() as f:
        bldgs_raw = json.load(f)
    bldgs = {b["uuid"]: b for b in bldgs_raw if "uuid" in b}

    print(f"Loading {ROOF_PATH} ({ROOF_PATH.stat().st_size / 1e6:.1f} MB)...")
    with ROOF_PATH.open() as f:
        roofs = json.load(f)

    return bldgs, roofs


def audit_story_disagreement(bldgs: dict, roofs: dict) -> dict:
    """V1 stories_found vs. roof's max(dominant_story)+1 view."""
    matches = 0
    disagreements = []
    no_oblique = 0
    no_v1_stories = 0

    for uuid, b in bldgs.items():
        v1 = _safe_int(b.get("stories_found"))
        roof = roofs.get(uuid) or {}
        oblique = (roof.get("roof_surfaces") or {}).get("oblique") or []
        dom_stories = [
            _safe_int(s.get("dominant_story"))
            for s in oblique
            if _safe_int(s.get("dominant_story")) is not None
        ]
        if v1 is None:
            no_v1_stories += 1
            continue
        if not dom_stories:
            no_oblique += 1
            continue
        roof_view = max(dom_stories) + 1
        if v1 == roof_view:
            matches += 1
        else:
            disagreements.append({"uuid": uuid, "v1": v1, "roof_view": roof_view})

    return {
        "total": len(bldgs),
        "no_v1_stories": no_v1_stories,
        "no_oblique": no_oblique,
        "compared": matches + len(disagreements),
        "matches": matches,
        "disagreements": len(disagreements),
        "disagreement_rate_pct": (
            round(100 * len(disagreements) / max(1, matches + len(disagreements)), 1)
        ),
        "samples": disagreements[:10],
    }


def audit_oblique_split_coverage(roofs: dict) -> dict:
    with_obl = 0
    with_split = 0
    only_obl = []
    no_obl = 0

    for uuid, r in roofs.items():
        surfaces = (r or {}).get("roof_surfaces") or {}
        obl = surfaces.get("oblique") or []
        split = surfaces.get("oblique_split") or []
        if not obl:
            no_obl += 1
            continue
        with_obl += 1
        if split:
            with_split += 1
        else:
            only_obl.append(uuid)

    return {
        "total_roofs": len(roofs),
        "no_oblique": no_obl,
        "with_oblique": with_obl,
        "with_oblique_and_split": with_split,
        "split_coverage_pct": (round(100 * with_split / max(1, with_obl), 1)),
        "without_split_count": len(only_obl),
        "without_split_samples": only_obl[:10],
    }


def audit_thermal_cap_incidence(roofs: dict) -> dict:
    counter = Counter()
    cap_uuids = []
    for uuid, r in roofs.items():
        thermals = ((r or {}).get("ceiling") or {}).get("thermal") or []
        kinds_present = {t.get("kind") for t in thermals if t}
        for k in kinds_present:
            counter[k] += 1
        if "thermal-cap" in kinds_present:
            cap_uuids.append(uuid)

    return {
        "total_roofs": len(roofs),
        "kind_counts": dict(counter),
        "thermal_cap_buildings": len(cap_uuids),
        "thermal_cap_pct": round(100 * len(cap_uuids) / max(1, len(roofs)), 1),
        "thermal_cap_samples": cap_uuids[:10],
    }


def audit_tier8_cohort(bldgs: dict, roofs: dict) -> dict:
    """
    Enumerate the current tier-8 'Other' bucket so we can hand-classify HIP / MANSARD.
    """
    try:
        from reconcile.complexity_tiers import classify_building
    except ImportError as e:
        return {"error": f"could not import classify_building: {e}"}

    tier_counts: dict[int, int] = defaultdict(int)
    tier8 = []

    for uuid, b in bldgs.items():
        roof = roofs.get(uuid) or {}
        try:
            res = classify_building(b, roof)
        except Exception:
            tier_counts[-1] += 1
            continue
        tier = int(res.get("tier") or 0)
        tier_counts[tier] += 1
        if tier == 8:
            obl = (roof.get("roof_surfaces") or {}).get("oblique") or []
            obl_summary = []
            for s in obl[:8]:
                cluster = s.get("cluster") or {}
                obl_summary.append(
                    {
                        "az": round(float(cluster.get("avgAzimuth") or 0), 1),
                        "incl": round(float(cluster.get("avgIncl") or 0), 1),
                        "dom_story": s.get("dominant_story"),
                    }
                )
            flats = (roof.get("roof_surfaces") or {}).get("flat") or []
            tier8.append(
                {
                    "uuid": uuid,
                    "address": (
                        b.get("address")
                        if isinstance(b.get("address"), str)
                        else (b.get("address") or {}).get("address")
                    ),
                    "n_oblique": len(obl),
                    "n_flat": len(flats),
                    "stories_found": b.get("stories_found"),
                    "split_level": b.get("split_level"),
                    "oblique": obl_summary,
                }
            )

    return {
        "total": len(bldgs),
        "tier_distribution": dict(sorted(tier_counts.items())),
        "tier8_count": len(tier8),
        "tier8_pct": round(100 * len(tier8) / max(1, len(bldgs)), 1),
        "tier8_samples": tier8[:30],
    }


def main() -> int:
    bldgs, roofs = _load()

    report = {
        "buildings_total": len(bldgs),
        "roof_results_total": len(roofs),
        "audit1_story_disagreement": audit_story_disagreement(bldgs, roofs),
        "audit2_oblique_split_coverage": audit_oblique_split_coverage(roofs),
        "audit3_thermal_cap_incidence": audit_thermal_cap_incidence(roofs),
        "audit4_tier8_cohort": audit_tier8_cohort(bldgs, roofs),
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUTPUT_PATH}\n")

    print("=" * 72)
    print("Phase 0 audit summary")
    print("=" * 72)
    print(f"buildings_3d.json:                {report['buildings_total']} buildings")
    print(f"roof_results:                     {report['roof_results_total']} buildings")
    print()
    a1 = report["audit1_story_disagreement"]
    print("AUDIT 1 — Story index disagreement")
    print(f"  no V1 stories_found:            {a1['no_v1_stories']}")
    print(f"  no obliques (skipped):          {a1['no_oblique']}")
    print(f"  comparable:                     {a1['compared']}")
    print(f"  match:                          {a1['matches']}")
    print(
        f"  disagree:                       {a1['disagreements']}  "
        f"({a1['disagreement_rate_pct']}%)"
    )
    print()
    a2 = report["audit2_oblique_split_coverage"]
    print("AUDIT 2 — oblique_split coverage")
    print(f"  no obliques:                    {a2['no_oblique']}")
    print(f"  with obliques:                  {a2['with_oblique']}")
    print(
        f"  with obliques + split:          {a2['with_oblique_and_split']}  "
        f"({a2['split_coverage_pct']}%)"
    )
    print(f"  with obliques but NO split:     {a2['without_split_count']}")
    print()
    a3 = report["audit3_thermal_cap_incidence"]
    print("AUDIT 3 — Thermal kinds emitted")
    for k, v in sorted(a3["kind_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:30s} {v:4d}")
    print(
        f"  thermal-cap buildings:          {a3['thermal_cap_buildings']}  "
        f"({a3['thermal_cap_pct']}%)"
    )
    print()
    a4 = report["audit4_tier8_cohort"]
    if "error" in a4:
        print(f"AUDIT 4 — Tier distribution: ERROR {a4['error']}")
    else:
        print("AUDIT 4 — Current tier distribution")
        for t, c in a4["tier_distribution"].items():
            print(f"    tier {t}: {c}")
        print(
            f"  tier 8 (Other):                {a4['tier8_count']}  "
            f"({a4['tier8_pct']}%)"
        )
        print(
            f"  See {OUTPUT_PATH} for tier8_samples (n_oblique + cluster az/incl per "
            f"UUID)."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
