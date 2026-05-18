"""Side-by-side diff of v1 vs v2 ceiling envelope output for one or more UUIDs.

Runs `build_tier_payload` twice per UUID (envelope_version=v1 then v2), reads
both payloads, and emits a numeric diff that summarises what the v2 redesign
actually changes:

  - per-source area totals (m^2) for v1 ceiling, v2 ceiling, v2 visual_shells
  - locator ids that disappeared from v1 -> v2 (and what their source was)
  - locator ids that appeared in v2 (and what their source/tag is)
  - per-locator coverage delta for shared ids (rare, since v2 renames most)

Outputs:
  .context/envelope-compare/<uuid>/diff.json    machine-readable summary
  .context/envelope-compare/index.html          one-page index across all uuids

Usage:
  python scripts/compare_envelope.py 0a5032e9-... [<uuid> ...]
  python scripts/compare_envelope.py --all   # every uuid in pipeline-outputs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shapely.geometry import Polygon

from reconcile_tiers._core.shapely2 import make_valid
from reconcile_tiers.build import (
    build_many,
    list_uuids,
    output_path_for_uuid_versioned,
)

# Use cwd-relative defaults — scripts/ is a symlink into archive/legacy-runtime/,
# so __file__.resolve() lands in the wrong place. Run from the project root.
PIPELINE_DIR = Path("pipeline-outputs")
SCAN_ROOT = Path(".scan-cache")
OUT_ROOT = Path(".context") / "envelope-compare"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("uuids", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pipeline-dir", default=str(PIPELINE_DIR))
    parser.add_argument("--scan-root", default=str(SCAN_ROOT))
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Diff existing tier_payload.json + tier_payload_v2.json files; skip the "
        "rebuild step.",
    )
    parser.add_argument("-j", "--workers", type=int, default=1)
    args = parser.parse_args(argv)

    pipeline_dir = Path(args.pipeline_dir)
    scan_root = Path(args.scan_root) if args.scan_root else None

    if args.all:
        uuids = list_uuids(pipeline_dir)
    else:
        uuids = list(args.uuids)
    if not uuids:
        parser.error("pass at least one uuid or --all")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.no_build:
        for version in ("v1", "v2"):
            results = build_many(
                uuids,
                pipeline_dir=pipeline_dir,
                scan_root=scan_root,
                force=True,
                envelope_version=version,
                workers=max(1, args.workers),
            )
            for uuid, status, message in results:
                if status == "failed":
                    print(f"[{version}] {uuid}: FAILED {message}", file=sys.stderr)

    summaries = []
    for uuid in uuids:
        v1_path = output_path_for_uuid_versioned(uuid, pipeline_dir, "v1")
        v2_path = output_path_for_uuid_versioned(uuid, pipeline_dir, "v2")
        if not v1_path.exists() or not v2_path.exists():
            print(f"{uuid}: missing payload(s)", file=sys.stderr)
            continue
        diff = diff_payloads(
            json.loads(v1_path.read_text()), json.loads(v2_path.read_text())
        )
        out_dir = OUT_ROOT / uuid
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "diff.json").write_text(json.dumps(diff, indent=2, sort_keys=True))
        summaries.append((uuid, diff))
        print(_format_summary_line(uuid, diff))

    (OUT_ROOT / "index.html").write_text(_render_index(summaries))
    return 0


def diff_payloads(v1: dict, v2: dict) -> dict:
    v1_ceiling = v1.get("ceiling", [])
    v2_ceiling = v2.get("ceiling", [])
    v2_shells = v2.get("visual_shells", [])

    v1_by_source: dict[str, float] = {}
    for piece in v1_ceiling:
        v1_by_source[piece["source"]] = v1_by_source.get(
            piece["source"], 0.0
        ) + _xz_area(piece["corners"])
    v2_by_source: dict[str, float] = {}
    for piece in v2_ceiling:
        v2_by_source[piece["source"]] = v2_by_source.get(
            piece["source"], 0.0
        ) + _xz_area(piece["corners"])
    v2_by_shell_tag: dict[str, float] = {}
    for shell in v2_shells:
        v2_by_shell_tag[shell["tag"]] = v2_by_shell_tag.get(
            shell["tag"], 0.0
        ) + _xz_area(shell["corners"])

    v1_locators = {piece["locator_id"]: piece["source"] for piece in v1_ceiling}
    v2_locators = {piece["locator_id"]: piece["source"] for piece in v2_ceiling}
    shell_locators = {shell["locator_id"]: shell["tag"] for shell in v2_shells}

    return {
        "uuid": v1.get("uuid"),
        "ceiling_count_v1": len(v1_ceiling),
        "ceiling_count_v2": len(v2_ceiling),
        "visual_shell_count_v2": len(v2_shells),
        "ceiling_area_v1_m2": round(sum(v1_by_source.values()), 4),
        "ceiling_area_v2_m2": round(sum(v2_by_source.values()), 4),
        "ceiling_area_delta_m2": round(
            sum(v2_by_source.values()) - sum(v1_by_source.values()), 4
        ),
        "shell_area_v2_m2": round(sum(v2_by_shell_tag.values()), 4),
        "v1_area_by_source_m2": _round_dict(v1_by_source),
        "v2_area_by_source_m2": _round_dict(v2_by_source),
        "v2_shell_area_by_tag_m2": _round_dict(v2_by_shell_tag),
        "removed_locators": sorted(set(v1_locators) - set(v2_locators)),
        "added_locators": sorted(set(v2_locators) - set(v1_locators)),
        "shell_locators": sorted(shell_locators),
    }


def _xz_area(corners: list[dict]) -> float:
    if len(corners) < 3:
        return 0.0
    poly = Polygon([(c["x"], c["z"]) for c in corners])
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty:
        return 0.0
    if hasattr(poly, "area"):
        return float(poly.area)
    return 0.0


def _round_dict(d: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in sorted(d.items())}


def _format_summary_line(uuid: str, diff: dict) -> str:
    ceiling_delta = diff["ceiling_area_delta_m2"]
    return (
        f"{uuid}: "
        f"ceiling {diff['ceiling_count_v1']}->{diff['ceiling_count_v2']} pieces, "
        f"area delta {ceiling_delta:+.2f} m^2, "
        f"shells {diff['visual_shell_count_v2']} ({diff['shell_area_v2_m2']:.2f} m^2), "
        f"removed {len(diff['removed_locators'])}, added {len(diff['added_locators'])}"
    )


def _render_index(summaries: list[tuple[str, dict]]) -> str:
    rows = []
    for uuid, diff in summaries:
        rows.append(
            f"<tr>"
            f"<td><a href='{uuid}/diff.json'>{uuid[:8]}</a></td>"
            f"<td>{diff['ceiling_count_v1']}</td>"
            f"<td>{diff['ceiling_count_v2']}</td>"
            f"<td>{diff['ceiling_area_delta_m2']:+.2f}</td>"
            f"<td>{diff['visual_shell_count_v2']}</td>"
            f"<td>{diff['shell_area_v2_m2']:.2f}</td>"
            f"<td>{len(diff['removed_locators'])}</td>"
            f"<td>{len(diff['added_locators'])}</td>"
            f"</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>envelope v1 vs v2</title>
<style>
body {{ font-family: system-ui, sans-serif; padding: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
</head><body>
<h1>Envelope v1 vs v2</h1>
<p>Compare each building's tier_payload.json (v1) to tier_payload_v2.json (v2). Open
each row's diff.json for full per-source breakdown and locator churn.</p>
<table>
<thead><tr>
<th>uuid</th><th>v1 ceiling</th><th>v2 ceiling</th><th>area Δ (m²)</th>
<th>v2 shells</th><th>shell area (m²)</th><th>removed</th><th>added</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
