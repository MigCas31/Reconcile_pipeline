"""Summarize a building's pipeline outputs into report.md + report.json."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "pipeline-outputs"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _summarize_datapack(datapack: dict | None) -> dict:
    if not datapack:
        return {}
    loc = datapack.get("resolved_location") or {}
    return {
        "address": datapack.get("address"),
        "lat": loc.get("lat") or loc.get("latitude"),
        "lon": loc.get("lon") or loc.get("longitude"),
        "parcel_present": bool(datapack.get("parcel_boundary")),
        "google_solar_present": bool(datapack.get("google_solar")),
        "danish_pointcloud_present": bool(datapack.get("danish_pointcloud")),
        "notes_count": len(datapack.get("notes") or []),
    }


def _summarize_tier(tier_payload: dict | None) -> dict:
    if not tier_payload:
        return {}
    cls = tier_payload.get("classification") or {}
    return {
        "tier": cls.get("tier"),
        "tier_label": cls.get("tier_label"),
        "roof_type": cls.get("roof_type"),
        "n_stories": cls.get("n_stories"),
        "n_rooms": cls.get("n_rooms"),
        "n_oblique": cls.get("n_oblique"),
        "n_flat": cls.get("n_flat"),
        "has_gable": cls.get("has_gable"),
        "has_half_height": cls.get("has_half_height"),
        "knee_walls": len(tier_payload.get("knee_walls") or []),
        "gaps": len(tier_payload.get("gaps") or []),
    }


def _summarize_roof(roof: dict | None) -> dict:
    if not roof:
        return {}
    out: dict = {}
    if isinstance(roof, dict):
        for key in ("ridges", "eaves", "segments", "planes", "surfaces"):
            value = roof.get(key)
            if isinstance(value, list):
                out[f"{key}_count"] = len(value)
    return out


def _list_files(building_dir: Path) -> list[dict]:
    files: list[dict] = []
    for path in sorted(building_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(building_dir)
        files.append(
            {
                "path": str(rel),
                "size_bytes": path.stat().st_size,
            }
        )
    return files


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def build_report(uuid: str, element_id: str | None, out_dir: Path) -> dict:
    building_dir = PIPELINE_ROOT / uuid
    if not building_dir.exists():
        raise FileNotFoundError(f"No pipeline-outputs entry for {uuid}")

    datapack = _load_json(building_dir / "datapack.json")
    tier_payload = _load_json(building_dir / "tier_payload.json")
    roofdiffusion = _load_json(building_dir / "roofdiffusion.json")
    topology_qa = _load_json(REPO_ROOT / "topology-v2.qa.json") or _load_json(
        building_dir / "topology-v2.qa.json"
    )
    element = _load_json(out_dir / "element.json")
    realism = _load_json(out_dir / "metrics.json")
    audit_data = _load_json(out_dir / "audit.json")
    val3dity = _load_json(out_dir / "val3dity.json")

    report = {
        "uuid": uuid,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "element_id": element_id,
        "element": element,
        "metadata": _summarize_datapack(datapack),
        "tier": _summarize_tier(tier_payload),
        "roof": _summarize_roof(roofdiffusion),
        "topology_v2": topology_qa,
        "realism": realism,
        "audit": audit_data,
        "val3dity": val3dity,
        "files": _list_files(building_dir),
    }
    return report


_LEVEL_BADGE = {"fail": "❌", "warn": "⚠️", "pass": "✅"}


def _render_realism_section(realism: dict) -> list[str]:
    lines: list[str] = ["## Realism (LoD2 reconstruction sanity)", ""]
    flags = realism.get("flags") or []
    if flags:
        lines.append(
            f"**{len(flags)} failed check(s)**: " + ", ".join(f"`{f}`" for f in flags),
        )
        lines.append("")
    extents = realism.get("extents") or {}
    if extents:
        lines.append(
            f"- Extents (x, y, z): {extents.get('x', 0):.2f} x "
            f"{extents.get('y', 0):.2f} x {extents.get('z', 0):.2f} m"
        )
    if realism.get("vertex_count") is not None:
        lines.append(
            f"- Mesh: {realism['vertex_count']} vertices, "
            f"{realism['face_count']} faces, "
            f"{realism.get('component_count', '?')} component(s)"
        )
    if realism.get("volume_m3") is not None:
        lines.append(f"- Volume: {realism['volume_m3']:.0f} m^3")
    if realism.get("convex_hull_volume_m3") is not None:
        lines.append(
            f"- Convex hull volume: {realism['convex_hull_volume_m3']:.0f} m^3"
        )
    if realism.get("convexity_ratio") is not None:
        lines.append(f"- Convexity ratio: {realism['convexity_ratio']:.2f}")
    nd = realism.get("normal_distribution") or {}
    if nd:
        lines.append(
            f"- Face area split: walls {nd.get('wall_pct', 0) * 100:.0f}% / "
            f"roof flat {nd.get('roof_flat_pct', 0) * 100:.0f}% / "
            f"roof oblique {nd.get('roof_oblique_pct', 0) * 100:.0f}% / "
            f"ground {nd.get('ground_pct', 0) * 100:.0f}%"
        )
    lines.append("")
    detail = realism.get("flags_detail") or []
    if detail:
        lines.append("| Check | Result | Reason |")
        lines.append("| --- | :---: | --- |")
        for entry in detail:
            badge = _LEVEL_BADGE.get(entry.get("level"), "?")
            lines.append(
                f"| `{entry.get('name')}` | {badge} | {entry.get('reason', '')} |"
            )
        lines.append("")
    holes = realism.get("holes") or []
    if holes:
        lines.append(f"### Watertight holes ({len(holes)})")
        lines.append("")
        lines.append(
            "Each row is one boundary loop on the LoD2 mesh -- i.e. a place "
            "where the shell is open."
        )
        lines.append("")
        lines.append("| # | Location | Perimeter (m) | Centroid (x, y, z) | Vertices |")
        lines.append("| ---: | --- | ---: | --- | ---: |")
        for i, hole in enumerate(holes[:20]):
            cx, cy, cz = hole.get("centroid", [0, 0, 0])
            lines.append(
                f"| {i + 1} | `{hole.get('location', '?')}` | "
                f"{hole.get('perimeter_m', 0):.2f} | "
                f"({cx:.2f}, {cy:.2f}, {cz:.2f}) | "
                f"{hole.get('vertex_count', 0)} |"
            )
        if len(holes) > 20:
            lines.append(f"| ... | _{len(holes) - 20} more_ | | | |")
        lines.append("")
    return lines


def _render_audit_section(audit: dict) -> list[str]:
    lines: list[str] = ["## Defects (tier_payload audit)", ""]
    flags = audit.get("flags") or []
    if flags:
        lines.append(
            f"**{len(flags)} class(es) of defect**: "
            + ", ".join(f"`{f}`" for f in flags)
        )
    else:
        lines.append("No defects flagged.")
    lines.append("")

    coverage = audit.get("ceiling_coverage_gaps") or []
    if coverage:
        lines.append(f"### Rooms with low ceiling coverage ({len(coverage)})")
        lines.append("")
        lines.append("Paste the locator into the viewer search box to inspect.")
        lines.append("")
        lines.append("| Room | Story | Floor (m^2) | Covered (m^2) | Coverage |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for entry in coverage[:15]:
            lines.append(
                f"| `{entry.get('room_locator_id')}` | "
                f"{entry.get('story', '?')} | "
                f"{entry.get('floor_area_m2', 0):.1f} | "
                f"{entry.get('covered_area_m2', 0):.1f} | "
                f"{entry.get('coverage_ratio', 0) * 100:.0f}% |"
            )
        if len(coverage) > 15:
            lines.append(f"| _{len(coverage) - 15} more_ | | | | |")
        lines.append("")

    orient = audit.get("ceiling_orientation_wrong") or []
    if orient:
        lines.append(f"### Ceilings with wrong orientation ({len(orient)})")
        lines.append("")
        lines.append("Normal Y component should be > 0.10 for upward-facing surfaces.")
        lines.append("")
        lines.append("| Locator | Normal.y | Source |")
        lines.append("| --- | ---: | --- |")
        for entry in orient[:15]:
            lines.append(
                f"| `{entry.get('locator_id')}` | "
                f"{entry.get('normal_y', 0):.2f} | "
                f"{entry.get('source', '')} |"
            )
        lines.append("")

    outside = audit.get("out_of_envelope") or []
    if outside:
        lines.append(f"### Geometry outside the building envelope ({len(outside)})")
        lines.append("")
        lines.append(
            "Envelope = convex hull of ground-story room floors, buffered "
            "by 0.5 m. Listed entries extend further out than that."
        )
        lines.append("")
        lines.append("| Kind | Locator | Outside (m^2) | Outside ratio |")
        lines.append("| --- | --- | ---: | ---: |")
        for entry in outside[:15]:
            lines.append(
                f"| {entry.get('kind')} | `{entry.get('locator_id')}` | "
                f"{entry.get('outside_area_xz_m2', 0):.2f} | "
                f"{entry.get('outside_ratio', 0) * 100:.0f}% |"
            )
        if len(outside) > 15:
            lines.append(f"| _{len(outside) - 15} more_ | | | |")
        lines.append("")

    drops = audit.get("silent_drops") or []
    if drops:
        lines.append(f"### Silently dropped polygons ({len(drops)})")
        lines.append("")
        lines.append(
            "These exist in `tier_payload.json` but the renderer skips them -- "
            "you will never see them in the viewer."
        )
        lines.append("")
        lines.append("| Kind | Locator | Reason |")
        lines.append("| --- | --- | --- |")
        for entry in drops[:15]:
            lines.append(
                f"| {entry.get('kind')} | `{entry.get('locator_id')}` | "
                f"{entry.get('reason', '')} |"
            )
        if len(drops) > 15:
            lines.append(f"| _{len(drops) - 15} more_ | | |")
        lines.append("")

    knee = audit.get("knee_walls_misplaced") or []
    if knee:
        lines.append(f"### Knee walls not at top story ({len(knee)})")
        lines.append("")
        lines.append("| Locator | Y range | Top story Y_max | Δ (m) |")
        lines.append("| --- | --- | ---: | ---: |")
        for entry in knee[:15]:
            yr = entry.get("y_range") or [0, 0]
            lines.append(
                f"| `{entry.get('locator_id')}` | "
                f"[{yr[0]:.2f}, {yr[1]:.2f}] | "
                f"{entry.get('top_story_y_max', 0):.2f} | "
                f"{entry.get('delta_m', 0):.2f} |"
            )
        lines.append("")

    story_cov = audit.get("story_coverage") or {}
    if story_cov:
        lines.append("### Story coverage census")
        lines.append("")
        lines.append("| Story | Rooms | Ceilings | Y range |")
        lines.append("| ---: | ---: | ---: | --- |")
        for story, info in sorted(story_cov.items(), key=lambda kv: int(kv[0])):
            y0 = info.get("y_min")
            y1 = info.get("y_max")
            yrange = (
                f"[{y0:.2f}, {y1:.2f}]" if y0 is not None and y1 is not None else "?"
            )
            lines.append(
                f"| {story} | {info.get('room_count', 0)} | "
                f"{info.get('ceiling_count', 0)} | {yrange} |"
            )
        lines.append("")

    census = audit.get("gap_census") or []
    if census:
        lines.append("### Gap census")
        lines.append("")
        lines.append("| Kind | Scope | Count |")
        lines.append("| --- | --- | ---: |")
        for entry in census:
            lines.append(
                f"| {entry.get('kind')} | {entry.get('scope')} | "
                f"{entry.get('count', 0)} |"
            )
        lines.append("")

    return lines


def render_markdown(report: dict, out_dir: Path) -> str:
    uuid = report["uuid"]
    meta = report.get("metadata") or {}
    tier = report.get("tier") or {}
    realism = report.get("realism") or {}
    audit_data = report.get("audit") or {}
    val3dity = report.get("val3dity") or {}
    element = report.get("element")
    shots_dir = out_dir / "shots"

    lines: list[str] = []
    address = meta.get("address") or "(no address)"
    lines.append(f"# {address}")
    lines.append("")
    lines.append(f"**UUID**: `{uuid}`")
    lines.append(f"**Generated**: {report['generated_at']}")
    if meta.get("lat") and meta.get("lon"):
        lines.append(f"**Location**: {meta['lat']:.6f}, {meta['lon']:.6f}")
    lines.append("")

    if tier:
        lines.append("## Tier classification")
        lines.append("")
        for key in (
            "tier",
            "tier_label",
            "roof_type",
            "n_stories",
            "n_rooms",
            "n_oblique",
            "n_flat",
            "has_gable",
            "has_half_height",
            "knee_walls",
            "gaps",
        ):
            if tier.get(key) is not None:
                lines.append(f"- **{key}**: {tier[key]}")
        lines.append("")

    if element:
        lines.append("## Element trace")
        lines.append("")
        lines.append(f"`{report.get('element_id')}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(element, indent=2, default=str)[:2000])
        lines.append("```")
        lines.append("")

    if realism:
        lines.extend(_render_realism_section(realism))

    if audit_data:
        lines.extend(_render_audit_section(audit_data))

    if val3dity:
        lines.append("## val3dity validation")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(val3dity, indent=2, default=str)[:1500])
        lines.append("```")
        lines.append("")
    elif (out_dir / ".val3dity-missing").exists():
        lines.append("## val3dity validation")
        lines.append("")
        lines.append("Not installed. To enable:")
        lines.append("")
        lines.append("    brew tap tudelft3d/software && brew install val3dity")
        lines.append("")

    if shots_dir.exists():
        shots = sorted(p.name for p in shots_dir.glob("*.png"))
        if shots:
            lines.append("## Screenshots")
            lines.append("")
            for name in shots:
                lines.append(f"![{name}](shots/{name})")
                lines.append("")

    files = report.get("files") or []
    if files:
        lines.append("## Pipeline outputs")
        lines.append("")
        lines.append("| File | Size |")
        lines.append("| --- | ---: |")
        for entry in files:
            lines.append(f"| `{entry['path']}` | {_human_size(entry['size_bytes'])} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--element-id", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(args.uuid, args.element_id, out_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "report.md").write_text(render_markdown(report, out_dir))
    print(out_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
