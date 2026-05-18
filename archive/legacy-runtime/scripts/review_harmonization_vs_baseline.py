"""Compare freshly-regenerated tier_payloads against the committed (HEAD)
baseline to surface what the height_align harmonization actually changed
across the full corpus.

For each building:
  - classification deltas (tier, roof_type, n_oblique, n_flat, n_dormers,
    n_ceiling_pieces, n_knee_walls, n_thermal_caps)
  - total ceiling XZ area drift
  - count of physical inter-story overlaps (rooms whose XZ bboxes intersect
    AND whose y ranges conflict — true geometric overlaps, not just
    same-vertical-band collisions across different rooms)

Aggregate corpus-level stats. Highlight any building whose classification
shifted — those are the cases worth eyeballing in the viewer before merge.

Usage:
    python scripts/review_harmonization_vs_baseline.py [--limit N]
    --report path defaults to .context/corpus_harmonization_review.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


def _find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pipeline-outputs").is_dir() and (
            parent / "reconcile_tiers"
        ).is_dir():
            return parent
    return Path.cwd()


ROOT = _find_workspace_root()
DEFAULT_OUTPUTS = ROOT / "pipeline-outputs"
DEFAULT_REPORT = ROOT / ".context" / "corpus_harmonization_review.md"


def _xz_area(corners: list[dict[str, float]]) -> float:
    if len(corners) < 3:
        return 0.0
    pts = [(float(c["x"]), float(c["z"])) for c in corners]
    a = 0.0
    for i, (x0, z0) in enumerate(pts):
        x1, z1 = pts[(i + 1) % len(pts)]
        a += x0 * z1 - x1 * z0
    return abs(a) * 0.5


def _classification(payload: dict) -> dict:
    cls = payload.get("classification") or {}
    ceiling = payload.get("ceiling") or []
    dormers = payload.get("dormer_faces") or []
    knee = payload.get("knee_walls") or []
    return {
        "tier": cls.get("tier"),
        "roof_type": cls.get("roof_type"),
        "n_oblique": cls.get("n_oblique"),
        "n_flat": cls.get("n_flat"),
        "n_dormers": sum(1 for f in dormers if f.get("kind") == "dormer_header"),
        "n_ceiling_pieces": len(ceiling),
        "n_knee_walls": len(knee),
        "n_thermal_caps": sum(1 for x in ceiling if x.get("source") == "thermal_cap"),
        "total_ceiling_xz_area": round(sum(_xz_area(p["corners"]) for p in ceiling), 4),
    }


def _xz_bbox(
    corners: list[dict[str, float]],
) -> tuple[float, float, float, float] | None:
    if not corners:
        return None
    xs = [c["x"] for c in corners]
    zs = [c["z"] for c in corners]
    return (min(xs), min(zs), max(xs), max(zs))


def _xz_overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _physical_inter_story_overlaps(payload: dict) -> int:
    rows = []
    for r in payload.get("rooms") or []:
        floor_ys: list[float] = []
        ceil_ys: list[float] = []
        all_corners: list[dict[str, float]] = []
        for w in r.get("walls") or []:
            ys = [c["y"] for c in w["corners"]]
            if max(ys) - min(ys) < 0.10:
                continue
            floor_ys.append(min(ys))
            ceil_ys.append(max(ys))
            all_corners.extend(w["corners"])
        if not floor_ys:
            continue
        bbox = _xz_bbox(all_corners)
        if bbox is None:
            continue
        rows.append((int(r["story"]), min(floor_ys), max(ceil_ys), bbox))
    n = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][0] == rows[j][0]:
                continue
            if not _xz_overlaps(rows[i][3], rows[j][3]):
                continue
            lo, up = (
                (rows[i], rows[j]) if rows[i][0] < rows[j][0] else (rows[j], rows[i])
            )
            gap = up[1] - lo[2]
            if gap < -0.005:
                n += 1
    return n


def _load_baseline(uuid: str) -> dict | None:
    path = f"pipeline-outputs/{uuid}/tier_payload.json"
    try:
        result = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=str(ROOT),
            capture_output=True,
            check=True,
            text=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


@dataclass
class _Diff:
    uuid: str
    address: str | None
    classification_changes: dict[str, tuple]
    area_delta: float
    overlap_before: int
    overlap_after: int


def review(outputs: Path, limit: int) -> list[_Diff]:
    payload_paths = sorted(outputs.glob("*/tier_payload.json"))
    if limit > 0:
        payload_paths = payload_paths[:limit]

    diffs: list[_Diff] = []
    for path in payload_paths:
        uuid = path.parent.name
        try:
            current = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        baseline = _load_baseline(uuid)
        if baseline is None:
            continue

        cur_cls = _classification(current)
        base_cls = _classification(baseline)
        changes = {
            k: (base_cls[k], cur_cls[k])
            for k in cur_cls
            if k != "total_ceiling_xz_area" and cur_cls[k] != base_cls[k]
        }
        diffs.append(
            _Diff(
                uuid=uuid,
                address=current.get("address"),
                classification_changes=changes,
                area_delta=cur_cls["total_ceiling_xz_area"]
                - base_cls["total_ceiling_xz_area"],
                overlap_before=_physical_inter_story_overlaps(baseline),
                overlap_after=_physical_inter_story_overlaps(current),
            )
        )
    return diffs


def write_report(diffs: list[_Diff], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(diffs)
    n_classification_changes = sum(1 for d in diffs if d.classification_changes)
    n_area_changed = sum(1 for d in diffs if abs(d.area_delta) > 0.05)
    overlap_improved = sum(1 for d in diffs if d.overlap_after < d.overlap_before)
    overlap_worsened = sum(1 for d in diffs if d.overlap_after > d.overlap_before)
    overlap_same = n_total - overlap_improved - overlap_worsened
    total_overlap_before = sum(d.overlap_before for d in diffs)
    total_overlap_after = sum(d.overlap_after for d in diffs)

    diffs_by_area = sorted(diffs, key=lambda d: abs(d.area_delta), reverse=True)

    field_changes = Counter()
    for d in diffs:
        for field in d.classification_changes:
            field_changes[field] += 1

    lines: list[str] = []
    lines.append(
        "# Corpus harmonization review — committed baseline vs. current code\n"
    )
    lines.append(
        "Compares each building's freshly-regenerated `tier_payload.json` "
        "against the committed (HEAD) version. Surfaces classification "
        "deltas, ceiling-area drift, and inter-story overlap counts so we "
        "can spot regressions before merging the height_align changes.\n"
    )

    lines.append("## Corpus rollup\n")
    lines.append(f"- Buildings analysed: **{n_total}**")
    lines.append(
        f"- Buildings with classification flip "
        f"(tier / roof_type / n_oblique / n_flat / n_dormers / "
        f"n_ceiling_pieces / n_knee_walls / n_thermal_caps): "
        f"**{n_classification_changes}** "
        f"({n_classification_changes / n_total * 100:.1f}% of corpus)"
    )
    lines.append(f"- Buildings with |ceiling area Δ| > 5 cm²: **{n_area_changed}**")
    lines.append(
        "- Inter-story overlap counts (rooms whose XZ overlaps with another "
        "on a different story AND whose y ranges conflict by > 5 mm):"
    )
    lines.append(f"  - total before: **{total_overlap_before}**")
    lines.append(
        f"  - total after:  **{total_overlap_after}** "
        f"({(total_overlap_after - total_overlap_before):+d})"
    )
    lines.append(f"  - buildings improved: {overlap_improved}")
    lines.append(f"  - buildings worsened: {overlap_worsened}")
    lines.append(f"  - buildings unchanged: {overlap_same}\n")

    if field_changes:
        lines.append("## Classification field-change frequency\n")
        lines.append("| field | buildings changed |")
        lines.append("|---|---|")
        for field, count in field_changes.most_common():
            lines.append(f"| {field} | {count} |")
        lines.append("")

    lines.append("## Buildings with classification flip (review-required)\n")
    flipped = [d for d in diffs if d.classification_changes]
    if not flipped:
        lines.append("(none — every building keeps its committed classification)\n")
    else:
        lines.append(
            "| building | address | changes | ceiling Δ m² | overlaps before→after |"
        )
        lines.append("|---|---|---|---|---|")
        for d in flipped:
            change_strs = [
                f"{k}: {a}→{b}" for k, (a, b) in d.classification_changes.items()
            ]
            addr = (d.address or "")[:40]
            lines.append(
                f"| `{d.uuid[:8]}` | {addr} | {', '.join(change_strs)} | "
                f"{d.area_delta:+.2f} | {d.overlap_before}→{d.overlap_after} |"
            )
        lines.append("")

    lines.append("## Top 20 buildings by absolute ceiling-area drift\n")
    lines.append(
        "| building | address | ceiling Δ m² | classification | overlaps before→after |"
    )
    lines.append("|---|---|---|---|---|")
    for d in diffs_by_area[:20]:
        addr = (d.address or "")[:40]
        cls = (
            ", ".join(f"{k}:{a}→{b}" for k, (a, b) in d.classification_changes.items())
            or "unchanged"
        )
        lines.append(
            f"| `{d.uuid[:8]}` | {addr} | {d.area_delta:+.3f} | {cls} | "
            f"{d.overlap_before}→{d.overlap_after} |"
        )
    lines.append("")

    lines.append("## Buildings whose inter-story overlaps grew (regressions)\n")
    worsened = [d for d in diffs if d.overlap_after > d.overlap_before]
    if not worsened:
        lines.append("(none — no building gained inter-story overlaps)\n")
    else:
        lines.append("| building | address | overlaps before→after | ceiling Δ |")
        lines.append("|---|---|---|---|")
        for d in worsened:
            addr = (d.address or "")[:40]
            lines.append(
                f"| `{d.uuid[:8]}` | {addr} | "
                f"{d.overlap_before}→{d.overlap_after} | {d.area_delta:+.2f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    diffs = review(args.inputs, args.limit)
    write_report(diffs, args.report)
    print(f"reviewed {len(diffs)} buildings; report at {args.report}")


if __name__ == "__main__":
    main()
