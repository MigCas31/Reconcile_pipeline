from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from audit_full_model_payloads import _area_metrics, _build_payload

from reconcile.viewer_server import FULL_BUILDING_PART_ID


def _classify_failure_mode(
    *,
    total_rooms: int,
    exposed_room_count: int,
    shell_ratio: float | None,
    shell_overlap_ratio: float | None,
    semantic_ratio: float | None,
    semantic_overlap_ratio: float | None,
    role_area: dict[str, float],
    mixed_room_count: int,
) -> str:
    shell_ratio = float(shell_ratio or 0.0)
    shell_overlap_ratio = float(shell_overlap_ratio or 0.0)
    semantic_ratio = float(semantic_ratio or 0.0)
    semantic_overlap_ratio = float(semantic_overlap_ratio or 0.0)
    exposed_fraction = (exposed_room_count / total_rooms) if total_rooms > 0 else 0.0
    flat_ceiling_area = float(role_area.get("flat_ceiling", 0.0) or 0.0)
    semantic_sensitive_area = sum(
        float(role_area.get(role, 0.0) or 0.0)
        for role in (
            "sloped_ceiling",
            "flat_transition_cap",
            "flat_transition_cap_inferred",
            "attic_floor",
            "attic_floor_inferred",
        )
    )

    if shell_ratio < 0.85 or shell_overlap_ratio < 0.85:
        return "shell_ceiling_gap"
    if exposed_fraction < 0.5 and semantic_ratio < 0.6:
        return "non_exposed_room_scope_gap"
    if (
        flat_ceiling_area > semantic_sensitive_area * 1.5
        and semantic_overlap_ratio < 0.6
    ):
        return "ordinary_flat_ceiling_not_semanticized"
    if mixed_room_count > 0 and semantic_overlap_ratio < 0.6:
        return "mixed_semantic_undercoverage"
    return "semantic_scope_gap_other"


def main() -> None:
    source_path = Path(".context/full_model_full_building_audit.json")
    output_json = Path(".context/ceiling_parity_deficit_audit.json")
    output_md = Path(".context/ceiling_parity_deficit_audit.md")
    report = json.loads(source_path.read_text())
    ok_rows = [
        row for row in (report.get("results") or []) if row.get("status") == "ok"
    ]
    ranked_rows = sorted(
        ok_rows,
        key=lambda row: (
            -float(
                (row.get("area_metrics") or {}).get(
                    "semantic_ceiling_heuristic_only_projected_area_m2", 0.0
                )
                or 0.0
            ),
            float(
                (row.get("area_metrics") or {}).get(
                    "semantic_ceiling_overlap_ratio_vs_heuristic"
                )
                or 0.0
            ),
            row["uuid"],
        ),
    )
    targets = [
        row
        for row in ranked_rows
        if float(
            (row.get("area_metrics") or {}).get("heuristic_ceiling_area_m2", 0.0) or 0.0
        )
        > 1.0
    ][:25]

    bucket_counts: Counter[str] = Counter()
    results: list[dict] = []

    for row in targets:
        uuid = row["uuid"]
        started = perf_counter()
        summary, parts, roof = _build_payload(uuid)
        payload = parts[FULL_BUILDING_PART_ID]
        top_boundary_graph = (roof.get("ceiling") or {}).get("top_boundary_graph") or {}
        room_partitions = (roof.get("ceiling") or {}).get("room_partitions") or []
        role_area: dict[str, float] = defaultdict(float)
        for node in top_boundary_graph.get("nodes") or []:
            if not isinstance(node, dict) or node.get("type") != "TopBoundaryAtom":
                continue
            role_area[str(node.get("role") or "")] += float(
                node.get("area_m2", 0.0) or 0.0
            )
        total_rooms = len(
            {
                int(surface.get("room_index"))
                for surface in (payload.get("renderable_surfaces") or [])
                if isinstance(surface, dict)
                and str(surface.get("category") or "") == "base_room_ceiling"
                and isinstance(surface.get("room_index"), int)
            }
        )
        exposed_room_count = len((roof.get("ceiling") or {}).get("exposed_rooms") or [])
        room_summaries = top_boundary_graph.get("room_summaries") or {}
        resolved_room_count = sum(
            1
            for value in room_summaries.values()
            if isinstance(value, dict) and value.get("has_resolved_roof_relation")
        )
        mixed_room_count = sum(
            1
            for room in room_partitions
            if isinstance(room, dict) and room.get("mixed")
        )
        area_metrics = _area_metrics(summary, payload, roof)
        failure_mode = _classify_failure_mode(
            total_rooms=total_rooms,
            exposed_room_count=exposed_room_count,
            shell_ratio=area_metrics.get("shell_ceiling_area_ratio"),
            shell_overlap_ratio=area_metrics.get(
                "shell_ceiling_overlap_ratio_vs_heuristic"
            ),
            semantic_ratio=area_metrics.get("ceiling_area_ratio"),
            semantic_overlap_ratio=area_metrics.get(
                "semantic_ceiling_overlap_ratio_vs_heuristic"
            ),
            role_area=role_area,
            mixed_room_count=mixed_room_count,
        )
        bucket_counts[failure_mode] += 1
        results.append(
            {
                "uuid": uuid,
                "timings_s": {"total": round(perf_counter() - started, 6)},
                "failure_mode": failure_mode,
                "area_metrics": area_metrics,
                "room_counts": {
                    "total_rooms": total_rooms,
                    "exposed_rooms": exposed_room_count,
                    "resolved_top_boundary_rooms": resolved_room_count,
                    "mixed_ceiling_partition_rooms": mixed_room_count,
                },
                "top_boundary_role_area_m2": {
                    key: round(value, 6)
                    for key, value in sorted(role_area.items())
                    if value > 0.0
                },
                "payload_metadata": payload.get("metadata") or {},
                "summary_metadata": summary.get("metadata") or {},
            }
        )

    aggregate = {
        "building_count": len(results),
        "failure_mode_counts": dict(sorted(bucket_counts.items())),
        "avg_semantic_ceiling_overlap_ratio_vs_heuristic": round(
            sum(
                float(
                    row["area_metrics"].get(
                        "semantic_ceiling_overlap_ratio_vs_heuristic"
                    )
                    or 0.0
                )
                for row in results
            )
            / len(results),
            6,
        )
        if results
        else None,
        "avg_shell_ceiling_overlap_ratio_vs_heuristic": round(
            sum(
                float(
                    row["area_metrics"].get("shell_ceiling_overlap_ratio_vs_heuristic")
                    or 0.0
                )
                for row in results
            )
            / len(results),
            6,
        )
        if results
        else None,
    }
    out = {
        "generated_from": str(source_path),
        "aggregate": aggregate,
        "results": results,
    }
    output_json.write_text(json.dumps(out, indent=2))

    lines = [
        "# Ceiling Parity Deficit Audit",
        "",
        f"- source_audit: `{source_path}`",
        f"- building_count: `{aggregate['building_count']}`",
        f"- avg_semantic_ceiling_overlap_ratio_vs_heuristic: "
        f"`{aggregate['avg_semantic_ceiling_overlap_ratio_vs_heuristic']}`",
        f"- avg_shell_ceiling_overlap_ratio_vs_heuristic: "
        f"`{aggregate['avg_shell_ceiling_overlap_ratio_vs_heuristic']}`",
        "",
        "## Failure Modes",
        "",
    ]
    for name, count in sorted(aggregate["failure_mode_counts"].items()):
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(["", "## Top Buildings", ""])
    for row in results:
        am = row["area_metrics"]
        sem = am.get("semantic_ceiling_overlap_ratio_vs_heuristic")
        shl = am.get("shell_ceiling_overlap_ratio_vs_heuristic")
        heu = am.get("semantic_ceiling_heuristic_only_projected_area_m2")
        lines.append(
            f"- `{row['uuid']}` mode=`{row['failure_mode']}` "
            f"semantic_overlap=`{sem}` shell_overlap=`{shl}` "
            f"heuristic_only=`{heu}`"
        )
        rc = row["room_counts"]
        lines.append(
            f"  rooms total/exposed/resolved/mixed = "
            f"`{rc['total_rooms']}/{rc['exposed_rooms']}/"
            f"{rc['resolved_top_boundary_rooms']}/"
            f"{rc['mixed_ceiling_partition_rooms']}`"
        )
        lines.append(f"  role_area = `{row['top_boundary_role_area_m2']}`")
    output_md.write_text("\n".join(lines))

    print(json.dumps(aggregate, indent=2))
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
