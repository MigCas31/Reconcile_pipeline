from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import prototype_raw_ceiling_plane_scorer as legacy

from .models import RelationGraph, RunOutput


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: _json_dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def build_run_output(
    per_target_rows: list[dict[str, Any]],
    split_piece_rows: list[dict[str, Any]],
    relation_graph_by_uuid: dict[str, RelationGraph],
) -> RunOutput:
    per_story_rows = legacy._summarize_story_rows(per_target_rows)
    summary = legacy._summary_payload(per_target_rows, per_story_rows)

    final_count = sum(1 for row in split_piece_rows if bool(row.get("final_layer")))
    summary["counts"]["n_plane_extent_split_pieces"] = len(split_piece_rows)
    summary["counts"]["n_plane_extent_split_pieces_final"] = final_count
    summary["counts"]["n_plane_extent_split_pieces_candidate"] = (
        len(split_piece_rows) - final_count
    )
    summary["final_layer_reason_counts"] = dict(
        Counter(str(row.get("final_layer_reason") or "") for row in split_piece_rows)
    )

    relation_counts = Counter()
    for graph in relation_graph_by_uuid.values():
        relation_counts.update(relation.relation_kind for relation in graph.relations)
    summary["relation_kind_counts"] = dict(relation_counts)

    relation_sidecar = {
        "available": bool(relation_graph_by_uuid),
        "buildings": {
            uuid: {
                "contexts": [
                    asdict(context) for context in graph.contexts_by_target.values()
                ],
                "relations": [asdict(relation) for relation in graph.relations],
            }
            for uuid, graph in relation_graph_by_uuid.items()
        },
    }

    return RunOutput(
        per_target_rows=per_target_rows,
        per_story_rows=per_story_rows,
        split_piece_rows=split_piece_rows,
        summary=summary,
        relation_sidecar=relation_sidecar,
    )


def write_outputs(out_dir: Path, output: RunOutput) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "per_target.csv", output.per_target_rows)
    write_csv(out_dir / "per_story.csv", output.per_story_rows)
    write_csv(out_dir / "plane_extent_splits.csv", output.split_piece_rows)

    (out_dir / "per_target.json").write_text(
        json.dumps({"rows": output.per_target_rows}, indent=2),
        encoding="utf-8",
    )
    split_payload = {
        "available": bool(output.split_piece_rows),
        "buildings": defaultdict(list),
    }
    for row in output.split_piece_rows:
        split_payload["buildings"][str(row["uuid"])].append(row)
    split_payload["buildings"] = dict(split_payload["buildings"])
    (out_dir / "plane_extent_splits.json").write_text(
        json.dumps(split_payload, indent=2),
        encoding="utf-8",
    )

    (out_dir / "summary.json").write_text(
        json.dumps(output.summary, indent=2), encoding="utf-8"
    )
    (out_dir / "plane_relations.json").write_text(
        json.dumps(output.relation_sidecar, indent=2),
        encoding="utf-8",
    )
