"""CLI for parallel topology V2 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from reconcile.loader import load_merged

from .graph_builder import build_topology_graph
from .ifc_mapping import run_ifc_conformance_checks
from .output import write_topology_outputs


def build_and_write_v2(
    merged_path: Path,
    scan_dir: Path,
    output_v2_path: Path,
    qa_output_path: Path,
    uuid: str = "",
) -> dict:
    graph = build_topology_graph(
        merged_path=merged_path,
        scan_dir=scan_dir,
        uuid=uuid,
    )

    building = load_merged(merged_path)
    conformance = run_ifc_conformance_checks(graph, building)
    payload, qa = write_topology_outputs(
        graph=graph,
        output_v2_path=output_v2_path,
        qa_output_path=qa_output_path,
        conformance=conformance,
    )

    return {
        "uuid": uuid,
        "version": payload["version"],
        "nodes": len(payload["nodes"]),
        "edges": len(payload["edges"]),
        "schema_valid": len(qa["schema"]["errors"]) == 0,
        "ifc_conformance_passed": qa["ifc_mapping"]["conformance"]["passed"],
        "output_v2": str(output_v2_path),
        "output_qa": str(qa_output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build topology-v2 graph artifact")
    parser.add_argument("--building", required=True, help="Path to merged.json")
    parser.add_argument(
        "--scan-cache", required=True, help="Path to scan-cache directory"
    )
    parser.add_argument(
        "--output-v2", required=True, help="Path to topology-v2.json output"
    )
    parser.add_argument(
        "--qa-output",
        default="",
        help="Path to topology-v2.qa.json (defaults next to --output-v2)",
    )
    parser.add_argument("--uuid", default="", help="Building UUID for reporting")
    args = parser.parse_args()

    output_v2 = Path(args.output_v2)
    qa_output = (
        Path(args.qa_output) if args.qa_output else output_v2.with_suffix(".qa.json")
    )

    result = build_and_write_v2(
        merged_path=Path(args.building),
        scan_dir=Path(args.scan_cache),
        output_v2_path=output_v2,
        qa_output_path=qa_output,
        uuid=args.uuid,
    )

    print(f"[TOPOLOGY-V2] {result['uuid']}")
    print(f"  Version: {result['version']}")
    print(f"  Nodes: {result['nodes']}, Edges: {result['edges']}")
    print(f"  Schema valid: {result['schema_valid']}")
    print(f"  IFC mapping checks passed: {result['ifc_conformance_passed']}")
    print(f"  Output: {result['output_v2']}")
    print(f"  QA: {result['output_qa']}")


if __name__ == "__main__":
    main()
