"""IFC alignment spec and conformance checks for topology V2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from reconcile.models import Building
from reconcile.story_fix import cluster_stories, get_floor_y

from .models import TopologyGraph

IFC_MAPPING_TABLE = {
    "Room": "IfcSpace",
    "Story": "IfcBuildingStorey",
    "Surface.wall": "IfcWall",
    "Surface.door": "IfcDoor",
    "Surface.window": "IfcWindow",
    "Surface.opening": "IfcOpeningElement",
    "Surface.floor": "IfcSlab",
    "Edge.CONTAINS": "IfcRelContainedInSpatialStructure",
    "Boundary": "IfcRelSpaceBoundary",
}


@dataclass
class ConformanceIssue:
    code: str
    severity: str  # error|warning
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ConformanceReport:
    passed: bool
    issues: list[ConformanceIssue]
    checks: dict

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity,
                    "message": i.message,
                    "details": i.details,
                }
                for i in self.issues
            ],
            "checks": self.checks,
        }


def _node_index(graph: TopologyGraph) -> dict:
    return {n.id: n for n in graph.nodes}


def _incoming(graph: TopologyGraph) -> dict[str, list]:
    idx = defaultdict(list)
    for e in graph.edges:
        idx[e.to_id].append(e)
    return idx


def _outgoing(graph: TopologyGraph) -> dict[str, list]:
    idx = defaultdict(list)
    for e in graph.edges:
        idx[e.from_id].append(e)
    return idx


def _check_room_parent_story(
    graph: TopologyGraph,
) -> tuple[list[ConformanceIssue], dict]:
    nodes = _node_index(graph)
    incoming = _incoming(graph)
    issues: list[ConformanceIssue] = []

    room_nodes = [n for n in graph.nodes if n.type == "Room"]
    ok = 0
    for room in room_nodes:
        parents = [
            e
            for e in incoming.get(room.id, [])
            if e.type == "CONTAINS"
            and nodes.get(e.from_id)
            and nodes[e.from_id].type == "Story"
        ]
        if len(parents) != 1:
            issues.append(
                ConformanceIssue(
                    code="ROOM_PARENT_STORY",
                    severity="error",
                    message="Room must have exactly one parent Story",
                    details={"room": room.id, "parent_count": len(parents)},
                )
            )
        else:
            ok += 1

    return issues, {"rooms_total": len(room_nodes), "rooms_ok": ok}


def _check_story_connected_to_building(
    graph: TopologyGraph,
) -> tuple[list[ConformanceIssue], dict]:
    _node_index(graph)
    incoming = _incoming(graph)
    issues: list[ConformanceIssue] = []

    building_ids = {n.id for n in graph.nodes if n.type == "Building"}
    stories = [n for n in graph.nodes if n.type == "Story"]

    ok = 0
    for story in stories:
        parents = [
            e
            for e in incoming.get(story.id, [])
            if e.type == "CONTAINS" and e.from_id in building_ids
        ]
        if not parents:
            issues.append(
                ConformanceIssue(
                    code="STORY_NOT_CONNECTED",
                    severity="error",
                    message="Story must have a path to Building"
                    " (direct CONTAINS in v2)",
                    details={"story": story.id},
                )
            )
        else:
            ok += 1

    return issues, {"stories_total": len(stories), "stories_ok": ok}


def _check_boundary_completeness(
    graph: TopologyGraph,
) -> tuple[list[ConformanceIssue], dict]:
    nodes = _node_index(graph)
    outgoing = _outgoing(graph)
    issues: list[ConformanceIssue] = []

    boundaries = [n for n in graph.nodes if n.type == "Boundary"]
    ok = 0
    for b in boundaries:
        bounds = [e for e in outgoing.get(b.id, []) if e.type == "BOUNDS"]
        room_refs = [
            e for e in bounds if nodes.get(e.to_id) and nodes[e.to_id].type == "Room"
        ]
        elem_refs = [
            e for e in bounds if nodes.get(e.to_id) and nodes[e.to_id].type == "Surface"
        ]
        if len(room_refs) != 1 or len(elem_refs) != 1:
            issues.append(
                ConformanceIssue(
                    code="BOUNDARY_COMPLETENESS",
                    severity="error",
                    message="Boundary must reference exactly one"
                    " Room and one boundary element",
                    details={
                        "boundary": b.id,
                        "room_refs": len(room_refs),
                        "element_refs": len(elem_refs),
                    },
                )
            )
        else:
            ok += 1

    return issues, {"boundaries_total": len(boundaries), "boundaries_ok": ok}


def _check_orphans(graph: TopologyGraph) -> tuple[list[ConformanceIssue], dict]:
    incoming = _incoming(graph)
    outgoing = _outgoing(graph)
    issues: list[ConformanceIssue] = []

    allowed_orphan_types = {"Building", "Source"}
    orphan_nodes = []
    for n in graph.nodes:
        if (
            not incoming.get(n.id)
            and not outgoing.get(n.id)
            and n.type not in allowed_orphan_types
        ):
            orphan_nodes.append(n.id)

    if orphan_nodes:
        issues.append(
            ConformanceIssue(
                code="ORPHAN_NODES",
                severity="warning",
                message="Graph has non-allowed orphan nodes",
                details={"orphans": orphan_nodes[:50], "count": len(orphan_nodes)},
            )
        )

    return issues, {"orphans": len(orphan_nodes)}


def _check_story_ordering(
    graph: TopologyGraph, building: Building | None
) -> tuple[list[ConformanceIssue], dict]:
    _node_index(graph)
    issues: list[ConformanceIssue] = []

    stories = sorted(
        [n for n in graph.nodes if n.type == "Story" and n.story is not None],
        key=lambda n: n.story,
    )
    follows = {(e.from_id, e.to_id) for e in graph.edges if e.type == "FOLLOWS"}

    follows_ok = 0
    expected_pairs = 0
    for i in range(len(stories) - 1):
        expected_pairs += 1
        if (stories[i].id, stories[i + 1].id) not in follows:
            issues.append(
                ConformanceIssue(
                    code="STORY_ORDERING",
                    severity="error",
                    message="Missing FOLLOWS edge between consecutive stories",
                    details={"from": stories[i].id, "to": stories[i + 1].id},
                )
            )
        else:
            follows_ok += 1

    clustering_mismatches = 0
    checked_rooms = 0
    if building is not None:
        floor_ys = []
        room_refs = []
        for room in building.rooms:
            fy = get_floor_y(room)
            if fy is not None:
                floor_ys.append(float(fy))
                room_refs.append(room)

        if floor_ys:
            clustered = cluster_stories(floor_ys)
            for room, cluster_story in zip(room_refs, clustered, strict=False):
                checked_rooms += 1
                if room.story != cluster_story:
                    clustering_mismatches += 1

            if clustering_mismatches:
                issues.append(
                    ConformanceIssue(
                        code="STORY_CLUSTER_MISMATCH",
                        severity="warning",
                        message="Story ordering differs from"
                        " floor-Y clustering for some rooms",
                        details={
                            "checked_rooms": checked_rooms,
                            "mismatches": clustering_mismatches,
                        },
                    )
                )

    return issues, {
        "expected_follows": expected_pairs,
        "follows_ok": follows_ok,
        "cluster_checked_rooms": checked_rooms,
        "cluster_mismatches": clustering_mismatches,
    }


def run_ifc_conformance_checks(
    graph: TopologyGraph,
    building: Building | None = None,
) -> ConformanceReport:
    """Run mapping-level conformance checks aligned to IFC semantics."""
    checks = {}
    issues: list[ConformanceIssue] = []

    for name, fn in (
        ("room_parent_story", _check_room_parent_story),
        ("story_connected_to_building", _check_story_connected_to_building),
        ("boundary_completeness", _check_boundary_completeness),
        ("orphans", _check_orphans),
    ):
        part_issues, detail = fn(graph)
        checks[name] = detail
        issues.extend(part_issues)

    part_issues, detail = _check_story_ordering(graph, building)
    checks["story_ordering"] = detail
    issues.extend(part_issues)

    passed = not any(i.severity == "error" for i in issues)
    return ConformanceReport(passed=passed, issues=issues, checks=checks)
