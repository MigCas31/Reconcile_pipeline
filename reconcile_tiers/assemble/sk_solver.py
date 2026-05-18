"""Python wrapper around the CGAL weighted-straight-skeleton CLI in
``tools/sk_solver/build/sk_solver``. Subprocess-call per polygon.

The CLI takes JSON {polygon, weights} and returns {ok, faces} where each face
is a list of {x, z, time} vertices and an edge_index that maps back to the
input polygon edge. See tools/sk_solver/sk_solver.cpp for full schema.

Polygon orientation MUST be counter-clockwise; the wrapper rotates if needed.
Weights must be strictly positive (CGAL rejects 0). The caller picks weight
values that encode the desired roof shape: high weight = fast wavefront =
*small* face for that edge (the flat-ceiling above a FLAT wall claims most of
the room and the EAVE strip is narrow), low weight = slow wavefront = the
opposite.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

SK_SOLVER_BINARY = (
    Path(__file__).resolve().parents[2] / "tools" / "sk_solver" / "build" / "sk_solver"
)
SOLVER_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class SkVertex:
    x: float
    z: float
    time: float  # 0 on polygon boundary; positive on skeleton interior


@dataclass(frozen=True, slots=True)
class SkFace:
    edge_index: int  # which input polygon edge this face corresponds to
    vertices: list[SkVertex]


class SkSolverError(RuntimeError):
    pass


def solve_weighted_skeleton(
    polygon_xz: list[tuple[float, float]],
    weights: list[float],
) -> list[SkFace]:
    """Run sk_solver on a single polygon. Returns one SkFace per polygon edge
    that produced a face (some edges may be omitted if collinear/degenerate)."""
    if not SK_SOLVER_BINARY.exists():
        raise SkSolverError(
            f"sk_solver binary not built at {SK_SOLVER_BINARY}. Build with:\n"
            f"  cd tools/sk_solver && mkdir -p build && cd build && cmake .. && make"
        )
    if len(polygon_xz) < 3:
        raise SkSolverError("polygon must have >= 3 vertices")
    if len(weights) != len(polygon_xz):
        raise SkSolverError(
            f"weights ({len(weights)}) must match polygon length ({len(polygon_xz)})"
        )
    poly = list(polygon_xz)
    w = list(weights)
    n = len(poly)
    reversed_input = False
    if _signed_area(poly) < 0:
        poly = list(reversed(poly))
        w = list(reversed(w))
        reversed_input = True
    payload = json.dumps({"polygon": poly, "weights": w})
    try:
        proc = subprocess.run(
            [str(SK_SOLVER_BINARY)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=SOLVER_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SkSolverError(f"sk_solver timed out after {SOLVER_TIMEOUT_S}s") from exc
    if proc.returncode != 0:
        raise SkSolverError(
            f"sk_solver exit {proc.returncode}: {proc.stdout.strip()} / "
            f"{proc.stderr.strip()}"
        )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SkSolverError(f"sk_solver bad json: {proc.stdout[:200]!r}") from exc
    if not result.get("ok"):
        raise SkSolverError(f"sk_solver error: {result.get('error')}")
    faces: list[SkFace] = []
    for f in result.get("faces", []):
        verts = [SkVertex(x=v["x"], z=v["z"], time=v["time"]) for v in f["vertices"]]
        edge_idx = int(f["edge_index"])
        if reversed_input and 0 <= edge_idx < n:
            # Reversed[i] = original[n-1-i]; reversed edge i goes from
            # original[n-1-i] to original[n-2-i], i.e. original edge (n-2-i).
            edge_idx = (n - 2 - edge_idx) % n
        faces.append(SkFace(edge_index=edge_idx, vertices=verts))
    return faces


def _signed_area(poly: list[tuple[float, float]]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        area += (x1 - x0) * (z1 + z0)
    # Shoelace: positive for CW (in our xz where x→right, z→down/inward),
    # negative for CCW. CGAL wants CCW, so a negative signed area means
    # already CCW; positive means CW (needs reversal).
    return -area  # invert so positive = CCW
