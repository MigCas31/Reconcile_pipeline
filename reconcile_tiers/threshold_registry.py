"""Registry of tunable thresholds the dev workflow keeps reaching for.

Powering the ``apply_threshold_tweak`` Gemini tool. Each entry maps a
short name → file path + module-level attribute name, plus an optional
human-readable description and a (low, high) sanity bound.

Adding a constant: confirm the file declares it as a module-level literal
(``NAME = value``) — the editor uses a regex line-rewrite, so anything more
exotic (computed, conditional, class-level) needs a smarter editor.

NOTE: ``AZIMUTH_FILTER_THRESHOLD`` is intentionally absent — see the
"No 90° azimuth relaxation" feedback memory. The 180° filter was
deliberate; tuning it caused production regressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ThresholdEntry:
    name: str
    file: str  # workspace-relative
    attribute: str  # module-level name
    description: str
    low: float | None = None
    high: float | None = None
    unit: str | None = None


REGISTRY: dict[str, ThresholdEntry] = {
    "WALL_HALF_M": ThresholdEntry(
        name="WALL_HALF_M",
        file="reconcile_tiers/extract/gaps.py",
        attribute="WALL_HALF_M",
        description="Half-wall morphological buffer for cross-floor / inter-story gap "
        "detection.",
        low=0.05,
        high=0.5,
        unit="m",
    ),
    "MAX_GAP_M_GAPS": ThresholdEntry(
        name="MAX_GAP_M_GAPS",
        file="reconcile_tiers/extract/gaps.py",
        attribute="MAX_GAP_M",
        description="Max distance considered when pairing rooms for gap inference.",
        low=0.3,
        high=2.5,
        unit="m",
    ),
    "MAX_WALL_THICKNESS_M": ThresholdEntry(
        name="MAX_WALL_THICKNESS_M",
        file="reconcile_tiers/extract/wall_pairs.py",
        attribute="MAX_WALL_THICKNESS_M",
        description="Maximum distance between deduplicated walls to count as a single "
        "wall pair.",
        low=0.2,
        high=1.5,
        unit="m",
    ),
    "MAX_GAP_TO_SLAB_M": ThresholdEntry(
        name="MAX_GAP_TO_SLAB_M",
        file="reconcile_tiers/extract/extension.py",
        attribute="MAX_GAP_TO_SLAB_M",
        description="Max wall-top extension distance to story slab during ceiling lid "
        "lift.",
        low=0.1,
        high=2.0,
        unit="m",
    ),
    "STITCH_MAX_GAP_M": ThresholdEntry(
        name="STITCH_MAX_GAP_M",
        file="reconcile_tiers/extract/stitches.py",
        attribute="MAX_GAP_M",
        description="Max gap between rooms to bridge with a stitch wall.",
        low=0.3,
        high=3.0,
        unit="m",
    ),
    "STITCH_DIRECT_QUAD_MAX_GAP_M": ThresholdEntry(
        name="STITCH_DIRECT_QUAD_MAX_GAP_M",
        file="reconcile_tiers/extract/stitches.py",
        attribute="DIRECT_QUAD_MAX_GAP_M",
        description="Below this stitch gap, replace L-closure with direct quad (avoids "
        "degenerate triangles).",
        low=0.05,
        high=0.5,
        unit="m",
    ),
    "STITCH_T_JUNCTION_MAX_GAP_M": ThresholdEntry(
        name="STITCH_T_JUNCTION_MAX_GAP_M",
        file="reconcile_tiers/extract/stitches.py",
        attribute="T_JUNCTION_MAX_GAP_M",
        description="Max gap to treat as a T-junction tied to a non-owner wall.",
        low=0.02,
        high=0.3,
        unit="m",
    ),
}


def list_entries() -> list[dict[str, object]]:
    return [
        {
            "name": e.name,
            "file": e.file,
            "attribute": e.attribute,
            "description": e.description,
            "low": e.low,
            "high": e.high,
            "unit": e.unit,
        }
        for e in REGISTRY.values()
    ]


def get(name: str) -> ThresholdEntry:
    if name not in REGISTRY:
        raise KeyError(f"unknown threshold {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def read_current(name: str) -> tuple[float, str, int]:
    """Return (current_value, file_path, line_number)."""
    entry = get(name)
    path = WORKSPACE_ROOT / entry.file
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    pattern = re.compile(rf"^(\s*){entry.attribute}\s*=\s*({_NUMBER_RE.pattern})")
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        m = pattern.match(line)
        if m:
            return float(m.group(2)), str(path), i
    raise LookupError(
        f"attribute {entry.attribute!r} not found at module scope of {entry.file}"
    )


def make_diff(name: str, new_value: float) -> dict[str, object]:
    """Return a structured diff for confirmation, without writing.

    Validates the value falls within the entry's sanity bounds.
    """
    entry = get(name)
    current, path_str, line_no = read_current(name)
    if entry.low is not None and new_value < entry.low:
        raise ValueError(
            f"{name}: value {new_value} below sanity floor "
            f"{entry.low}{entry.unit or ''}"
        )
    if entry.high is not None and new_value > entry.high:
        raise ValueError(
            f"{name}: value {new_value} above sanity ceiling "
            f"{entry.high}{entry.unit or ''}"
        )
    path = Path(path_str)
    lines = path.read_text().splitlines()
    old_line = lines[line_no - 1]
    new_line = re.sub(_NUMBER_RE, _format(new_value), old_line, count=1)
    return {
        "name": name,
        "file": entry.file,
        "line": line_no,
        "current": current,
        "new": new_value,
        "diff": {"-": old_line, "+": new_line},
    }


def _format(value: float) -> str:
    if float(int(value)) == value:
        return str(int(value))
    return repr(value)


def apply(name: str, new_value: float) -> dict[str, object]:
    """Write the new value to disk after re-validating."""
    diff = make_diff(name, new_value)
    path = WORKSPACE_ROOT / diff["file"]  # type: ignore[index]
    lines = path.read_text().splitlines(keepends=True)
    line_idx = int(diff["line"]) - 1  # type: ignore[index]
    new_content = diff["diff"]["+"] + (
        "\n" if not lines[line_idx].endswith("\n") else ""
    )
    if lines[line_idx].endswith("\n") and not new_content.endswith("\n"):
        new_content += "\n"
    lines[line_idx] = new_content
    path.write_text("".join(lines))
    return diff


__all__ = [
    "REGISTRY",
    "ThresholdEntry",
    "apply",
    "get",
    "list_entries",
    "make_diff",
    "read_current",
]
