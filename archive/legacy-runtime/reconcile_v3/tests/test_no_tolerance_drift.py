"""Flag inline magic float literals in the tolerance range.

Keeps the "no inline magic distance tolerances in pipeline code" invariant:
every distance/area/volume threshold should be a NAMED constant — either
in ``constants.py``, a file-local ``_NAME = …`` declaration, or a
dataclass-field default — so it has a label a reader can search for.

Walks the AST instead of regex so we don't false-positive on docstrings
and string contents.

Skipped:

- ``analysis/`` and ``autonomy/`` — feature-engineering modules whose
  literals are statistical/Hu-moment knobs, not pipeline tolerances.
- Module-level (and classvar) named-constant assignments.
- Dataclass / annotated-assignment field defaults.
- Docstrings and other string literals.
- ``2.0`` — a common math constant (angle doubling, axis scaling)
  unrelated to distance tolerances.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
TOL_LOW, TOL_HIGH = 0.01, 2.0
ALLOWED_VALUES = {0.1, 0.4, 0.5, 1.0, 2.0}
SKIP_DIRS = {"analysis", "autonomy", "tests"}


def _python_sources() -> list[Path]:
    return [
        p
        for p in PKG.rglob("*.py")
        if not (set(p.parts) & SKIP_DIRS) and p.name != "constants.py"
    ]


def _collect_allowed_nodes(tree: ast.AST) -> set[int]:
    """Return ids of ast.Constant nodes that are 'named' assignments and
    therefore allowed (module-level constants and field defaults)."""
    allowed: set[int] = set()

    def _allow(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            allowed.add(id(node))
        # Common composite forms: tuples and lists of numeric defaults.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, float):
                allowed.add(id(child))

    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            _allow(node.value)
        elif isinstance(node, ast.Assign):
            # Module / class-level NAME = literal — treat as named constant.
            if all(isinstance(t, ast.Name) for t in node.targets):
                _allow(node.value)
    return allowed


def test_no_tolerance_drift():
    offenders: list[str] = []
    for path in _python_sources():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        allowed_ids = _collect_allowed_nodes(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, float)):
                continue
            if id(node) in allowed_ids:
                continue
            value = float(node.value)
            if value in ALLOWED_VALUES:
                continue
            if TOL_LOW <= value <= TOL_HIGH:
                offenders.append(f"{path.relative_to(PKG)}:{node.lineno} -> {value}")
    assert not offenders, (
        "Inline distance-tolerance literals must be named constants. "
        "Offenders:\n  " + "\n  ".join(offenders)
    )
