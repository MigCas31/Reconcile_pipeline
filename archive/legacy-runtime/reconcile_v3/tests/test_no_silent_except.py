"""Flag silent exception suppression inside reconcile_v3/.

Every caught exception must either re-raise, return a structured
unresolved region, or surface via an audit trail. Bare ``pass`` /
``continue`` after ``except`` is banned here.
"""

from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
SILENT_PATTERN = re.compile(
    r"except[^\n:]*:\s*(?:#.*\n)?\s*(pass|continue)\b",
    re.MULTILINE,
)


def _python_sources() -> list[Path]:
    return [p for p in PKG.rglob("*.py") if "tests" not in p.parts]


def test_no_silent_except():
    offenders: list[str] = []
    for path in _python_sources():
        text = path.read_text()
        for m in SILENT_PATTERN.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{path.relative_to(PKG)}:{line}")
    assert not offenders, (
        "Silent exception handlers are not allowed in reconcile_v3 "
        "(they hide data bugs). Offenders: " + ", ".join(offenders)
    )
