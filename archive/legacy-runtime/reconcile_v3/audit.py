"""Hypothesis tracing — every emitted element records why it exists."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

_geom_log = logging.getLogger("reconcile_v3.geom")


def note_geom_skip(exc: BaseException, where: str = "") -> None:
    """Audit-log a Shapely op failure that was intentionally caught and skipped.

    Use this in every ``except`` block that wraps a Shapely call and would
    otherwise silently ``pass``/``continue`` — it surfaces the failure
    via the standard logging system (default level: DEBUG, off in
    production) so the underlying TopologicalError/GEOSException can be
    audited if a downstream metric looks wrong.
    """
    _geom_log.debug("geom-op skip%s: %s", f" [{where}]" if where else "", exc)


@dataclass(frozen=True)
class HypothesisTrace:
    stage: str
    rule: str
    inputs: dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "rule": self.rule,
            "inputs": dict(self.inputs),
            "decision_reason": self.decision_reason,
        }


@dataclass
class AuditLog:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, element_id: str, trace: HypothesisTrace) -> None:
        self.entries.append({"element_id": element_id, **trace.to_dict()})

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.entries)
