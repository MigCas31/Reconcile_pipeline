---
name: triage-queue
description: >
  Use when the user pastes a flag-queue path (.context/flag-queues/<uuid>/<timestamp>.json)
  or asks to triage a building's flag queue. The queue can come from the viewer
  (Shift+right-click flags) or the cohort defect scanner
  (`python -m reconcile_tiers.audit.cohort_scan`). Group items by rule, run
  debug-element on representatives, propose fixes targeting rules — not
  individual elements.
---

# Triage a Flag Queue

The flag queue is a building-scoped batch of suspected geometry defects. It replaces the previous single-element loop where the user copy-pasted one locator at a time. Every queue file conforms to the schema below and may include automatic findings (the `rule` field is set) or manual flags from the viewer (`rule` is null, `note` may carry user intent).

The job here is **batch triage** — find the underlying rule that produced N items, fix the rule, not each instance.

## Queue file schema (`flag-queue/v1`)

Path: `.context/flag-queues/<building_uuid>/<timestamp>.json` (and an `auto-scan-latest.json` pointer alongside the most recent auto-scan).

```json
{
  "schema": "flag-queue/v1",
  "building_uuid": "016980bc-…",
  "created": "20260428T180000Z",
  "source": "viewer" | "auto-scan" | "merged",
  "screenshot": "20260428T180000Z.png" | null,
  "items": [
    {
      "id": "f7b3…",
      "locator": "<uuid>::tier-ceiling-raw::1:0:2",
      "kind": "tier-ceiling-raw",
      "parts": ["1", "0", "2"],
      "rule": "ceiling_orientation_inverted" | null,
      "note": "this looks tilted" | null,
      "severity": "high" | "med" | "low" | null,
      "evidence": { "normal_y": 0.04, … } | null,
      "dismissed": false
    }
  ]
}
```

Each rule is implemented in `reconcile_tiers/audit/rules.py`. The same rule names are reused across cohort scans so frequencies are comparable building-to-building.

## Triage protocol

When the user gives you a queue path:

1. **Read the queue.** Skip items where `dismissed: true` — the user already vetoed them.
2. **Group by `rule`.** Manual items (`rule == null`) cluster by `note` keyword or just by `kind`.
3. **Pick representatives, not exhaustively.** For each group of size N, pick 1 (small group) or 2 (large/heterogeneous group) representatives. Running debug-element on every item is wasteful when the underlying cause is shared.
4. **Run `debug-element` on representatives in parallel** via subagents. Each agent gets the locator and the queue's `evidence` for that item.
5. **Synthesise.** For each group, state:
   - the rule that fired and the user-visible symptom,
   - the underlying pipeline step / threshold the representatives trace back to,
   - the proposed fix targeting the rule (e.g. "fix the ceiling-orientation step in `reconcile_tiers/assemble/ceiling_painter.py:NNN` so flat-emit normals always face +Y", not "reorient these 13 corners").
6. **Rank groups** by severity + group size. Largest high-severity group first.

## Human-first stance still applies

Auto-scan rules are heuristics — they encode "what would a human notice from one angle" into thresholds. The framework rules in `CLAUDE.md` (synthesis follows scan, no public DK data as truth, keep diagnostic signal, etc.) trump the rule output. If a rule fires for the wrong reason, the fix may be to the rule itself — note it, propose a refinement, but do not silently widen thresholds.

## Counter-examples — when not to use this skill

- One-off element triage with no queue file: use `debug-element` directly.
- Building-wide health check with no specific defects yet: use `inspect-building` first to generate screenshots + audit, then this skill to chew through what audit found.
- Performance / pipeline-runtime questions: this skill does not profile; use the relevant pipeline skill.

## Producers

The queue can be produced two ways. Both write the same schema.

| Producer | How |
|---|---|
| Cohort defect scanner | `python -m reconcile_tiers.audit.cohort_scan --uuid <uuid>` (or `--all`) |
| Viewer flag button | Shift+right-click in `viewer-tiers.html`, accumulate flags, click "Send queue" |

Both write to `.context/flag-queues/<uuid>/<timestamp>.json`. The viewer also fetches `auto-scan-latest.json` on building load so the user can review/refine auto-flags before sending.

## After triage

Update `tracking_progress.md` per the project convention with what you found and what you proposed — even if the fix isn't applied this session. The queue file itself is a useful artefact; do not delete it.
