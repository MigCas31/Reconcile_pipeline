# Scan Inventory Documentation

This package is structured for both team reading and LLM ingestion.
All `usage` classifications in this package refer to **Calor** only (not this workspace's experimental pipelines).

## Structure
- how-to: maintenance and LLM ingestion procedures
- reference: field definitions, strict usage rules, and Calor usage matrix
- explanation: geometry and mapping rationale
- schemas: JSON Schema contracts
- artifacts: machine-readable exports (JSON + JSONL)

## Entry points
- Human readers: `docs/scan-inventory/reference/calor-usage-matrix.md`
- Strict rules: `docs/scan-inventory/reference/strict-usage-audit.md`
- LLM ingestion flow: `docs/scan-inventory/how-to/llm-ingestion.md`

## Canonical Artifacts
- Field reference JSON: `artifacts/scan-inventory/field-reference.json`
- Field reference JSONL: `artifacts/scan-inventory/field-reference.jsonl`
- Field catalog JSON: `artifacts/scan-inventory/field-catalog.json`
- Schema: `schemas/scan-inventory-field-reference.schema.json`

## Coverage
- Total fields: **681**
- Families: data, roomplan, mergedFloor, ceiling, ceilingMerged, ceilingMetadata
- Usage counts: {"currently_unused":239,"used_directly":292,"used_indirectly_mapped":150}
