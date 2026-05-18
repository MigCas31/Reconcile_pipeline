# How-To: Ingest With LLMs

## Recommended input order
1. `artifacts/scan-inventory/manifest.json`
2. `schemas/scan-inventory-field-reference.schema.json`
3. `artifacts/scan-inventory/field-reference.jsonl`
4. `docs/scan-inventory/reference/calor-usage-matrix.md` (only if narrative context is needed)

## Prompt hygiene
- Treat JSONL rows as the source of truth for classification and counts.
- Use markdown docs for explanation and caveats.
- Do not infer `used_directly` unless `usageInCalor` says so.
- Interpret `usageInCalor` as Calor-only behavior (ignore experimental usage in this repo).

## Suggested retrieval keys
- `fieldPath`
- `family`
- `usageInCalor`
- `trustLevel`

## Example query pattern
- "Find all fields where `usageInCalor=currently_unused` and `family=data`."
- "List geometry fields used directly in mergedFloor."
