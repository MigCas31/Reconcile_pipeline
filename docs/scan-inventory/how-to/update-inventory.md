# How-To: Update The Inventory

## Regenerate Source Catalog

```bash
node .context/build-field-catalog.mjs
```

## Regenerate Human Dictionaries

```bash
node .context/generate-full-dictionary.mjs
node .context/generate-calor-usage-report.mjs
```

## Publish Team Package

```bash
node .context/publish-scan-inventory-package.mjs
```

## Validation Checks
- Ensure `artifacts/scan-inventory/field-reference.json` exists and is non-empty.
- Ensure `artifacts/scan-inventory/field-reference.jsonl` has one JSON object per line.
- Ensure schema file exists: `schemas/scan-inventory-field-reference.schema.json`.
- Spot-check sample values in `docs/scan-inventory/reference/field-reference.md` to avoid empty placeholders.
