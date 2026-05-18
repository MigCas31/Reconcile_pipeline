# Root archive

This directory holds legacy runtime packages that are no longer part of the
static `reconcile_tiers` viewer path.

Current layout:

- `legacy-runtime/reconcile`
- `legacy-runtime/reconcile_ext`
- `legacy-runtime/reconcile_v2`
- `legacy-runtime/reconcile_v3`
- `legacy-runtime/scripts`
- `legacy-runtime/reports`
- `legacy-runtime/artifacts`
- `legacy-runtime/tests`
- `legacy-runtime/docs`

The repository root keeps symlinks named `reconcile`, `reconcile_ext`,
`reconcile_v2`, `reconcile_v3`, `scripts`, `reports`, and `artifacts` pointing
here. That keeps old scripts, generated report references, and reference
commands working while making the archive boundary explicit.

Legacy root `tests/test_*.py` files are moved under `legacy-runtime/tests`
without a root compatibility symlink. Active tier tests stay under
`tests/reconcile_tiers/`.

The static viewer at `reconcile_tiers/web/viewer-tiers.html` must not import or
fetch through these archived packages. It should use only static
`pipeline-outputs/tier_index.json` and
`pipeline-outputs/{uuid}/tier_payload.json` files produced by
`reconcile_tiers`.
