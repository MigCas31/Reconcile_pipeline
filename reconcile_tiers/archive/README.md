# reconcile_tiers archive

This directory is for migration work that was reviewed but deliberately not
executed as part of the static `reconcile_tiers` migration.

Files here are non-runtime records. Moving a document here must not change the
payload builder, renderer, generated artefacts, or legacy viewer behaviour.

The static viewer at `reconcile_tiers/web/viewer-tiers.html` is not allowed to
use the old `reconcile/`, `reconcile_v2/`, or `reconcile_v3/` code paths. It
loads only static `pipeline-outputs/tier_index.json` and
`pipeline-outputs/{uuid}/tier_payload.json` artefacts produced by
`reconcile_tiers`.

Current contents:

- `MIGRATION_AUDIT.md`: Phase J audit for the legacy tier viewer path. It
  records why deletion was blocked and which legacy files still have live
  consumers.
- `DEFERRED_ITEMS.md`: Short index of migration scope that was not completed or
  was converted into follow-up work.
