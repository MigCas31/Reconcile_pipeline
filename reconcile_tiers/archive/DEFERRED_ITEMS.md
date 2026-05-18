# Deferred migration items

Date: 2026-04-26

These are the items we did not complete in the static tier migration and should
not leave mixed into active runtime code.

## Legacy deletion

The old tier viewer path was not deleted:

- `reconcile/viewer-tiers.html`
- `reconcile/viewer-modules/tier-preview.js`
- `reconcile/viewer_server.py` tier routes
- `reconcile/complexity_tiers.py`
- legacy classifier tests that still define the baseline

Reason: generated static payload tier counts still drift from the legacy
`/tier-index` baseline, and the old files still have legacy-only consumers
outside the new static viewer path. The static viewer at
`reconcile_tiers/web/viewer-tiers.html` must not use any of these files. See
`MIGRATION_AUDIT.md` for the blocker audit.

## Roof parity drift

The new roof path recovered the major directional-clustering and support-domain
issues, but generated-payload tier counts still over-produce tier 8 and
under-recover tier 7 relative to the legacy endpoint.

Reason: this is producer parity work, not a safe deletion cleanup.

## Thermal cap emission

The decision log says `thermal-cap` should be part of the new ceiling priority
model, but the current implementation still emits only knee walls and dormer
cheek/header thermal pieces.

Reason: `thermal-cap` needs an explicit producer implementation and cohort
review before it becomes runtime geometry.

## Process-only checks

Some original checklist items were process requirements rather than runtime
behaviour that can be proven in this checkout:

- red/green commit-history evidence for earlier phases;
- optional `hypothesis` property-test coverage where the dependency is not part
  of the current environment;
- coverage percentage gates where `pytest-cov` is not installed.

Reason: these remain review notes, not active runtime tasks.
