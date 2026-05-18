# Explanation: Geometry And Mapping

## Three Geometry Layers
1. Standalone room payloads (raw): room-local geometry with `polygonCorners` and local transforms.
2. Merged floor payloads (derived): cross-room merged geometry with reconciled room set.
3. Ceiling variants: separate ceiling capture/merge channels plus minimal source metadata.

## Why polygons differ between raw and merged
Merged payloads are not a simple concat of room files. Apple merge can alter transforms, dimensions, and identifiers.

## Calor reconciliation strategy
- Room identity is anchored through metadata room IDs and wall intersection mapping.
- Wall geometry mismatch is corrected by combining raw wall geometry with merged transform (`NewSegmentFromIOSSurfaceRaw`).
- Parent links for windows/doors/openings rely on merged `parentIdentifier`.
- Orphaned parent refs are tolerated and logged as known RoomPlan merge quality issues.

## Trust model
- `authoritative_*`: source-exported fields in that layer.
- `derived_*`: merged/processed representations.
- `*_optional`: sparse/nullable or not always emitted.
