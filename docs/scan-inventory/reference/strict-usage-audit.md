# Strict Usage Audit (Calor)

This matrix uses a strict rule:
- `used_directly`: explicit field consumption in Calor import/mapping logic.
- `used_indirectly_mapped`: container/collection traversal or conditional usage (for example object arrays where only stairs branch is consumed).
- `currently_unused`: no explicit consumption found.

## Key strict decisions
- Non-stair object categories (chair/bed/table/etc): `currently_unused`.
- `objects[].category.stairs`: `used_directly`.
- Surface `category` fields (walls/doors/windows/openings/floors): `currently_unused`.
- `confidence`, `completedEdges`, `sections`, `coreModel`, `referenceOriginTransform`, geometry-level `story`: `currently_unused`.
- Geometry primitives (`transform`, `dimensions`, `polygonCorners`, identifier/parentIdentifier where linked): used according to direct usage in segment creation and parent-link logic.

## Primary evidence files (Calor)
- `/Users/martincollignon/conductor/workspaces/calor/port-louis-v8/internal/domain/ios/roomplan/service.go`
- `/Users/martincollignon/conductor/workspaces/calor/port-louis-v8/internal/domain/ios/roomplan/mappers.go`
- `/Users/martincollignon/conductor/workspaces/calor/port-louis-v8/internal/domain/ios/roomplan/app_models.go`
- `/Users/martincollignon/conductor/workspaces/calor/port-louis-v8/internal/domain/ios/roomplan/ios_models.go`
