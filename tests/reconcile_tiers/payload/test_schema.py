import json

from reconcile_tiers.payload.emit_jsonschema import emit_schema, schema_json
from reconcile_tiers.payload.schema import (
    CeilingPiece,
    CeilingSource,
    DormerFace,
    DormerFaceKind,
    GableClosure,
    GableClosureKind,
    GapKind,
    GapPiece,
    GapScope,
    HorizontalLid,
    KneeWall,
    KneeWallKind,
    Plane,
    Quad,
    RoofType,
    Room,
    TierClassification,
    TierPayload,
    Vec3,
    Wall,
    payload_from_dict,
    payload_to_dict,
)


def _square_lid(y: float) -> HorizontalLid:
    return HorizontalLid(
        corners=[
            Vec3(0.0, y, 1.0),
            Vec3(1.0, y, 1.0),
            Vec3(1.0, y, 0.0),
            Vec3(0.0, y, 0.0),
        ]
    )


def sample_payload() -> TierPayload:
    wall = Wall(
        corners=[
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(1.0, 1.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        ],
        descent_strip=None,
        uplift_strip=None,
        cutouts=[
            Quad(
                corners=[
                    Vec3(0.2, 0.2, 0.0),
                    Vec3(0.4, 0.2, 0.0),
                    Vec3(0.4, 0.5, 0.0),
                    Vec3(0.2, 0.5, 0.0),
                ]
            )
        ],
        locator_id="uuid-1::tier-wall::0:0",
    )
    return TierPayload(
        schema_version="1",
        uuid="uuid-1",
        address="Examplevej 1",
        building_center=Vec3(0.5, 0.5, 0.5),
        classification=TierClassification(
            tier=6,
            tier_label="Gable roof",
            roof_type=RoofType.GABLE,
            n_stories=1,
            n_rooms=1,
            n_oblique=2,
            n_flat=0,
            has_half_height=False,
            has_gable=True,
        ),
        rooms=[
            Room(
                story=0,
                floor=[_square_lid(0.0)],
                walls=[wall],
                doors=[],
                windows=[],
                locator_id="uuid-1::tier-room::0",
            )
        ],
        gaps=[
            GapPiece(
                corners=_square_lid(0.0).corners,
                kind=GapKind.GAP_FLOOR,
                scope=GapScope.INTRA_STORY,
                locator_id="uuid-1::tier-gap::0",
            )
        ],
        ceiling=[
            CeilingPiece(
                corners=_square_lid(2.5).corners,
                holes=[],
                plane=Plane(0.0, 1.0, 0.0, 2.5),
                source=CeilingSource.FLAT_EMIT,
                arrangement_cell_id=None,
                locator_id="uuid-1::tier-ceiling::0",
            )
        ],
        knee_walls=[
            KneeWall(
                corners=[
                    Vec3(0.0, 1.0, 0.0),
                    Vec3(1.0, 1.0, 0.0),
                    Vec3(1.0, 2.0, 0.0),
                    Vec3(0.0, 2.0, 0.0),
                ],
                kind=KneeWallKind.KNEE,
                locator_id="uuid-1::tier-knee-wall::0",
            )
        ],
        dormer_faces=[
            DormerFace(
                corners=[
                    Vec3(0.2, 1.0, 0.2),
                    Vec3(0.4, 1.0, 0.2),
                    Vec3(0.4, 1.3, 0.2),
                    Vec3(0.2, 1.3, 0.2),
                ],
                kind=DormerFaceKind.DORMER_HEADER,
                locator_id="uuid-1::tier-dormer-face::dormer_header:0",
                cutouts=[],
            )
        ],
        gable_closures=[
            GableClosure(
                corners=[
                    Vec3(0.0, 1.0, 0.0),
                    Vec3(1.0, 1.0, 0.0),
                    Vec3(0.5, 1.5, 0.0),
                ],
                kind=GableClosureKind.UPPER,
                locator_id="uuid-1::tier-gable-closure::upper:0",
            )
        ],
    )


def test_schema_emits_expected_top_level_keys():
    schema = emit_schema(TierPayload)

    assert set(schema["properties"]) >= {
        "schema_version",
        "uuid",
        "rooms",
        "gaps",
        "ceiling",
        "knee_walls",
        "dormer_faces",
        "classification",
    }
    assert schema["required"] == [
        "schema_version",
        "uuid",
        "address",
        "building_center",
        "classification",
        "rooms",
        "gaps",
        "ceiling",
        "knee_walls",
        "dormer_faces",
        "gable_closures",
    ]


def test_payload_round_trips_through_json_dict():
    payload = sample_payload()

    encoded = json.loads(json.dumps(payload_to_dict(payload)))
    decoded = payload_from_dict(encoded)

    assert decoded == payload


def test_payload_decodes_legacy_dormer_faces_without_cutouts():
    payload = payload_to_dict(sample_payload())
    del payload["dormer_faces"][0]["cutouts"]

    decoded = payload_from_dict(payload)

    assert decoded.dormer_faces[0].cutouts == []


def test_schema_is_stable():
    with open("reconcile_tiers/payload/tier_payload_schema.json") as f:
        committed = f.read()

    assert schema_json(TierPayload) == committed


def test_gap_kind_is_typed_enum_not_free_substring_contract():
    assert {kind.value for kind in GapKind} == {
        "gap_floor",
        "gap_ceiling",
        "side",
        "stitch",
        "stitch_floor",
        "stitch_ceiling",
        "exterior_side",
        "exterior_floor",
        "exterior_ceiling",
    }
    assert all(
        "_" in kind.value or kind.value in {"side", "stitch"} for kind in GapKind
    )


def test_dormer_face_kind_includes_visible_front_face():
    assert {kind.value for kind in DormerFaceKind} == {
        "dormer_cheek",
        "dormer_header",
        "dormer_front",
    }
