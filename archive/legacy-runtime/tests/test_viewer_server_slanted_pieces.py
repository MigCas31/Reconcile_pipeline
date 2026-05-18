from __future__ import annotations

import json
import os

from shapely.geometry import Polygon

from reconcile import viewer_server as vs

UUID = "viewer-tier-test-building"


def _write_v2_sidecar(tmp_path, pieces):
    path = tmp_path / "plane_extent_splits.json"
    path.write_text(json.dumps({"buildings": {UUID: pieces}}))
    return path


def _patch_v2_sidecar(monkeypatch, path):
    monkeypatch.setitem(vs.RAW_CEILING_PLANE_SPLITS_PATHS, "v2", path)
    monkeypatch.setitem(vs.RAW_CEILING_PLANE_SPLITS_CACHE_BY_VERSION, "v2", {})
    monkeypatch.setitem(vs.RAW_CEILING_PLANE_SPLITS_CACHE_MTIME_BY_VERSION, "v2", 0.0)


def _patch_roof_results_path(monkeypatch, tmp_path, *, mtime):
    path = tmp_path / "roof_algorithms_py_results.json"
    path.write_text("{}")
    os.utime(path, (mtime, mtime))
    monkeypatch.setattr(vs, "ROOF_RESULTS_PATH", path)
    return path


def _square(x0=0.0, y=1.0, z0=0.0, size=1.0):
    return [
        [x0, y, z0],
        [x0 + size, y, z0],
        [x0 + size, y, z0 + size],
        [x0, y, z0 + size],
    ]


def _rect(x0=0.0, y=1.0, z0=0.0, width=1.0, depth=1.0):
    return [
        [x0, y, z0],
        [x0 + width, y, z0],
        [x0 + width, y, z0 + depth],
        [x0, y, z0 + depth],
    ]


def _xz_area(pieces):
    total = 0.0
    for piece in pieces:
        corners = piece.get("corners") or piece.get("poly") or []
        if len(corners) < 3:
            continue
        total += Polygon([(c[0], c[2]) for c in corners]).area
    return total


def _final_piece(piece_id, corners=None, **extra):
    return {
        "piece_id": piece_id,
        "target_element_id": f"{UUID}::ridge-eave-candidate::plane-group::a",
        "piece_role": "supported",
        "target_kind": "ridge_eave_candidate",
        "final_layer": True,
        "corners": corners or _square(),
        "roof_hypothesis_id": "roof-hypothesis:oblique:1",
        **extra,
    }


def test_slanted_pieces_prefer_v2_over_backend_arrangement(tmp_path, monkeypatch):
    sidecar = _write_v2_sidecar(tmp_path, [_final_piece("v2-piece")])
    os.utime(sidecar, (2000.0, 2000.0))
    _patch_v2_sidecar(monkeypatch, sidecar)
    _patch_roof_results_path(monkeypatch, tmp_path, mtime=1000.0)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    monkeypatch.setitem(
        vs.ROOF_RESULTS_CACHE,
        UUID,
        {
            "roof_surfaces": {
                "oblique_split": [
                    {
                        "corners": _square(10.0, 2.0),
                        "roof_hypothesis_id": "roof-hypothesis:oblique:old",
                    }
                ]
            }
        },
    )

    pieces = vs._slanted_pieces_for_uuid(UUID)

    assert len(pieces) == 1
    assert pieces[0]["source"] == "v2_sidecar"
    assert pieces[0]["piece_id"] == "v2-piece"
    assert pieces[0]["target_element_id"] == (
        f"{UUID}::ridge-eave-candidate::plane-group::a"
    )
    assert pieces[0]["target_kind"] == "ridge_eave_candidate"
    assert pieces[0]["roof_hypothesis_id"] == "roof-hypothesis:oblique:1"


def test_slanted_pieces_prefer_newer_backend_arrangement(tmp_path, monkeypatch):
    sidecar = _write_v2_sidecar(tmp_path, [_final_piece("stale-v2-piece")])
    os.utime(sidecar, (1000.0, 1000.0))
    _patch_v2_sidecar(monkeypatch, sidecar)
    _patch_roof_results_path(monkeypatch, tmp_path, mtime=2000.0)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    monkeypatch.setitem(
        vs.ROOF_RESULTS_CACHE,
        UUID,
        {
            "roof_surfaces": {
                "oblique_split": [
                    {
                        "corners": _square(10.0, 2.0),
                        "arrangement_cell_id": "cell-new",
                        "roof_hypothesis_id": "roof-hypothesis:oblique:new",
                        "intersection_kind": "single",
                    }
                ]
            }
        },
    )

    pieces = vs._slanted_pieces_for_uuid(UUID)

    assert len(pieces) == 1
    assert pieces[0]["source"] == "roof_arrangement"
    assert pieces[0]["arrangement_cell_id"] == "cell-new"
    assert pieces[0]["roof_hypothesis_id"] == "roof-hypothesis:oblique:new"


def test_slanted_pieces_prefer_backend_oblique_surfaces_over_split_cells(
    tmp_path, monkeypatch
):
    sidecar = _write_v2_sidecar(tmp_path, [_final_piece("stale-v2-piece")])
    os.utime(sidecar, (1000.0, 1000.0))
    _patch_v2_sidecar(monkeypatch, sidecar)
    _patch_roof_results_path(monkeypatch, tmp_path, mtime=2000.0)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    monkeypatch.setitem(
        vs.ROOF_RESULTS_CACHE,
        UUID,
        {
            "roof_surfaces": {
                "oblique": [
                    {
                        "corners": _square(20.0, 2.0),
                        "roof_hypothesis_id": "roof-hypothesis:oblique:whole",
                    }
                ],
                "oblique_split": [
                    {
                        "corners": _square(10.0, 2.0),
                        "roof_hypothesis_id": "roof-hypothesis:oblique:split",
                    }
                ],
            }
        },
    )

    pieces = vs._slanted_pieces_for_uuid(UUID)

    assert len(pieces) == 1
    assert pieces[0]["source"] == "roof_arrangement"
    assert pieces[0]["roof_hypothesis_id"] == "roof-hypothesis:oblique:whole"
    assert pieces[0]["corners"] == _square(20.0, 2.0)


def test_v2_final_loader_matches_overlay_and_gate_is_opt_in(tmp_path, monkeypatch):
    unsupported_lower = _final_piece(
        "unsupported-lower",
        corners=_square(0.0, 1.0),
        chain_ids=[],
        piece_anchor_chain_count=0,
        creator_rain_area_fraction=0.0,
    )
    supported_upper = _final_piece(
        "supported-upper",
        corners=_square(0.0, 1.5),
        chain_ids=["raw-chain:1"],
    )
    sidecar = _write_v2_sidecar(tmp_path, [unsupported_lower, supported_upper])
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])

    overlay_pieces = vs._load_v2_final_pieces(UUID)
    gated_pieces = vs._load_v2_final_pieces(UUID, gate_unsupported=True)
    full_model_pieces = vs._load_v2_full_model_pieces(UUID)
    slanted_pieces = vs._slanted_pieces_for_uuid(UUID)

    assert {p["piece_id"] for p in overlay_pieces} == {
        "unsupported-lower",
        "supported-upper",
    }
    assert {p["piece_id"] for p in gated_pieces} == {"supported-upper"}
    assert {p["piece_id"] for p in full_model_pieces} == {
        "unsupported-lower",
        "supported-upper",
    }
    assert {p["piece_id"] for p in slanted_pieces} == {
        "unsupported-lower",
        "supported-upper",
    }


def test_v2_loader_excludes_suppressed_and_candidate_pieces(tmp_path, monkeypatch):
    pieces = [
        _final_piece("visible-final"),
        _final_piece("suppressed-final", overlay_suppressed=True),
        {
            **_final_piece("candidate"),
            "final_layer": False,
            "target_kind": "ridge_eave_candidate",
            "piece_role": "supported",
        },
        {
            **_final_piece("committed"),
            "final_layer": False,
            "target_kind": "committed_oblique",
        },
        {
            **_final_piece("seam"),
            "final_layer": False,
            "piece_role": "intersection_seam",
        },
    ]
    sidecar = _write_v2_sidecar(tmp_path, pieces)
    _patch_v2_sidecar(monkeypatch, sidecar)

    loaded = vs._load_v2_final_pieces(UUID)

    assert {p["piece_id"] for p in loaded} == {"visible-final", "committed", "seam"}


def test_full_model_pieces_exclude_overlay_only_context(tmp_path, monkeypatch):
    pieces = [
        _final_piece("visible-final"),
        {
            **_final_piece("committed-context"),
            "final_layer": False,
            "target_kind": "committed_oblique",
        },
        {
            **_final_piece("intersection-seam"),
            "final_layer": False,
            "piece_role": "intersection_seam",
        },
    ]
    sidecar = _write_v2_sidecar(tmp_path, pieces)
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])

    full_model_pieces = vs._load_v2_full_model_pieces(UUID)
    slanted_pieces = vs._slanted_pieces_for_uuid(UUID)

    assert {p["piece_id"] for p in full_model_pieces} == {"visible-final"}
    assert {p["piece_id"] for p in slanted_pieces} == {"visible-final"}


def test_full_model_filters_gap_ceiling_lids_under_slanted_roof():
    slanted_roof_xz = vs._slanted_pieces_xz_union(
        [{"corners": _square(0.0, 2.0, 0.0, 2.0)}]
    )
    gap_records = [
        {"type": "gap_ceiling", "corners": _square(0.25, 1.0, 0.25, 0.5)},
        {"type": "gap_floor", "corners": _square(0.25, 0.0, 0.25, 0.5)},
        {"type": "gap_ceiling", "corners": _square(3.0, 1.0, 3.0, 0.5)},
        {"type": "within_story", "corners": _square(0.5, 1.0, 0.5, 0.5)},
    ]

    filtered = vs._filter_gap_ceiling_caps_for_full_model(gap_records, slanted_roof_xz)

    assert [r["type"] for r in filtered] == [
        "gap_floor",
        "gap_ceiling",
        "within_story",
    ]
    assert filtered[1]["corners"] == gap_records[2]["corners"]


def test_full_model_removes_only_covered_cross_floor_ceiling_lid():
    slanted_pieces = [{"corners": _square(0.0, 1.1, 0.0, 2.0)}]
    records = [
        {
            "corners": _square(0.25, 0.0, 0.25, 0.5),
            "ceiling_corners": _square(0.25, 1.0, 0.25, 0.5),
            "type": "cross_story",
        },
        {
            "corners": _square(3.0, 0.0, 3.0, 0.5),
            "ceiling_corners": _square(3.0, 1.0, 3.0, 0.5),
            "type": "cross_story",
        },
    ]

    filtered = vs._filter_cross_floor_gap_ceiling_lids_for_full_model(
        records, slanted_pieces
    )

    assert filtered[0]["corners"] == records[0]["corners"]
    assert filtered[0]["ceiling_corners"] is None
    assert filtered[1]["ceiling_corners"] == records[1]["ceiling_corners"]


def test_full_model_keeps_intermediate_cross_floor_ceiling_below_high_roof():
    slanted_pieces = [{"corners": _square(0.0, 7.0, 0.0, 2.0)}]
    records = [
        {
            "corners": _square(0.25, 0.0, 0.25, 0.5),
            "ceiling_corners": _square(0.25, 1.0, 0.25, 0.5),
            "type": "cross_story",
        },
    ]

    filtered = vs._filter_cross_floor_gap_ceiling_lids_for_full_model(
        records, slanted_pieces
    )

    assert filtered[0]["ceiling_corners"] == records[0]["ceiling_corners"]


def test_flat_raw_ceiling_fallback_clips_only_under_slanted_piece_above(
    tmp_path, monkeypatch
):
    sidecar = _write_v2_sidecar(
        tmp_path,
        [_final_piece("slanted-above", corners=_rect(0.0, 2.0, 0.0, 1.0, 2.0))],
    )
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    building = {
        "rooms": [
            {
                "ceiling_type": "flat",
                "ceiling_polygon": _rect(0.0, 1.0, 0.0, 2.0, 2.0),
                "raw_ceiling_planes": [{"corners": _rect(0.0, 1.0, 0.0, 2.0, 2.0)}],
            }
        ]
    }

    pieces = vs._raw_ceiling_fallback_for_uuid(UUID, building)

    assert {p["source"] for p in pieces} == {"raw_flat_ceiling"}
    assert round(_xz_area(pieces), 6) == 2.0


def test_flat_raw_ceiling_keeps_overlap_when_slanted_piece_is_not_above(
    tmp_path, monkeypatch
):
    sidecar = _write_v2_sidecar(
        tmp_path,
        [_final_piece("lower-piece", corners=_rect(0.0, 0.9, 0.0, 1.0, 2.0))],
    )
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    building = {
        "rooms": [
            {
                "ceiling_type": "flat",
                "ceiling_polygon": _rect(0.0, 1.0, 0.0, 2.0, 2.0),
                "raw_ceiling_planes": [{"corners": _rect(0.0, 1.0, 0.0, 2.0, 2.0)}],
            }
        ]
    }

    pieces = vs._raw_ceiling_fallback_for_uuid(UUID, building)

    assert round(_xz_area(pieces), 6) == 4.0


def test_flat_raw_ceiling_subtracts_fresh_backend_arrangement(tmp_path, monkeypatch):
    sidecar = _write_v2_sidecar(tmp_path, [])
    os.utime(sidecar, (1000.0, 1000.0))
    _patch_v2_sidecar(monkeypatch, sidecar)
    _patch_roof_results_path(monkeypatch, tmp_path, mtime=2000.0)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    monkeypatch.setitem(
        vs.ROOF_RESULTS_CACHE,
        UUID,
        {
            "roof_surfaces": {
                "oblique_split": [
                    {
                        "corners": _rect(0.0, 0.9, 0.0, 1.0, 2.0),
                        "arrangement_cell_id": "cell-backend",
                        "roof_hypothesis_id": "roof-hypothesis:oblique:backend",
                    }
                ]
            }
        },
    )
    building = {
        "rooms": [
            {
                "ceiling_type": "flat",
                "ceiling_polygon": _rect(0.0, 1.0, 0.0, 2.0, 2.0),
                "raw_ceiling_planes": [{"corners": _rect(0.0, 1.0, 0.0, 2.0, 2.0)}],
            }
        ]
    }

    pieces = vs._raw_ceiling_fallback_for_uuid(UUID, building)

    assert round(_xz_area(pieces), 6) == 2.0


def test_flat_room_wall_top_fallback_is_clipped_when_no_raw_patch(
    tmp_path, monkeypatch
):
    sidecar = _write_v2_sidecar(
        tmp_path,
        [_final_piece("slanted-above", corners=_rect(0.0, 2.0, 0.0, 1.0, 2.0))],
    )
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    building = {
        "rooms": [
            {
                "ceiling_type": "flat",
                "ceiling_polygon": _rect(0.0, 1.0, 0.0, 2.0, 2.0),
                "raw_ceiling_planes": [],
            }
        ]
    }

    pieces = vs._raw_ceiling_fallback_for_uuid(UUID, building)

    assert {p["source"] for p in pieces} == {"wall_top_flat_ceiling"}
    assert round(_xz_area(pieces), 6) == 2.0


def test_flat_partition_patch_replaces_whole_room_fallback(monkeypatch):
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    monkeypatch.setitem(
        vs.ROOF_RESULTS_CACHE,
        UUID,
        {
            "ceiling_partitions": {
                "flat": [
                    {
                        "id": "ceiling-partition:flat-cap",
                        "room_index": 0,
                        "flat_role": "ceiling_cap",
                        "poly": _rect(0.0, 1.0, 0.0, 1.0, 1.0),
                        "holes": [],
                    }
                ]
            }
        },
    )
    building = {
        "rooms": [
            {
                "ceiling_type": "flat",
                "ceiling_polygon": _rect(0.0, 1.0, 0.0, 2.0, 2.0),
                "raw_ceiling_planes": [],
            }
        ]
    }

    pieces = vs._raw_ceiling_fallback_for_uuid(UUID, building)

    assert {p["source"] for p in pieces} == {"ceiling_partition_flat"}
    assert pieces[0]["partition_id"] == "ceiling-partition:flat-cap"
    assert round(_xz_area(pieces), 6) == 1.0


def test_fully_covered_flat_raw_ceiling_emits_no_fallback(tmp_path, monkeypatch):
    sidecar = _write_v2_sidecar(
        tmp_path,
        [_final_piece("slanted-above", corners=_rect(0.0, 2.0, 0.0, 2.0, 2.0))],
    )
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    building = {
        "rooms": [
            {
                "ceiling_type": "flat",
                "ceiling_polygon": _rect(0.0, 1.0, 0.0, 2.0, 2.0),
                "raw_ceiling_planes": [{"corners": _rect(0.0, 1.0, 0.0, 2.0, 2.0)}],
            }
        ]
    }

    assert vs._raw_ceiling_fallback_for_uuid(UUID, building) == []


def test_slanted_pieces_apply_dormer_cutout_and_preserve_metadata(
    tmp_path, monkeypatch
):
    piece = _final_piece("with-dormer-hole", corners=_square(0.0, 1.0, 0.0, 2.0))
    sidecar = _write_v2_sidecar(tmp_path, [piece])
    _patch_v2_sidecar(monkeypatch, sidecar)
    cutout_xz = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
    monkeypatch.setattr(
        vs,
        "_dormer_cutouts_for_uuid",
        lambda uuid: [(cutout_xz, (0.0, 1.0, 0.0, 1.0))],
    )

    pieces = vs._slanted_pieces_for_uuid(UUID)

    assert len(pieces) == 1
    assert pieces[0]["piece_id"] == "with-dormer-hole"
    assert pieces[0]["source"] == "v2_sidecar"
    assert len(pieces[0]["holes"]) == 1
    assert len(pieces[0]["holes"][0]) == 4


# ---------------------------------------------------------------------------
# Envelope-clip regressions: clip slanted pieces to the slabs of the rooms
# whose oblique walls seeded the V3 slanted_roofs cluster, when the building
# is not a gable and has at least one room-level flat_ceiling. See
# `_slant_envelope_clip_active` in viewer_server.py.
# ---------------------------------------------------------------------------


def _patch_envelope_inputs(
    monkeypatch,
    *,
    flat_ceilings,
    slanted_roofs,
    slabs,
    building=None,
    roof=None,
):
    """Stage V3_CACHE + BUILDINGS_3D_CACHE + ROOF_RESULTS_CACHE for one UUID."""
    monkeypatch.setattr(vs, "_ensure_v3_cache", lambda: None)
    monkeypatch.setitem(
        vs.V3_CACHE,
        UUID,
        {
            "flat_ceilings": flat_ceilings,
            "slanted_roofs": slanted_roofs,
            "slabs": slabs,
        },
    )
    monkeypatch.setitem(
        vs.BUILDINGS_3D_CACHE,
        UUID,
        building or {"rooms": [{"story": 0}]},
    )
    monkeypatch.setitem(vs.ROOF_RESULTS_CACHE, UUID, roof or {})


def _slab(room_id, footprint_xz):
    """A V3 slab record. footprint_xz is a list of [x, y, z]."""
    return {"room_id": room_id, "polygon": footprint_xz}


def _flat_ceiling(over, room_id=None, footprint_xz=None):
    return {
        "over": over,
        "room_id": room_id,
        "footprint_xz": footprint_xz or [],
        "y": 1.0,
    }


def _v3_slanted_roof(room_ids):
    return {
        "id": "v3-slanted-roof::cluster-0",
        "trace": {"inputs": {"room_ids": list(room_ids)}},
    }


def test_envelope_clip_fires_when_slant_extends_past_source_room(tmp_path, monkeypatch):
    """Non-gable + room flat_ceiling + source-room slab -> piece clipped."""
    # Slant sweeps a 6 m x 1 m strip; only a 1 m x 1 m source-room slab below.
    sidecar = _write_v2_sidecar(
        tmp_path,
        [_final_piece("over-extended", corners=_rect(0.0, 1.0, 0.0, 6.0, 1.0))],
    )
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    _patch_envelope_inputs(
        monkeypatch,
        flat_ceilings=[
            _flat_ceiling(
                "room",
                room_id="room:5",
                footprint_xz=[
                    [3.0, 1.0, 0.0],
                    [6.0, 1.0, 0.0],
                    [6.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                ],
            )
        ],
        slanted_roofs=[_v3_slanted_roof(["room:0"])],
        slabs=[
            _slab(
                "room:0",
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            )
        ],
    )

    pieces = vs._slanted_pieces_for_uuid(UUID)

    assert len(pieces) == 1
    assert pieces[0]["clip_source"] == "slanted_wall_envelope"
    # Source-room slab is 1 m^2, dilated by EAVE_OVERHANG_M (1.5 m).
    # The buffered envelope captures the full piece in z but clips x to ~2.5 m.
    area = _xz_area(pieces)
    assert area < 6.0  # was 6 m^2 unclipped
    assert area > 1.0  # at least covers the source slab footprint


def test_envelope_clip_stays_off_for_gable(tmp_path, monkeypatch):
    """A real gable (two opposite-azimuth oblique surfaces) bypasses the clip."""
    sidecar = _write_v2_sidecar(
        tmp_path,
        [_final_piece("gable-piece", corners=_rect(0.0, 1.0, 0.0, 6.0, 1.0))],
    )
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    # Gable: two oblique surfaces, ~180° apart, similar inclination, equal area.
    gable_roof = {
        "roof_surfaces": {
            "oblique": [
                {
                    "cluster": {"avgAzimuth": 0.0, "avgIncl": 30.0},
                    "corners": _rect(0.0, 1.0, 0.0, 4.0, 4.0),
                },
                {
                    "cluster": {"avgAzimuth": 180.0, "avgIncl": 30.0},
                    "corners": _rect(0.0, 1.0, 4.0, 4.0, 4.0),
                },
            ]
        }
    }
    _patch_envelope_inputs(
        monkeypatch,
        flat_ceilings=[
            _flat_ceiling(
                "room",
                room_id="room:0",
                footprint_xz=[
                    [0.0, 1.0, 0.0],
                    [6.0, 1.0, 0.0],
                    [6.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
            )
        ],
        slanted_roofs=[_v3_slanted_roof(["room:0"])],
        slabs=[
            _slab(
                "room:0",
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            )
        ],
        roof=gable_roof,
    )

    pieces = vs._slanted_pieces_for_uuid(UUID)

    assert len(pieces) == 1
    assert "clip_source" not in pieces[0]
    assert round(_xz_area(pieces), 4) == 6.0


def test_envelope_clip_stays_off_without_room_flat_ceiling(tmp_path, monkeypatch):
    """If V3 has no room-level flat_ceiling, the clip never fires."""
    sidecar = _write_v2_sidecar(
        tmp_path,
        [_final_piece("no-room-fc", corners=_rect(0.0, 1.0, 0.0, 6.0, 1.0))],
    )
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    _patch_envelope_inputs(
        monkeypatch,
        flat_ceilings=[
            # Only gap-synthesised closure ceilings — not scan-derived flat rooms.
            _flat_ceiling("gap", room_id=None, footprint_xz=[]),
        ],
        slanted_roofs=[_v3_slanted_roof(["room:0"])],
        slabs=[
            _slab(
                "room:0",
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            )
        ],
    )

    pieces = vs._slanted_pieces_for_uuid(UUID)

    assert len(pieces) == 1
    assert "clip_source" not in pieces[0]
    assert round(_xz_area(pieces), 4) == 6.0


def test_envelope_clip_stays_off_without_source_rooms(tmp_path, monkeypatch):
    """If V3 slanted_roofs has no source room_ids, the envelope is empty -> no clip."""
    sidecar = _write_v2_sidecar(
        tmp_path,
        [_final_piece("no-source-rooms", corners=_rect(0.0, 1.0, 0.0, 6.0, 1.0))],
    )
    _patch_v2_sidecar(monkeypatch, sidecar)
    monkeypatch.setattr(vs, "_dormer_cutouts_for_uuid", lambda uuid: [])
    _patch_envelope_inputs(
        monkeypatch,
        flat_ceilings=[
            _flat_ceiling(
                "room",
                room_id="room:5",
                footprint_xz=[
                    [3.0, 1.0, 0.0],
                    [6.0, 1.0, 0.0],
                    [6.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                ],
            )
        ],
        slanted_roofs=[],  # no V3 cluster at all
        slabs=[],
    )

    pieces = vs._slanted_pieces_for_uuid(UUID)

    assert len(pieces) == 1
    assert "clip_source" not in pieces[0]
    assert round(_xz_area(pieces), 4) == 6.0


def test_envelope_clip_combines_with_dormer_cutout(tmp_path, monkeypatch):
    """When the gate fires, dormer cutouts still apply on top of the envelope clip."""
    sidecar = _write_v2_sidecar(
        tmp_path,
        [_final_piece("with-dormer-and-clip", corners=_rect(0.0, 1.0, 0.0, 6.0, 1.0))],
    )
    _patch_v2_sidecar(monkeypatch, sidecar)
    cutout_xz = Polygon([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
    monkeypatch.setattr(
        vs,
        "_dormer_cutouts_for_uuid",
        lambda uuid: [(cutout_xz, (0.0, 1.0, 0.0, 1.0))],
    )
    _patch_envelope_inputs(
        monkeypatch,
        flat_ceilings=[
            _flat_ceiling(
                "room",
                room_id="room:5",
                footprint_xz=[
                    [3.0, 1.0, 0.0],
                    [6.0, 1.0, 0.0],
                    [6.0, 1.0, 1.0],
                    [3.0, 1.0, 1.0],
                ],
            )
        ],
        slanted_roofs=[_v3_slanted_roof(["room:0"])],
        slabs=[
            _slab(
                "room:0",
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
            )
        ],
    )

    pieces = vs._slanted_pieces_for_uuid(UUID)

    assert len(pieces) == 1
    assert pieces[0]["clip_source"] == "slanted_wall_envelope"
    # Dormer cutout (0.04 m^2) becomes a hole inside the clipped piece.
    assert len(pieces[0]["holes"]) == 1
