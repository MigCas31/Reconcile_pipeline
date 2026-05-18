"""Post-tag ceiling cleanup for the tier payload.

`_polish_payload_ceilings` runs the four-step polish chain after
`tag_payload`: drop redundant split overlaps + clean rings, clip coplanar
overlaps against rooms, merge coplanar pieces (then re-clean), and
optionally clip pieces to per-story footprints when
`TIER_PER_STORY_CLIP=1`.
"""

from __future__ import annotations

import os
from dataclasses import replace

from reconcile_tiers.build_internals.envelope_pieces import (
    _clean_ceiling_piece_geometries,
    _drop_redundant_ceiling_split_overlaps,
)
from reconcile_tiers.payload.schema import TierPayload, payload_to_dict


def _polish_payload_ceilings(payload: TierPayload) -> TierPayload:
    """Run the post-tag ceiling cleanup chain on a tagged payload.

    The intermediate `clip_coplanar_overlaps` pass is *not* wrapped in
    `_clean_ceiling_piece_geometries` because its sub-cm cut strips would
    be collapsed by the 2 cm edge cleaner.
    """
    from reconcile_tiers.assemble.coplanar_merge import (
        clip_coplanar_overlaps,
        merge_coplanar_ceilings,
    )

    payload = replace(
        payload,
        ceiling=_clean_ceiling_piece_geometries(
            _drop_redundant_ceiling_split_overlaps(payload.ceiling)
        ),
    )
    payload = replace(
        payload,
        ceiling=clip_coplanar_overlaps(payload.ceiling, payload.rooms),
    )
    payload = replace(
        payload,
        ceiling=_clean_ceiling_piece_geometries(
            merge_coplanar_ceilings(payload.ceiling, building_uuid=payload.uuid)
        ),
    )
    if os.environ.get("TIER_PER_STORY_CLIP") == "1":
        from reconcile_tiers.assemble.per_story_clip import clip_pieces_to_per_story

        payload = replace(
            payload,
            ceiling=_clean_ceiling_piece_geometries(
                clip_pieces_to_per_story(
                    payload.ceiling,
                    payload_to_dict(payload).get("rooms") or [],
                    building_uuid=payload.uuid,
                )
            ),
        )
    return payload
