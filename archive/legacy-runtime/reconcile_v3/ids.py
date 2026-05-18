"""Element-id construction / parsing for reconcile_v3.

Format: ``<building_uuid>::<kind>::<id>`` where ``kind`` starts with
``v3-``. Compatible with ``reconcile.element_locator.parse_element_id``.
"""

from __future__ import annotations

KINDS = (
    "v3-part",
    "v3-gap",
    "v3-slab",
    "v3-wall-extension",
    "v3-flat-ceiling",
    "v3-slanted-roof",
    "v3-roof-proposal",
    "v3-merged-roof-segment",
    "v3-dormer",
    "v3-unresolved",
    "v3-gable-status",
    "v3-gable-ridge",
    "v3-gable-extension-target",
)


def is_v3_kind(kind: str) -> bool:
    return kind.startswith("v3-")


def make_element_id(building_uuid: str, kind: str, inner: str) -> str:
    if not is_v3_kind(kind):
        raise ValueError(f"Not a v3 kind: {kind}")
    return f"{building_uuid}::{kind}::{inner}"
