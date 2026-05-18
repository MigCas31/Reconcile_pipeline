"""Shared utility functions for roof graph modules."""

import hashlib


def stable_hash(parts: list[str], length: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:length]


def room_key(room_index: int) -> str:
    return f"room:{room_index}"
