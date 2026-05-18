from __future__ import annotations


def make_locator_id(uuid: str, scope: str, *parts: object) -> str:
    if not uuid:
        raise ValueError("uuid must be non-empty")
    if not scope:
        raise ValueError("scope must be non-empty")
    if not parts:
        raise ValueError("at least one locator part is required")
    encoded_parts: list[str] = []
    for part in parts:
        text = str(part)
        if not text:
            raise ValueError("locator part must be non-empty")
        if "::" in text:
            raise ValueError("locator part cannot contain '::'")
        encoded_parts.append(text)
    clean_scope = scope[5:] if scope.startswith("tier-") else scope
    return f"{uuid}::tier-{clean_scope}::{':'.join(encoded_parts)}"
