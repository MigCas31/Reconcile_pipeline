from __future__ import annotations

import dataclasses
import json
from enum import StrEnum
from types import UnionType
from typing import Any, Literal, get_args, get_origin, get_type_hints


def _is_optional(field_type: Any) -> bool:
    return get_origin(field_type) in (UnionType, __import__("typing").Union) and type(
        None
    ) in get_args(field_type)


def _schema_for(field_type: Any) -> dict[str, Any]:
    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin is Literal:
        return {"enum": list(args)}
    if _is_optional(field_type):
        non_none = next(arg for arg in args if arg is not type(None))
        return {"oneOf": [_schema_for(non_none), {"type": "null"}]}
    if origin is list:
        (item_type,) = args
        return {"type": "array", "items": _schema_for(item_type)}
    if origin is dict:
        _key_type, val_type = args
        return {
            "type": "object",
            "additionalProperties": _schema_for(val_type),
        }
    if isinstance(field_type, type) and issubclass(field_type, StrEnum):
        return {"type": "string", "enum": [item.value for item in field_type]}
    if dataclasses.is_dataclass(field_type):
        return emit_schema(field_type)
    if field_type is str:
        return {"type": "string"}
    if field_type is int:
        return {"type": "integer"}
    if field_type is float:
        return {"type": "number"}
    if field_type is bool:
        return {"type": "boolean"}
    raise TypeError(f"unsupported schema field type: {field_type!r}")


def emit_schema(root_type: type) -> dict[str, Any]:
    if not dataclasses.is_dataclass(root_type):
        raise TypeError(f"expected dataclass type, got {root_type!r}")
    hints = get_type_hints(root_type)
    properties = {
        field.name: _schema_for(hints[field.name])
        for field in dataclasses.fields(root_type)
    }
    required = [
        field.name
        for field in dataclasses.fields(root_type)
        if field.metadata.get("schema_required", True)
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": root_type.__name__,
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def schema_json(root_type: type) -> str:
    return json.dumps(emit_schema(root_type), indent=2, sort_keys=True) + "\n"
