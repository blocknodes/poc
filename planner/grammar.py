"""Grammar —— 从 Field Registry 自动生成「按 domain 收紧的 IR JSON Schema」，
用于 vLLM 的约束解码 (guided_json / guided decoding)。

约束解码保证：
  * 输出一定是合法 JSON；
  * field 只能取该 domain 的合法枚举（消灭字段名幻觉）；
  * 精确/状态/范围三类叶子的结构（value/values+op/range、0|1、from/to）被强制区分；
  * operator / order 只能取 and|or / asc|desc；
  * 布尔节点可任意递归嵌套（$ref 自引用）。

这样模型只需产出「域无关 IR」，剩余的语义正确性交给 ir.validate_ir + 编译器。
"""
from __future__ import annotations

from typing import Any

from .registry import (
    Kind,
    PLAYBACK_FIELDS,
    fields_for_domain,
    field_names,
    sort_keys_for_domain,
)


def build_ir_schema(domain: str) -> dict[str, Any]:
    """生成某 domain 的 IR JSON Schema（draft-07，含 $defs 递归）。"""
    exact = field_names(domain, Kind.EXACT)
    status = field_names(domain, Kind.STATUS)
    range_fields = field_names(domain, Kind.RANGE)
    sort_keys = sort_keys_for_domain(domain)

    defs: dict[str, Any] = {
        "Node": {
            "oneOf": [
                {"$ref": "#/$defs/And"},
                {"$ref": "#/$defs/Or"},
                {"$ref": "#/$defs/Not"},
                {"$ref": "#/$defs/Leaf"},
            ]
        },
        "And": {
            "type": "object",
            "properties": {
                "and": {"type": "array", "items": {"$ref": "#/$defs/Node"}, "minItems": 1}
            },
            "required": ["and"],
            "additionalProperties": False,
        },
        "Or": {
            "type": "object",
            "properties": {
                "or": {"type": "array", "items": {"$ref": "#/$defs/Node"}, "minItems": 1}
            },
            "required": ["or"],
            "additionalProperties": False,
        },
        "Not": {
            "type": "object",
            "properties": {"not": {"$ref": "#/$defs/Node"}},
            "required": ["not"],
            "additionalProperties": False,
        },
        "Leaf": {"oneOf": _leaf_variants(exact, status, range_fields)},
    }

    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": f"{domain}_planner_ir",
        "type": "object",
        "properties": {
            "domain": {"const": domain},
            "action": {"type": "string", "enum": ["search", "play"]},
            "query": {"$ref": "#/$defs/Node"},
        },
        "required": ["domain", "query"],
        "additionalProperties": False,
        "$defs": defs,
    }

    if sort_keys:
        schema["properties"]["sort"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "enum": sort_keys},
                    "order": {"type": "string", "enum": ["asc", "desc"]},
                },
                "required": ["key", "order"],
                "additionalProperties": False,
            },
        }

    if domain == "vod":
        schema["properties"]["playback"] = {
            "type": "object",
            "properties": {k: {"type": "integer", "minimum": 0} for k in PLAYBACK_FIELDS},
            "additionalProperties": False,
        }

    return schema


def _leaf_variants(exact: list[str], status: list[str], range_fields: list[str]) -> list[dict]:
    variants: list[dict] = []

    if exact:
        # 精确单值
        variants.append({
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": exact},
                "value": {"type": "string"},
            },
            "required": ["field", "value"],
            "additionalProperties": False,
        })
        # 精确多值 + op
        variants.append({
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": exact},
                "values": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "op": {"type": "string", "enum": ["and", "or"]},
            },
            "required": ["field", "values"],
            "additionalProperties": False,
        })

    if status:
        variants.append({
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": status},
                "value": {"type": "integer", "enum": [0, 1]},
            },
            "required": ["field", "value"],
            "additionalProperties": False,
        })

    if range_fields:
        variants.append({
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": range_fields},
                "range": {
                    "type": "object",
                    "properties": {
                        "from": {"type": ["string", "number"]},
                        "to": {"type": ["string", "number"]},
                    },
                    "required": ["from", "to"],
                    "additionalProperties": False,
                },
            },
            "required": ["field", "range"],
            "additionalProperties": False,
        })

    return variants


# 路由阶段的小 schema：把工具选择也约束成枚举
ROUTE_TOOLS = {
    "vod": [
        "vod_search",
        "vod_slow_search_data_search",
        "vod_relate_recommend",
        "vod_personalized_recommend",
        "vod_history",
        "vod_clip_search",
    ],
    "educ": [
        "educ_search",
        "educ_slow_search_data_search",
        "educ_relate_recommend",
        "educ_history",
    ],
}


def build_route_schema() -> dict[str, Any]:
    all_tools = sorted({t for tools in ROUTE_TOOLS.values() for t in tools})
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "planner_route",
        "type": "object",
        "properties": {
            "domain": {"type": "string", "enum": ["vod", "educ"]},
            "intent": {
                "type": "string",
                "enum": ["search", "slow_search", "relate", "personalized", "history", "clip"],
            },
            "tool": {"type": "string", "enum": all_tools},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["domain", "tool", "confidence"],
        "additionalProperties": False,
    }
