"""schemas.py —— 各阶段的 guided_json schema。"""
from __future__ import annotations


def route_schema(available_tools: list[str] | None = None) -> dict:
    """路由 schema。当传入 available_tools 时，tool 字段约束为枚举。"""
    tool_prop: dict = {"type": "string"}
    if available_tools:
        tool_prop["enum"] = available_tools

    return {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "enum": ["vod", "educ", "audio", "device"]},
            "intent": {"type": "string"},
            "tool": tool_prop,
            "confidence": {"type": "number"},
        },
        "required": ["domain", "tool", "confidence"],
    }


def ir_schema(domain: str = "vod") -> dict:
    """IR schema。"""
    return {
        "type": "object",
        "properties": {
            "domain": {"type": "string"},
            "action": {"type": "string", "enum": ["search", "play"]},
            "query": {"type": "object"},
            "sort": {"type": "array"},
            "playback": {"type": "object"},
        },
        "required": ["action", "query"],
    }


def audio_schema() -> dict:
    """Audio slot-fill schema。"""
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string"},
            "query": {"type": "string"},
            "play_mode": {"type": "string"},
            "screen_mode": {"type": "string"},
        },
        "required": ["query", "play_mode"],
    }


def device_schema(available_tools: list[str] | None = None) -> dict:
    """Device slot-fill schema。当传入 available_tools 时，tool 字段约束为枚举。"""
    tool_prop: dict = {"type": "string"}
    if available_tools:
        tool_prop["enum"] = available_tools

    return {
        "type": "object",
        "properties": {
            "tool": tool_prop,
            "operation": {"type": "string"},
            "object": {"type": "string"},
            "value": {"type": "string"},
            "date_time": {"type": "string"},
        },
        "required": ["tool", "operation", "object"],
    }


def intent_split_schema() -> dict:
    """意图拆分 schema。"""
    return {
        "type": "object",
        "properties": {
            "multi": {"type": "boolean"},
            "sub_queries": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["multi", "sub_queries"],
    }
