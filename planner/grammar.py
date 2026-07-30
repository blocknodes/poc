"""Grammar —— 从 Field Registry 自动生成各类约束解码 JSON Schema。

生成三大类 schema：
  1. IR schema（vod/educ 检索类工具 → 约束解码产 IR）
  2. Audio slot-fill schema（有声域 → 约束解码产 slot）
  3. Device slot-fill schema（设备域 → 约束解码产 slot）
  4. 路由 schema（一级路由 → 约束解码选 domain/tool）

约束解码保证：
  * 输出一定是合法 JSON；
  * field 只能取该 domain 的合法枚举（消灭字段名幻觉）；
  * 精确/状态/范围三类叶子的结构被强制区分；
  * operator / order 只能取 and|or / asc|desc；
  * 布尔节点可任意递归嵌套（$ref 自引用）。

0725-v1 变更：
  * vod_search 和 vod_search_all 统一意图，路由都指向 vod_search，IR 后再选
  * 新增 vod_relate_search 路由
  * vod_slow_search 只传 query（路由即结束）
  * playback 字段更新：video_index
"""
from __future__ import annotations

from typing import Any

from .registry import (
    AUDIO_PLAY_MODES,
    AUDIO_SCREEN_MODES,
    AUDIO_TOOLS,
    DEVICE_TOOLS,
    Kind,
    PLAYBACK_FIELDS,
    fields_for_domain,
    field_names,
    sort_keys_for_domain,
)


# ===========================================================================
# 1. IR Schema（影视 / 少儿检索类工具）
# ===========================================================================

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


# ===========================================================================
# 2. Audio Slot-Fill Schema（有声域）
# ===========================================================================

def build_audio_schema() -> dict[str, Any]:
    """有声域的 slot-fill schema —— 模型只需填 query + play_mode。"""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "audio_slot_fill",
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": AUDIO_TOOLS,
                "description": "有声工具名。audio_search: 搜索/播放有声内容；audio_chat_qa: 有声内容问答。",
            },
            "query": {
                "type": "string",
                "description": "用户查询文本（归一化后），如「郭德纲的相声」「三国演义评书」。",
            },
            "play_mode": {
                "type": "string",
                "enum": AUDIO_PLAY_MODES,
                "description": "播放模式。search: 搜索列表展示；play: 直接起播；screen_off_play: 熄屏播放。",
            },
            "screen_mode": {
                "type": "string",
                "enum": AUDIO_SCREEN_MODES,
                "description": "屏幕状态。normal: 亮屏；screen_standby: 熄屏待机。",
            },
        },
        "required": ["tool", "query", "play_mode"],
        "additionalProperties": False,
    }


# ===========================================================================
# 3. Device Slot-Fill Schema（设备控制域）
# ===========================================================================

def build_device_schema() -> dict[str, Any]:
    """设备控制域的 slot-fill schema —— 模型选工具 + 填 operation/object/value。"""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "device_slot_fill",
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": DEVICE_TOOLS,
                "description": "设备控制工具名（20 选 1）。",
            },
            "operation": {
                "type": "string",
                "description": "操作类型，如：提高、降低、打开、关闭、设置、查询、切换。",
            },
            "object": {
                "type": "string",
                "description": "控制对象，如：音量、静音、亮度、WiFi、蓝牙、屏幕、电源等。",
            },
            "value": {
                "type": "string",
                "description": "参数值（可选），如数字（30）、百分比（50%）、模式名等。",
            },
        },
        "required": ["tool", "operation", "object"],
        "additionalProperties": False,
    }


# ===========================================================================
# 4. 路由 Schema —— 覆盖全部 4 域
# ===========================================================================

# 按 domain 列举所有可路由到的工具
# 0725-v1: vod_search 代表 search/search_all 统一意图；
#           vod_slow_search_data_search 代表慢链路（只传 query）；
#           vod_relate_search 代表相关推荐（布尔 DSL）
ROUTE_TOOLS = {
    "vod": [
        "vod_search",                    # 精确检索（含 search_all，生成 IR 后再选）
        "vod_slow_search_data_search",   # 慢链路语义搜索（只传 query）
        "vod_relate_search",             # 相关推荐（布尔 DSL，4 字段）
        "vod_personalized_search",       # 个性化推荐（只传 category）
        "vod_history",                   # 历史记录（category + time）
    ],
    "educ": [
        "educ_search",
        "educ_slow_search_data_search",
        "educ_relate_recommend",
        "educ_history",
    ],
    "audio": AUDIO_TOOLS,
    "device": DEVICE_TOOLS,
}

# 全部 intent 枚举（跨域）
ALL_INTENTS = [
    # 影视/少儿共有
    "search", "slow_search", "relate", "personalized", "history",
    "play",  # 明确起播（播放+具体片名/集数）
    # 有声
    "audio_search", "audio_play", "audio_screen_off_play", "audio_chat_qa",
    # 设备
    "device_control",
]


def build_route_schema(vod_only: bool = False) -> dict[str, Any]:
    """路由 schema。vod_only=True 时只包含影视域工具。"""
    if vod_only:
        all_tools = sorted(ROUTE_TOOLS["vod"])
        all_domains = ["vod"]
        intents = ["search", "slow_search", "relate", "personalized", "history", "play"]
    else:
        all_tools = sorted({t for tools in ROUTE_TOOLS.values() for t in tools})
        all_domains = sorted(ROUTE_TOOLS.keys())
        intents = ALL_INTENTS
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "planner_route",
        "type": "object",
        "properties": {
            "domain": {"type": "string", "enum": all_domains},
            "intent": {"type": "string", "enum": intents},
            "tool": {"type": "string", "enum": all_tools},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["domain", "intent", "tool", "confidence"],
        "additionalProperties": False,
    }
