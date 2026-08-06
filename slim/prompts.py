"""Prompts —— 各阶段的 system/user prompt 模板。"""
from __future__ import annotations


def route_system_prompt(available_tools: list[str] | None = None) -> str:
    base = """你是一个智能电视助手的意图路由器。
根据用户请求，判断所属域和应使用的工具。
输出 JSON：{"domain": "...", "intent": "...", "tool": "...", "confidence": 0.0~1.0}

域：
- vod：影视搜索/播放（电影、电视剧、综艺、纪录片等）
- educ：少儿/教育内容
- audio：有声内容（有声书、播客、音乐）
- device：设备控制（音量、亮度、电源、网络、信号源等）"""

    if available_tools:
        tool_list_str = "、".join(available_tools)
        base += f"\n\n【重要】你只能从以下工具中选择，不得输出列表之外的工具名：\n{tool_list_str}"

    return base


def route_user_prompt(query: str, memory_hint: str = "") -> str:
    parts = []
    if memory_hint:
        parts.append(f"对话上下文：\n{memory_hint}\n")
    parts.append(f'用户说："{query}"')
    return "\n".join(parts)


def ir_user_prompt(query: str, domain: str, memory_hint: str = "", intent: str = "") -> str:
    return f"""现在请为以下请求生成结构化 IR（中间表示）。

域：{domain}
意图：{intent or "search"}
用户说："{query}"

输出域无关的布尔 IR JSON，包含 domain、action、query（嵌套布尔结构）、sort（可选）、playback（可选）。"""


def audio_user_prompt(query: str, memory_hint: str = "") -> str:
    parts = []
    if memory_hint:
        parts.append(f"对话上下文：\n{memory_hint}\n")
    parts.append(f"""现在请为以下有声请求填槽。

用户说："{query}"

输出 JSON，包含 tool、query、play_mode、screen_mode 等字段。""")
    return "\n".join(parts)


def device_user_prompt(query: str, memory_hint: str = "") -> str:
    parts = []
    if memory_hint:
        parts.append(f"对话上下文：\n{memory_hint}\n")
    parts.append(f"""现在请为以下设备控制请求填槽。

用户说："{query}"

输出 JSON，包含 tool、operation、object、value（可选）、date_time（可选）字段。""")
    return "\n".join(parts)


def intent_split_system_prompt() -> str:
    return """你是一个意图拆分器。判断用户请求是否包含多个独立意图。
输出 JSON：{"multi": true/false, "sub_queries": ["子请求1", "子请求2", ...]}
如果只有单意图，multi=false，sub_queries 为原始请求。"""


def intent_split_user_prompt(query: str, memory_hint: str = "") -> str:
    parts = []
    if memory_hint:
        parts.append(f"对话上下文：\n{memory_hint}\n")
    parts.append(f'用户说："{query}"')
    return "\n".join(parts)
