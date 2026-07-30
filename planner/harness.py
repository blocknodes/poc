"""Harness —— 推理侧编排：驱动共享的 `PlannerAgent` 状态机走完整条链路。

覆盖 4 域的完整流程：
  * vod/educ 检索类：route → IR generate(约束解码) → validate(+自修复) → compile
    - vod_search 意图：compile 后根据字段子集自动选 vod_search 或 vod_search_all
    - vod_relate_search：compile 成 relate 格式（4 字段布尔 DSL）
  * vod_slow_search_data_search：route → 直接结束，只传 query 原文
  * vod_personalized_search / vod_history：route → 独立 slot-fill
  * audio：route → audio slot-fill(约束解码) → compile
  * device：route → device slot-fill(约束解码) → compile

与训练侧 `train/planner_plugin.py::PlannerEnv` **共用同一个 agent 核心**
（`planner.agent.PlannerAgent`）：prompt 构造与状态转移完全一致。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .agent import (
    AUDIO_TOOLS,
    DEVICE_TOOLS,
    IR_TOOLS,
    SIMPLE_TOOLS,
    SLOW_SEARCH_TOOLS,
    PlannerAgent,
)
from .compiler import (
    CompileError,
    compile_audio,
    compile_device,
    compile_with_fallback,
)
from .grammar import (
    build_audio_schema,
    build_device_schema,
    build_ir_schema,
    build_route_schema,
)
from .ir import IRError, parse_ir
from .vllm_client import VLLMClient

__all__ = ["Planner", "PlanResult", "IR_TOOLS"]


@dataclass
class PlanResult:
    tool_name: str
    parameters: dict[str, Any]
    domain: str
    intent: Optional[str] = None
    ir: Optional[dict] = None
    slot_fill: Optional[dict] = None         # audio / device 的 slot-fill 结果
    fallback_from: Optional[str] = None      # 若发生工具切换
    repairs: int = 0                         # IR 自修复次数
    route_confidence: Optional[float] = None
    notes: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)  # LLM 调用轨迹


class Planner:
    def __init__(self, client: VLLMClient, max_repairs: int = 2, vod_only: bool = False,
                 use_eb_prompt: bool = True):
        self.client = client
        self.max_repairs = max_repairs
        self.vod_only = vod_only
        self.use_eb_prompt = use_eb_prompt

    def plan(self, query: str, memory_hint: str = "") -> PlanResult:
        agent = PlannerAgent(query, memory_hint=memory_hint, max_repairs=self.max_repairs,
                             vod_only=self.vod_only, use_eb_prompt=self.use_eb_prompt)

        trace: list[dict] = []  # 记录每次 LLM 调用

        # ---- 轨迹起点：system + 首个 observation ----
        messages = [
            {"role": "system", "content": agent.system_prompt()},
            {"role": "user", "content": agent.first_observation()},
        ]

        # ---- 阶段1：路由（约束解码）----
        route_schema = build_route_schema(vod_only=self.vod_only)
        route_raw = self.client.complete_json(messages, guided_json=route_schema)
        messages.append({"role": "assistant", "content": json.dumps(route_raw, ensure_ascii=False)})
        trace.append({
            "stage": "route",
            "messages": [m.copy() for m in messages],
            "guided_json": route_schema,
            "output": route_raw,
        })
        step = agent.observe(route_raw)

        # ---- 分支处理 ----

        # A) 慢链路：路由直接结束，只传 query
        if step.done and step.info.get("reason") == "slow_search_passthrough":
            return PlanResult(
                tool_name=agent.routed_tool or "vod_slow_search_data_search",
                parameters={"query": query},
                domain=agent.domain or "vod",
                intent=agent.intent,
                route_confidence=agent.confidence,
                notes=["慢链路只传 query 原文，服务端自行语义解析"],
                trace=trace,
            )

        # B) 简单工具：route_end，独立处理参数
        if step.done and step.phase == "route_end":
            result = self._handle_simple(agent)
            result.trace = trace
            return result

        # C) Audio slot-fill
        if step.phase == "audio":
            return self._handle_audio(agent, messages, step, trace)

        # D) Device slot-fill
        if step.phase == "device":
            return self._handle_device(agent, messages, step, trace)

        # E) IR 生成 + 校验自修复
        if step.phase == "ir":
            return self._handle_ir(agent, messages, step, query, trace)

        # F) 兜底
        return PlanResult(
            tool_name=agent.routed_tool or "unknown",
            parameters={},
            domain=agent.domain or "unknown",
            intent=agent.intent,
            route_confidence=agent.confidence,
            notes=["未知路由结果，无法进一步处理"],
            trace=trace,
        )

    def _handle_simple(self, agent: PlannerAgent) -> PlanResult:
        """简单工具（personalized/history 以及非 IR 的 route_end）。"""
        notes = []
        tool = agent.routed_tool or "unknown"
        query = agent.query

        params: dict[str, Any] = {}

        # personalized / history：从 query 中提取 category
        if tool in ("vod_personalized_search", "vod_history"):
            category = _extract_category(query)
            if category:
                params["category"] = category
            notes.append(f"工具 '{tool}' 简单 slot-fill（提取 category）")
        elif tool in SIMPLE_TOOLS:
            notes.append(f"工具 '{tool}' 走独立 schema（参数留空待扩展）")
        else:
            notes.append(f"工具 '{tool}' 未由 IR 编译器处理")

        return PlanResult(
            tool_name=tool,
            parameters=params,
            domain=agent.domain or "unknown",
            intent=agent.intent,
            route_confidence=agent.confidence,
            notes=notes,
        )

    def _handle_audio(self, agent: PlannerAgent, messages: list, step,
                      trace: list[dict]) -> PlanResult:
        """有声域：slot-fill → compile_audio。"""
        messages.append({"role": "user", "content": step.next_observation})
        audio_schema = build_audio_schema()
        slot_raw = self.client.complete_json(messages, guided_json=audio_schema)
        messages.append({"role": "assistant", "content": json.dumps(slot_raw, ensure_ascii=False)})
        trace.append({
            "stage": "audio_slot_fill",
            "messages": [m.copy() for m in messages],
            "guided_json": audio_schema,
            "output": slot_raw,
        })
        step = agent.observe(slot_raw)

        slot_fill = agent.slot_fill or {}
        tool_name, params = compile_audio(slot_fill)

        return PlanResult(
            tool_name=tool_name,
            parameters=params,
            domain="audio",
            intent=agent.intent,
            slot_fill=slot_fill,
            route_confidence=agent.confidence,
            trace=trace,
        )

    def _handle_device(self, agent: PlannerAgent, messages: list, step,
                       trace: list[dict]) -> PlanResult:
        """设备域：slot-fill → compile_device。"""
        messages.append({"role": "user", "content": step.next_observation})
        device_schema = build_device_schema()
        slot_raw = self.client.complete_json(messages, guided_json=device_schema)
        messages.append({"role": "assistant", "content": json.dumps(slot_raw, ensure_ascii=False)})
        trace.append({
            "stage": "device_slot_fill",
            "messages": [m.copy() for m in messages],
            "guided_json": device_schema,
            "output": slot_raw,
        })
        step = agent.observe(slot_raw)

        slot_fill = agent.slot_fill or {}
        tool_name, params = compile_device(slot_fill)

        return PlanResult(
            tool_name=tool_name,
            parameters=params,
            domain="device",
            intent=agent.intent,
            slot_fill=slot_fill,
            route_confidence=agent.confidence,
            trace=trace,
        )

    def _handle_ir(self, agent: PlannerAgent, messages: list, step, query: str,
                   trace: list[dict]) -> PlanResult:
        """IR 生成 + 校验自修复 + 编译。"""
        ir_schema = build_ir_schema(agent.domain)
        repair_round = 0
        while not step.done:
            messages.append({"role": "user", "content": step.next_observation})
            ir_raw = self.client.complete_json(messages, guided_json=ir_schema)
            messages.append({"role": "assistant", "content": json.dumps(ir_raw, ensure_ascii=False)})
            stage_name = "ir_generate" if repair_round == 0 else f"ir_repair_{repair_round}"
            trace.append({
                "stage": stage_name,
                "messages": [m.copy() for m in messages],
                "guided_json": ir_schema,
                "output": ir_raw,
                "valid": None,  # 暂存，下面更新
                "errors": None,
            })
            step = agent.observe(ir_raw)
            # 回填校验结果
            trace[-1]["valid"] = agent.ir_valid
            trace[-1]["errors"] = agent.errs if not agent.ir_valid else []
            repair_round += 1

        if not agent.ir_valid:
            raise IRError("IR 校验在最大重试后仍失败: " + "; ".join(agent.errs))

        # 编译
        ir = parse_ir(agent.final_ir)
        try:
            actual_tool, params = compile_with_fallback(
                ir, agent.routed_tool, retext=query, intent=agent.intent or ""
            )
        except CompileError as e:
            raise CompileError(f"IR 编译失败: {e}") from e

        # 记录编译结果到 trace
        trace.append({
            "stage": "compile",
            "input_ir": agent.final_ir,
            "routed_tool": agent.routed_tool,
            "actual_tool": actual_tool,
            "compiled_params": params,
        })

        fallback_from = agent.routed_tool if actual_tool != agent.routed_tool else None
        notes = []
        if fallback_from:
            notes.append(f"工具选择变化: {fallback_from} -> {actual_tool}")

        return PlanResult(
            tool_name=actual_tool,
            parameters=params,
            domain=agent.domain,
            intent=agent.intent,
            ir=agent.final_ir,
            fallback_from=fallback_from,
            repairs=agent.repairs,
            route_confidence=agent.confidence,
            notes=notes,
            trace=trace,
        )


# ===========================================================================
# 辅助：从 query 中提取 category（规则 + 关键词匹配）
# ===========================================================================
_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    # 精确匹配优先（长的在前）
    ("电视剧", "电视剧"),
    ("连续剧", "电视剧"),
    ("纪录片", "纪录片"),
    ("综艺", "综艺"),
    ("动漫", "动漫"),
    ("动画片", "动漫"),
    ("戏曲", "戏曲"),
    ("短视频", "短视频"),
    ("电影", "电影"),
    ("影片", "电影"),
    ("片", "电影"),
    ("剧", "电视剧"),
]


def _extract_category(query: str) -> Optional[str]:
    """从用户 query 中提取影视分类。简单关键词匹配。"""
    # 按长度降序匹配，避免"电视剧"被"剧"提前匹配
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in query:
            return category
    return None
