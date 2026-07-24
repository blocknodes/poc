"""Harness —— 编排整条链路：

  route  -> ir_generate(约束解码) -> validate_ir(+自修复重试) -> compile(+flat回退) -> step

对应设计里的分阶段 + 校验闭环。非检索类工具(relate/personalized/history/clip)
不走 IR，路由后交给各自独立处理（此处留出扩展点）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .compiler import CompileError, compile_with_fallback
from .grammar import build_ir_schema, build_route_schema
from .ir import IRError, parse_ir, validate_ir
from .prompts import build_ir_messages, build_route_messages
from .vllm_client import VLLMClient

# 由 IR 编译器负责的检索类工具
IR_TOOLS = {
    "vod_search", "vod_slow_search_data_search",
    "educ_search", "educ_slow_search_data_search",
}


@dataclass
class PlanResult:
    tool_name: str
    parameters: dict[str, Any]
    domain: str
    intent: Optional[str] = None
    ir: Optional[dict] = None
    fallback_from: Optional[str] = None      # 若发生 flat->nested 回退
    repairs: int = 0                         # IR 自修复次数
    route_confidence: Optional[float] = None
    notes: list[str] = field(default_factory=list)


class Planner:
    def __init__(self, client: VLLMClient, max_repairs: int = 2):
        self.client = client
        self.max_repairs = max_repairs

    # ---- 阶段1：路由 ----
    def route(self, query: str, memory_hint: str = "") -> dict:
        messages = build_route_messages(query, memory_hint)
        return self.client.complete_json(messages, guided_json=build_route_schema())

    # ---- 阶段2+3：IR 生成 + 校验自修复 ----
    def generate_ir(self, query: str, domain: str, memory_hint: str = "") -> tuple[dict, int]:
        schema = build_ir_schema(domain)
        messages = build_ir_messages(query, domain, memory_hint)
        repairs = 0
        last_errs: list[str] = []
        while True:
            raw = self.client.complete_json(messages, guided_json=schema)
            try:
                ir = parse_ir(raw)
                errs = validate_ir(ir)
            except IRError as e:
                errs = [str(e)]
            if not errs:
                return raw, repairs
            last_errs = errs
            if repairs >= self.max_repairs:
                break
            repairs += 1
            # 把错误回灌给模型自修复
            messages = messages + [
                {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
                {"role": "user", "content": "上面的 IR 有以下问题，请修正后重新只输出 JSON：\n- "
                 + "\n- ".join(last_errs)},
            ]
        raise IRError("IR 校验在最大重试后仍失败: " + "; ".join(last_errs))

    # ---- 端到端 ----
    def plan(self, query: str, memory_hint: str = "") -> PlanResult:
        route = self.route(query, memory_hint)
        domain = route["domain"]
        tool = route["tool"]
        intent = route.get("intent")
        conf = route.get("confidence")

        if tool not in IR_TOOLS:
            # 非检索类：留给独立 slot-filler（本 POC 不实现），直接回传路由结果
            return PlanResult(
                tool_name=tool, parameters={}, domain=domain, intent=intent,
                route_confidence=conf,
                notes=[f"工具 '{tool}' 走独立 schema，未由 IR 编译器处理"],
            )

        raw_ir, repairs = self.generate_ir(query, domain, memory_hint)
        ir = parse_ir(raw_ir)

        try:
            actual_tool, params = compile_with_fallback(ir, tool)
        except CompileError as e:
            raise CompileError(f"IR 编译失败: {e}") from e

        fallback_from = tool if actual_tool != tool else None
        notes = []
        if fallback_from:
            notes.append(f"慢链路不可无损编译，已回退 {fallback_from} -> {actual_tool}")

        return PlanResult(
            tool_name=actual_tool,
            parameters=params,
            domain=domain,
            intent=intent,
            ir=raw_ir,
            fallback_from=fallback_from,
            repairs=repairs,
            route_confidence=conf,
            notes=notes,
        )
