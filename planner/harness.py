"""Harness —— 推理侧编排：驱动共享的 `PlannerAgent` 状态机走完整条链路

  route -> ir_generate(约束解码) -> validate_ir(+自修复重试) -> compile(+flat回退)

与训练侧 `train/planner_plugin.py::PlannerEnv` **共用同一个 agent 核心**
（`planner.agent.PlannerAgent`）：prompt 构造与状态转移完全一致，唯一区别是
本类自己持有 vLLM client 负责生成 + 约束解码，并在收尾做 compile。
非检索类工具(relate/personalized/history/clip)路由后不走 IR，留出扩展点。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .agent import IR_TOOLS, PlannerAgent
from .compiler import CompileError, compile_with_fallback
from .grammar import build_ir_schema, build_route_schema
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
    fallback_from: Optional[str] = None      # 若发生 flat->nested 回退
    repairs: int = 0                         # IR 自修复次数
    route_confidence: Optional[float] = None
    notes: list[str] = field(default_factory=list)


class Planner:
    def __init__(self, client: VLLMClient, max_repairs: int = 2):
        self.client = client
        self.max_repairs = max_repairs

    def plan(self, query: str, memory_hint: str = "") -> PlanResult:
        agent = PlannerAgent(query, memory_hint=memory_hint, max_repairs=self.max_repairs)

        # ---- 轨迹起点：system + 首个 observation（与 rollout 的 reset 完全一致）----
        messages = [
            {"role": "system", "content": agent.system_prompt()},
            {"role": "user", "content": agent.first_observation()},
        ]

        # ---- 阶段1：路由（约束解码）----
        route_raw = self.client.complete_json(messages, guided_json=build_route_schema())
        messages.append({"role": "assistant", "content": json.dumps(route_raw, ensure_ascii=False)})
        step = agent.observe(route_raw)

        if step.phase == "route_end":
            # 非检索类：留给独立 slot-filler（本 POC 不实现），直接回传路由结果
            return PlanResult(
                tool_name=agent.routed_tool, parameters={}, domain=agent.domain,
                intent=agent.intent, route_confidence=agent.confidence,
                notes=[f"工具 '{agent.routed_tool}' 走独立 schema，未由 IR 编译器处理"],
            )

        # ---- 阶段2+3：IR 生成 + 校验自修复（约束解码）----
        while not step.done:
            messages.append({"role": "user", "content": step.next_observation})
            ir_raw = self.client.complete_json(messages, guided_json=build_ir_schema(agent.domain))
            messages.append({"role": "assistant", "content": json.dumps(ir_raw, ensure_ascii=False)})
            step = agent.observe(ir_raw)

        if not agent.ir_valid:
            raise IRError("IR 校验在最大重试后仍失败: " + "; ".join(agent.errs))

        # ---- 阶段4：编译（+flat 回退）----
        ir = parse_ir(agent.final_ir)
        try:
            actual_tool, params = compile_with_fallback(ir, agent.routed_tool)
        except CompileError as e:
            raise CompileError(f"IR 编译失败: {e}") from e

        fallback_from = agent.routed_tool if actual_tool != agent.routed_tool else None
        notes = []
        if fallback_from:
            notes.append(f"慢链路不可无损编译，已回退 {fallback_from} -> {actual_tool}")

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
        )
