"""Planner agent —— **单一事实源的对话状态机**（不含任何模型调用）。

设计目标：**训推一致**。同一个 agent 核心被两处复用：
  * 推理：`harness.Planner`（自己持有 vLLM client + 约束解码 + 最终 compile）；
  * 训练：`train/planner_plugin.py::PlannerEnv`（把生成交给 ms-swift 的 rollout 引擎）。

因此这里**只做两件与模型无关的事**：
  1. 产出每一轮要注入的 prompt（system / observation 文本，来自 `prompts.py`）；
  2. 消费模型输出，推进状态机（route -> IR -> validate/自修复 -> done）。

生成(policy 采样)本身不属于 agent：
  * 推理时由 `Planner` 调 client 生成；
  * 训练时由 GYMScheduler 驱动 rollout 引擎生成（这些 token 才是 GRPO 可训练对象）。
两边喂给 agent 的都是「模型这一轮的输出」，agent 返回「下一条 observation 或终止」，
从而保证两条链路走的是**逐轮、逐 token 一致**的对话序列。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .ir import IRError, parse_ir, validate_ir
from .prompts import (
    ir_observation,
    repair_observation,
    route_observation,
    route_system_prompt,
)

# 由 IR 编译器负责的检索类工具（route 命中这些才进入 IR 阶段）。单一事实源。
IR_TOOLS = {
    "vod_search", "vod_slow_search_data_search",
    "educ_search", "educ_slow_search_data_search",
}

# 阶段常量
PHASE_ROUTE = "route"
PHASE_IR = "ir"
PHASE_DONE = "done"


def loads_lenient(text: Any) -> Optional[dict]:
    """容错解析模型输出的 JSON：剥离 ```json fenced，截取首尾花括号。失败返回 None。

    agent 既可能拿到已解析的 dict（推理侧 client 已解析），也可能拿到原始字符串
    （rollout 侧从 messages 取 assistant 文本），统一在这里归一。
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


@dataclass
class AgentStep:
    """agent 消费一轮模型输出后的结果。"""
    done: bool
    next_observation: Optional[str]   # 下一轮要注入的 user observation（done 时为 None）
    phase: str                        # 语义标签：ir / route_end / ir_ok / ir_repair / ir_fail
    info: dict = field(default_factory=dict)


class PlannerAgent:
    """route -> IR -> validate(自修复) -> done 的纯状态机。

    轨迹形态（推理与 rollout 完全一致）::

        system : route_system_prompt()          (含路由 few-shot，文本内嵌)
        user   : route_observation(query)        (reset / 首轮)
        asst   : 路由 JSON                        (模型生成)
        user   : ir_observation(domain, query)   (含 IR few-shot + schema，文本内嵌)
        asst   : IR JSON                          (模型生成)
        [user  : repair_observation(errs)         (校验失败回灌, <= max_repairs)
         asst  : 修正后的 IR JSON]
    """

    def __init__(self, query: str, memory_hint: str = "", max_repairs: int = 2,
                 route_only: bool = False):
        self.query = query or ""
        self.memory_hint = memory_hint or ""
        self.max_repairs = int(max_repairs)
        self.route_only = route_only

        self.phase = PHASE_ROUTE
        self.repairs = 0

        # 路由结果
        self.route: Optional[dict] = None
        self.domain: Optional[str] = None
        self.routed_tool: Optional[str] = None
        self.intent: Optional[str] = None
        self.confidence: Optional[float] = None

        # IR 结果
        self.final_ir: Optional[dict] = None
        self.ir_valid: bool = False
        self.errs: list[str] = []

    # ---- 供两处调用者拿 prompt ----
    def system_prompt(self) -> str:
        return route_system_prompt()

    def first_observation(self) -> str:
        return route_observation(self.query, self.memory_hint)

    # ---- 消费模型输出，推进状态机 ----
    def observe(self, model_output: Any) -> AgentStep:
        if self.phase == PHASE_ROUTE:
            return self._on_route(model_output)
        if self.phase == PHASE_IR:
            return self._on_ir(model_output)
        return AgentStep(done=True, next_observation=None, phase=PHASE_DONE)

    def _on_route(self, model_output: Any) -> AgentStep:
        route = loads_lenient(model_output) or {}
        self.route = route
        self.domain = route.get("domain")
        self.routed_tool = route.get("tool")
        self.intent = route.get("intent")
        self.confidence = route.get("confidence")

        # route_only 模式：路由后直接结束，不进入 IR 阶段（用于只训路由能力）
        if self.route_only:
            self.phase = PHASE_DONE
            return AgentStep(
                done=True, next_observation=None, phase="route_end",
                info={"routed_tool": self.routed_tool, "domain": self.domain,
                      "reason": "route_only"},
            )

        # 非检索类工具 / 路由不合法 -> 不进入 IR 阶段，直接结束
        if self.routed_tool not in IR_TOOLS or self.domain not in ("vod", "educ"):
            self.phase = PHASE_DONE
            return AgentStep(
                done=True, next_observation=None, phase="route_end",
                info={"routed_tool": self.routed_tool, "domain": self.domain,
                      "reason": "non_ir_tool_or_bad_route"},
            )

        self.phase = PHASE_IR
        return AgentStep(
            done=False,
            next_observation=ir_observation(self.query, self.domain, self.memory_hint),
            phase="ir",
            info={"routed_tool": self.routed_tool, "domain": self.domain},
        )

    def _on_ir(self, model_output: Any) -> AgentStep:
        raw = loads_lenient(model_output)
        errs: list[str] = []
        ir_ok = False
        if raw is None:
            errs = ["输出不是合法 JSON"]
        else:
            try:
                ir = parse_ir(raw)
                errs = validate_ir(ir)
                ir_ok = not errs
            except IRError as e:
                errs = [str(e)]
            except Exception as e:  # 兜底：模型可能吐任意畸形结构，绝不能让 rollout/推理崩
                errs = [f"IR 解析异常: {type(e).__name__}: {e}"]

        self.final_ir = raw
        self.ir_valid = ir_ok
        self.errs = errs

        if ir_ok:
            self.phase = PHASE_DONE
            return AgentStep(done=True, next_observation=None, phase="ir_ok",
                             info={"repairs": self.repairs})

        # 校验失败：还有修复预算就回灌错误，否则结束
        if self.repairs < self.max_repairs:
            self.repairs += 1
            return AgentStep(done=False, next_observation=repair_observation(errs),
                             phase="ir_repair", info={"repairs": self.repairs, "errs": errs})

        self.phase = PHASE_DONE
        return AgentStep(done=True, next_observation=None, phase="ir_fail",
                         info={"repairs": self.repairs, "errs": errs})
