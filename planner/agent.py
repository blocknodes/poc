"""Planner agent —— **单一事实源的对话状态机**（不含任何模型调用）。

设计目标：**训推一致**。同一个 agent 核心被两处复用：
  * 推理：`harness.Planner`（自己持有 vLLM client + 约束解码 + 最终 compile）；
  * 训练：`train/planner_plugin.py::PlannerEnv`（把生成交给 ms-swift 的 rollout 引擎）。

因此这里**只做两件与模型无关的事**：
  1. 产出每一轮要注入的 prompt（system / observation 文本，来自 `prompts.py`）；
  2. 消费模型输出，推进状态机（route -> IR/slot-fill -> validate/自修复 -> done）。

覆盖 4 域：
  * vod/educ 检索类 → route → IR 生成 → 校验自修复
  * vod/educ 相关推荐(relate) → route → IR 生成（4字段子集） → 校验自修复
  * vod_slow_search → route → 直接结束（只传 query）
  * vod_personalized_search / vod_history → route → 简单 slot-fill → 结束
  * audio → route → slot-fill（query + play_mode）
  * device → route → slot-fill（tool + operation + object + value）

0725-v1 变更：
  * vod_search / vod_search_all 统一意图（都进 IR 阶段，编译后再选工具）
  * vod_relate_search 进 IR 阶段（4 字段布尔 DSL）
  * vod_slow_search_data_search 路由后直接结束（只传 query 原文）
  * vod_personalized_search 仅传 category（简单 slot）
  * vod_history 传 category + time（简单 slot）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .ir import IRError, parse_ir, validate_ir
from .prompts import (
    audio_observation,
    device_observation,
    ir_observation,
    repair_observation,
    route_observation,
    route_system_prompt,
)

# 由 IR 编译器负责的检索类工具（route 命中这些才进入 IR 阶段）。单一事实源。
# vod_search 代表 vod_search + vod_search_all 统一意图
IR_TOOLS = {
    "vod_search", "vod_search_all",
    "vod_relate_search",
    "educ_search", "educ_slow_search_data_search",
    "educ_relate_recommend",
}

# 有声域工具（route 命中这些进入 audio slot-fill 阶段）
AUDIO_TOOLS = {"audio_search", "audio_chat_qa"}

# 设备域工具（route 命中这些进入 device slot-fill 阶段）
DEVICE_TOOLS = {
    "volume_control", "power_control", "signal_source_control",
    "screen_display_control", "camera_control", "network_control",
    "bluetooth_control", "input_lang_control", "image_quality_control",
    "sound_mode_control", "projection_control", "media_center_control",
    "demo_control", "personalization_control", "scene_mode_control",
    "screen_safety_control", "playback_control", "ambient_light_control",
    "system_settings_control", "ai_picture_sound_control",
}

# 简单 slot-fill 工具（route 后直接结束，harness 侧独立处理参数）
SIMPLE_TOOLS = {
    "vod_personalized_search", "vod_history",
    "educ_history",
}

# 慢链路工具（route 后直接结束，只传 query 原文）
SLOW_SEARCH_TOOLS = {
    "vod_slow_search_data_search",
}

# 阶段常量
PHASE_ROUTE = "route"
PHASE_IR = "ir"
PHASE_AUDIO = "audio"
PHASE_DEVICE = "device"
PHASE_DONE = "done"


def loads_lenient(text: Any) -> Optional[dict]:
    """容错解析模型输出的 JSON：剥离 ```json fenced，截取首尾花括号。失败返回 None。"""
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
    phase: str                        # 语义标签
    info: dict = field(default_factory=dict)


class PlannerAgent:
    """route -> IR/audio/device -> validate(自修复) -> done 的纯状态机。

    轨迹形态（推理与 rollout 完全一致）::

        system : route_system_prompt()
        user   : route_observation(query)
        asst   : 路由 JSON
        [以下分支取决于路由结果]
        --- IR 分支（vod/educ 检索类 + relate）---
        user   : ir_observation(domain, query)
        asst   : IR JSON
        [user  : repair_observation(errs)    ≤ max_repairs]
        --- Slow search 分支 ---
        直接结束（只传 query 原文）
        --- Audio 分支 ---
        user   : audio_observation(query)
        asst   : audio slot-fill JSON
        --- Device 分支 ---
        user   : device_observation(query)
        asst   : device slot-fill JSON
        --- Simple 分支（personalized/history）---
        直接结束（harness 侧单独处理）
    """

    def __init__(self, query: str, memory_hint: str = "", max_repairs: int = 2,
                 route_only: bool = False, vod_only: bool = False,
                 use_eb_prompt: bool = True):
        self.query = query or ""
        self.memory_hint = memory_hint or ""
        self.max_repairs = int(max_repairs)
        self.route_only = route_only
        self.vod_only = vod_only
        self.use_eb_prompt = use_eb_prompt

        self.phase = PHASE_ROUTE
        self.repairs = 0

        # 路由结果
        self.route: Optional[dict] = None
        self.domain: Optional[str] = None
        self.routed_tool: Optional[str] = None
        self.intent: Optional[str] = None
        self.confidence: Optional[float] = None

        # IR 结果（vod/educ 检索类）
        self.final_ir: Optional[dict] = None
        self.ir_valid: bool = False
        self.errs: list[str] = []

        # Slot-fill 结果（audio / device）
        self.slot_fill: Optional[dict] = None

    # ---- 供两处调用者拿 prompt ----
    def system_prompt(self) -> str:
        return route_system_prompt(vod_only=self.vod_only)

    def first_observation(self) -> str:
        return route_observation(self.query, self.memory_hint)

    # ---- 消费模型输出，推进状态机 ----
    def observe(self, model_output: Any) -> AgentStep:
        if self.phase == PHASE_ROUTE:
            return self._on_route(model_output)
        if self.phase == PHASE_IR:
            return self._on_ir(model_output)
        if self.phase == PHASE_AUDIO:
            return self._on_audio(model_output)
        if self.phase == PHASE_DEVICE:
            return self._on_device(model_output)
        return AgentStep(done=True, next_observation=None, phase=PHASE_DONE)

    def _on_route(self, model_output: Any) -> AgentStep:
        route = loads_lenient(model_output) or {}
        self.route = route
        self.domain = route.get("domain")
        self.routed_tool = route.get("tool")
        self.intent = route.get("intent")
        self.confidence = route.get("confidence")

        # route_only 模式：路由后直接结束
        if self.route_only:
            self.phase = PHASE_DONE
            return AgentStep(
                done=True, next_observation=None, phase="route_end",
                info={"routed_tool": self.routed_tool, "domain": self.domain,
                      "reason": "route_only"},
            )

        # ---- 根据路由结果决定下一阶段 ----

        # 1) 有声域 → audio slot-fill
        if self.domain == "audio" or self.routed_tool in AUDIO_TOOLS:
            self.phase = PHASE_AUDIO
            return AgentStep(
                done=False,
                next_observation=audio_observation(self.query, self.memory_hint),
                phase="audio",
                info={"routed_tool": self.routed_tool, "domain": self.domain},
            )

        # 2) 设备域 → device slot-fill
        if self.domain == "device" or self.routed_tool in DEVICE_TOOLS:
            self.phase = PHASE_DEVICE
            return AgentStep(
                done=False,
                next_observation=device_observation(self.query, self.memory_hint),
                phase="device",
                info={"routed_tool": self.routed_tool, "domain": self.domain},
            )

        # 3) 慢链路 → 直接结束（只传 query 原文）
        if self.routed_tool in SLOW_SEARCH_TOOLS:
            self.phase = PHASE_DONE
            return AgentStep(
                done=True, next_observation=None, phase="route_end",
                info={"routed_tool": self.routed_tool, "domain": self.domain,
                      "reason": "slow_search_passthrough"},
            )

        # 4) 简单工具（personalized/history）→ 直接结束
        if self.routed_tool in SIMPLE_TOOLS:
            self.phase = PHASE_DONE
            return AgentStep(
                done=True, next_observation=None, phase="route_end",
                info={"routed_tool": self.routed_tool, "domain": self.domain,
                      "reason": "simple_slot_fill"},
            )

        # 5) IR 检索类（vod_search/vod_search_all/vod_relate_search/educ_*）→ 进入 IR 阶段
        if self.routed_tool in IR_TOOLS and self.domain in ("vod", "educ"):
            self.phase = PHASE_IR
            return AgentStep(
                done=False,
                next_observation=ir_observation(self.query, self.domain, self.memory_hint,
                                               intent=self.intent,
                                               use_experience_bank=self.use_eb_prompt),
                phase="ir",
                info={"routed_tool": self.routed_tool, "domain": self.domain},
            )

        # 6) 其余未知工具/路由不合法 → 直接结束
        self.phase = PHASE_DONE
        return AgentStep(
            done=True, next_observation=None, phase="route_end",
            info={"routed_tool": self.routed_tool, "domain": self.domain,
                  "reason": "non_ir_tool_or_bad_route"},
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
            except Exception as e:
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

    def _on_audio(self, model_output: Any) -> AgentStep:
        """有声域 slot-fill 结果：直接消费，不做自修复（结构由 schema 保证）。"""
        raw = loads_lenient(model_output)
        self.slot_fill = raw or {}
        self.phase = PHASE_DONE
        return AgentStep(done=True, next_observation=None, phase="audio_done",
                         info={"slot_fill": self.slot_fill})

    def _on_device(self, model_output: Any) -> AgentStep:
        """设备域 slot-fill 结果：直接消费，不做自修复（结构由 schema 保证）。"""
        raw = loads_lenient(model_output)
        self.slot_fill = raw or {}
        self.phase = PHASE_DONE
        return AgentStep(done=True, next_observation=None, phase="device_done",
                         info={"slot_fill": self.slot_fill})
