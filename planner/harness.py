"""Harness —— 推理侧编排：驱动共享的 `PlannerAgent` 状态机走完整条链路。

覆盖 4 域的完整流程：
  * vod/educ 检索类：route → IR generate(约束解码) → validate(+自修复) → compile
    - vod_search 意图：compile 后根据字段子集自动选 vod_search 或 vod_search_all
    - vod_relate_search：compile 成 relate 格式（4 字段布尔 DSL）
  * vod_fuzzy_search：route → 直接结束，只传 query 原文
  * vod_personalized_search / vod_history：route → 独立 slot-fill
  * audio：route → audio slot-fill(约束解码) → compile
  * device：route → device slot-fill(约束解码) → compile

支持多意图并发：
  * plan_multi() 先做意图拆分（约束解码），检测到多意图时并发 plan 各子请求。
  * 单意图时退化为普通 plan()，无额外开销。

与训练侧 `train/planner_plugin.py::PlannerEnv` **共用同一个 agent 核心**
（`planner.agent.PlannerAgent`）：prompt 构造与状态转移完全一致。
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    build_intent_split_schema,
    build_ir_schema,
    build_route_schema,
)
from .ir import IRError, parse_ir
from .prompts import intent_split_observation, intent_split_system_prompt
from .retrieve_client import RetrieveClient, RetrieveConfig, RetrieveResult
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
                 educ_only: bool = False, use_eb_prompt: bool = True,
                 use_retrieve: bool = False,
                 retrieve_config: Optional[RetrieveConfig] = None):
        self.client = client
        self.max_repairs = max_repairs
        self.vod_only = vod_only
        self.educ_only = educ_only
        self.use_eb_prompt = use_eb_prompt
        # 检索层开关：默认关闭，关闭时与原有逻辑完全一致
        self.use_retrieve = use_retrieve
        self._retrieve_client: Optional[RetrieveClient] = None
        if self.use_retrieve:
            self._retrieve_client = RetrieveClient(retrieve_config)

    def plan(self, query: str, memory_hint: str = "",
             available_tools: Optional[list[str]] = None) -> PlanResult:
        agent = PlannerAgent(query, memory_hint=memory_hint, max_repairs=self.max_repairs,
                             vod_only=self.vod_only, educ_only=self.educ_only,
                             use_eb_prompt=self.use_eb_prompt)

        trace: list[dict] = []  # 记录每次 LLM 调用

        # ---- [可选] 检索层召回 ----
        retrieve_result: Optional[RetrieveResult] = None
        if self.use_retrieve and self._retrieve_client is not None:
            retrieve_result = self._retrieve_client.retrieve(query)
            if retrieve_result and retrieve_result.raw:
                trace.append({
                    "stage": "retrieve",
                    "output": retrieve_result.raw,
                })

        # ---- 轨迹起点：system + 首个 observation ----
        messages = [
            {"role": "system", "content": agent.system_prompt()},
            {"role": "user", "content": self._build_route_observation(agent, retrieve_result)},
        ]

        # ---- 阶段1：路由（约束解码）----
        route_schema = self._build_route_schema(retrieve_result, available_tools)
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
                tool_name=agent.routed_tool or "vod_fuzzy_search",
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
            return self._handle_audio(agent, messages, step, trace, retrieve_result)

        # D) Device slot-fill
        if step.phase == "device":
            return self._handle_device(agent, messages, step, trace, retrieve_result,
                                       available_tools)

        # E) IR 生成 + 校验自修复
        if step.phase == "ir":
            return self._handle_ir(agent, messages, step, query, trace, retrieve_result)

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

    def plan_multi(self, query: str, memory_hint: str = "",
                   max_workers: int = 4,
                   available_tools: Optional[list[str]] = None) -> list[PlanResult]:
        """多意图规划：先做意图拆分，检测到多意图时并发 plan 各子请求。

        单意图时退化为 [plan(query)]，无额外开销（跳过拆分阶段）。
        多意图时并发调用 plan()，返回有序的 PlanResult 列表。

        Args:
            query: 原始用户请求
            memory_hint: 对话上下文摘要
            max_workers: 并发规划子意图的最大线程数
            available_tools: 可选，请求传入的可用工具名列表（用于收窄路由 schema）

        Returns:
            list[PlanResult]: 按子请求顺序排列的规划结果列表
        """
        # ---- 阶段0：意图拆分（约束解码）----
        split_schema = build_intent_split_schema()
        messages = [
            {"role": "system", "content": intent_split_system_prompt()},
            {"role": "user", "content": intent_split_observation(query, memory_hint)},
        ]
        split_raw = self.client.complete_json(messages, guided_json=split_schema)

        is_multi = split_raw.get("multi", False)
        sub_queries = split_raw.get("sub_queries", [query])

        # 校验：拆分结果必须非空
        if not sub_queries:
            sub_queries = [query]
            is_multi = False

        # 单意图 → 直接走原有 plan，避免不必要的开销
        if not is_multi or len(sub_queries) <= 1:
            return [self.plan(query, memory_hint=memory_hint, available_tools=available_tools)]

        # ---- 多意图并发规划 ----
        results: list[Optional[PlanResult]] = [None] * len(sub_queries)
        errors: list[Optional[Exception]] = [None] * len(sub_queries)

        def _plan_one(idx: int, sub_q: str) -> tuple[int, PlanResult]:
            return idx, self.plan(sub_q, memory_hint=memory_hint, available_tools=available_tools)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(sub_queries))) as pool:
            futures = {
                pool.submit(_plan_one, i, sq): i
                for i, sq in enumerate(sub_queries)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    _, result = future.result()
                    results[idx] = result
                except Exception as e:
                    errors[idx] = e

        # 收集成功结果（保序），失败的附带错误信息
        final: list[PlanResult] = []
        for i, (res, err) in enumerate(zip(results, errors)):
            if res is not None:
                res.notes.insert(0, f"多意图拆分: 子请求[{i}]='{sub_queries[i]}'")
                final.append(res)
            elif err is not None:
                # 子请求失败：生成一个带错误标记的 PlanResult
                final.append(PlanResult(
                    tool_name="error",
                    parameters={"query": sub_queries[i], "error": str(err)},
                    domain="unknown",
                    notes=[f"多意图拆分: 子请求[{i}]='{sub_queries[i]}' 执行失败: {err}"],
                ))

        return final if final else [self.plan(query, memory_hint=memory_hint)]

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
                      trace: list[dict],
                      retrieve_result: Optional[RetrieveResult] = None) -> PlanResult:
        """有声域：slot-fill → compile_audio。"""
        obs = step.next_observation
        # 注入检索提示
        if retrieve_result and self.use_retrieve:
            hint = self._build_retrieve_hint_for_slotfill(retrieve_result, agent.routed_tool)
            if hint:
                obs = obs + "\n\n" + hint
        messages.append({"role": "user", "content": obs})
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
                       trace: list[dict],
                       retrieve_result: Optional[RetrieveResult] = None,
                       available_tools: Optional[list[str]] = None) -> PlanResult:
        """设备域：slot-fill → compile_device。"""
        obs = step.next_observation
        # 注入检索提示
        if retrieve_result and self.use_retrieve:
            hint = self._build_retrieve_hint_for_slotfill(retrieve_result, agent.routed_tool)
            if hint:
                obs = obs + "\n\n" + hint
        messages.append({"role": "user", "content": obs})
        device_schema = build_device_schema()
        # 硬约束：device schema 的 tool enum 只能是请求传入的工具
        if available_tools:
            import copy
            device_schema = copy.deepcopy(device_schema)
            # 直接用 available_tools，不做兜底
            device_schema["properties"]["tool"]["enum"] = available_tools
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
        tool_name, params = compile_device(slot_fill, query=agent.query)

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
                   trace: list[dict],
                   retrieve_result: Optional[RetrieveResult] = None) -> PlanResult:
        """IR 生成 + 校验自修复 + 编译。"""
        # 若检索层有结果，用其 parameter_ids 收窄 IR schema 的 field enum
        ir_schema = self._build_ir_schema(agent.domain, retrieve_result)
        repair_round = 0
        while not step.done:
            obs = step.next_observation
            # 首轮 IR 生成时注入检索提示（修复轮不再重复注入）
            if repair_round == 0 and retrieve_result and self.use_retrieve:
                hint = self._build_retrieve_hint_for_ir(retrieve_result, agent.routed_tool)
                if hint:
                    obs = obs + "\n\n" + hint
            messages.append({"role": "user", "content": obs})
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
        ir = parse_ir(agent.final_ir, domain_hint=agent.domain)
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
    # 检索层辅助方法（use_retrieve=False 时不会被调用）
    # ===========================================================================

    def _build_route_schema(self, retrieve_result: Optional[RetrieveResult],
                            available_tools: Optional[list[str]] = None) -> dict:
        """构建路由 schema，严格限制 tool enum 为请求传入的 toolList。

        优先级：
        1. available_tools（来自请求 toolList）→ 硬约束，只能选这些工具
        2. retrieve_result → 在 available_tools 范围内进一步排序（可选）
        """
        base_schema = build_route_schema(vod_only=self.vod_only, educ_only=self.educ_only)

        import copy
        schema = copy.deepcopy(base_schema)
        original_tools = schema["properties"]["tool"]["enum"]
        original_domains = schema["properties"]["domain"]["enum"]

        # 硬约束：只能选请求传入的工具
        if available_tools:
            # 直接用请求传入的工具列表（不管 planner 内部支不支持）
            schema["properties"]["tool"]["enum"] = available_tools
            # 同步收窄 domain：只保留包含可用工具的域
            from .grammar import ROUTE_TOOLS
            active_domains = set()
            for domain_key, domain_tools in ROUTE_TOOLS.items():
                if any(t in available_tools for t in domain_tools):
                    active_domains.add(domain_key)
            # 也检查 available_tools 里是否有设备工具（不在 ROUTE_TOOLS 的也算 device）
            from .agent import DEVICE_TOOLS as AGENT_DEVICE_TOOLS
            if any(t in AGENT_DEVICE_TOOLS for t in available_tools):
                active_domains.add("device")
            narrowed_domains = [d for d in original_domains if d in active_domains]
            if narrowed_domains:
                schema["properties"]["domain"]["enum"] = narrowed_domains

        # 检索层进一步收窄（在 available_tools 硬约束后的基础上，可选）
        if self.use_retrieve and retrieve_result and retrieve_result.tools:
            retrieve_tools = [t.get("tool_name") for t in retrieve_result.tools if t.get("tool_name")]
            current_tools = schema["properties"]["tool"]["enum"]
            narrowed_tools = [t for t in retrieve_tools if t in current_tools]
            if narrowed_tools:
                schema["properties"]["tool"]["enum"] = narrowed_tools
                _DOMAIN_MAP = {"影视": "vod", "少儿": "educ", "有声": "audio", "设备": "device"}
                retrieve_domains = list(dict.fromkeys(
                    _DOMAIN_MAP.get(t.get("domain", ""), t.get("domain", ""))
                    for t in retrieve_result.tools if t.get("domain")
                ))
                current_domains = schema["properties"]["domain"]["enum"]
                narrowed_domains = [d for d in retrieve_domains if d in current_domains]
                if narrowed_domains:
                    schema["properties"]["domain"]["enum"] = narrowed_domains

        return schema

    def _build_ir_schema(self, domain: str,
                         retrieve_result: Optional[RetrieveResult]) -> dict:
        """构建 IR schema，若检索层有结果则收窄 field enum 到检索建议的参数。"""
        base_schema = build_ir_schema(domain)
        if not self.use_retrieve or not retrieve_result or not retrieve_result.parameters:
            return base_schema

        # 提取检索建议的 parameter_ids（去重）
        retrieve_params = list(dict.fromkeys(
            p.get("parameter_id") for p in retrieve_result.parameters
            if p.get("parameter_id")
        ))
        if not retrieve_params:
            return base_schema

        # 排除不适合放入 IR field 的系统参数（如 retext, action, query 是编译器处理的）
        _SYSTEM_PARAMS = {"retext", "action", "query"}
        ir_fields = [f for f in retrieve_params if f not in _SYSTEM_PARAMS]
        if not ir_fields:
            return base_schema

        # 检索接口返回的是落地名（如 is_fee），schema 中用的是 canonical（如 fee）
        # 建立双向映射以确保匹配
        _LANDING_TO_CANONICAL = {
            "is_fee": "fee",
            "age_range": "age",
            "release_time": "release_year",
        }
        # 将检索的落地名转为 canonical
        ir_fields_canonical = []
        for f in ir_fields:
            canonical = _LANDING_TO_CANONICAL.get(f, f)
            ir_fields_canonical.append(canonical)
            # 同时保留原名（有些字段 canonical == landing）
            if f != canonical:
                ir_fields_canonical.append(f)
        ir_fields_canonical = list(dict.fromkeys(ir_fields_canonical))  # 去重保序

        # 收窄 schema 中 Leaf 的 field enum
        import copy
        schema = copy.deepcopy(base_schema)
        defs = schema.get("$defs", {})
        leaf = defs.get("Leaf", {})
        if "oneOf" in leaf:
            for variant in leaf["oneOf"]:
                props = variant.get("properties", {})
                field_prop = props.get("field", {})
                if "enum" in field_prop:
                    original_fields = field_prop["enum"]
                    # 保留检索建议的 + 原始的交集
                    narrowed = [f for f in original_fields if f in ir_fields_canonical]
                    if narrowed:
                        field_prop["enum"] = narrowed
                    # 若交集为空则保留原始（兜底）

        return schema

    def _build_route_observation(self, agent: PlannerAgent,
                                 retrieve_result: Optional[RetrieveResult]) -> str:
        """构建路由阶段的 observation，可选注入检索提示（tool + parameter + value）。"""
        obs = agent.first_observation()
        if not self.use_retrieve or not retrieve_result:
            return obs
        parts: list[str] = []
        tool_hint = retrieve_result.format_tool_hint()
        if tool_hint:
            parts.append(tool_hint)
        param_hint = retrieve_result.format_parameter_hint()  # 不过滤 tool，全部展示
        if param_hint:
            parts.append(param_hint)
        value_hint = retrieve_result.format_value_hint()
        if value_hint:
            parts.append(value_hint)
        if parts:
            obs = obs.rstrip() + "\n\n" + "\n".join(parts) + "\n（以上为检索系统参考建议，请结合语义综合判断。）"
        return obs

    def _build_retrieve_hint_for_ir(self, retrieve_result: RetrieveResult,
                                    tool_name: Optional[str]) -> str:
        """构建 IR 阶段的检索提示（参数 + 取值）。"""
        parts: list[str] = []
        # 优先展示匹配当前工具的参数，若无则展示全部
        param_hint = retrieve_result.format_parameter_hint(tool_name=tool_name)
        if not param_hint:
            param_hint = retrieve_result.format_parameter_hint()
        if param_hint:
            parts.append(param_hint)
        value_hint = retrieve_result.format_value_hint()
        if value_hint:
            parts.append(value_hint)
        if not parts:
            return ""
        return "\n".join(parts) + "\n（以上为检索系统参考建议，仅供参考，以用户请求语义为准。）"

    def _build_retrieve_hint_for_slotfill(self, retrieve_result: RetrieveResult,
                                          tool_name: Optional[str]) -> str:
        """构建 slot-fill 阶段（audio/device）的检索提示。"""
        parts: list[str] = []
        param_hint = retrieve_result.format_parameter_hint(tool_name=tool_name)
        if param_hint:
            parts.append(param_hint)
        value_hint = retrieve_result.format_value_hint()
        if value_hint:
            parts.append(value_hint)
        if not parts:
            return ""
        return "\n".join(parts) + "\n（以上为检索系统参考建议，仅供参考。）"


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
