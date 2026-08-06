"""Planner —— 轻量编排：路由 → 分域调用 → 拿结果。"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from .vllm_client import VLLMClient
    from .prompts import (
        route_system_prompt,
        route_user_prompt,
        ir_user_prompt,
        audio_user_prompt,
        device_user_prompt,
        intent_split_system_prompt,
        intent_split_user_prompt,
    )
    from .schemas import (
        route_schema,
        ir_schema,
        audio_schema,
        device_schema,
        intent_split_schema,
    )
except ImportError:
    from vllm_client import VLLMClient
    from prompts import (
        route_system_prompt,
        route_user_prompt,
        ir_user_prompt,
        audio_user_prompt,
        device_user_prompt,
        intent_split_system_prompt,
        intent_split_user_prompt,
    )
    from schemas import (
        route_schema,
        ir_schema,
        audio_schema,
        device_schema,
        intent_split_schema,
    )


@dataclass
class PlanResult:
    tool_name: str
    parameters: dict[str, Any]
    domain: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    notes: list[str] = field(default_factory=list)


class Planner:
    def __init__(self, client: VLLMClient, max_repairs: int = 2):
        self.client = client
        self.max_repairs = max_repairs

    def plan(self, query: str, memory_hint: str = "",
             available_tools: Optional[list[str]] = None) -> PlanResult:
        """单意图规划：路由 → 分域 → 返回结果。"""

        # ---- 1. 路由 ----
        messages = [
            {"role": "system", "content": route_system_prompt(available_tools=available_tools)},
            {"role": "user", "content": route_user_prompt(query, memory_hint)},
        ]
        route = self.client.complete(messages, guided_json=route_schema(available_tools=available_tools))

        domain = route.get("domain", "")
        tool = route.get("tool", "")
        intent = route.get("intent", "")
        confidence = route.get("confidence")

        # ---- 2. 分域调用 ----
        messages.append({"role": "assistant", "content": json.dumps(route, ensure_ascii=False)})

        if domain in ("vod", "educ"):
            messages.append({"role": "user", "content": ir_user_prompt(query, domain, memory_hint, intent)})
            result = self.client.complete(messages, guided_json=ir_schema(domain))

        elif domain == "audio":
            messages.append({"role": "user", "content": audio_user_prompt(query, memory_hint)})
            result = self.client.complete(messages, guided_json=audio_schema())

        elif domain == "device":
            messages.append({"role": "user", "content": device_user_prompt(query, memory_hint)})
            result = self.client.complete(messages, guided_json=device_schema(available_tools=available_tools))

        else:
            return PlanResult(
                tool_name=tool or "unknown",
                parameters={"query": query},
                domain=domain or "unknown",
                intent=intent,
                confidence=confidence,
            )

        # ---- 3. 解析结果 ----
        tool_name = result.get("tool_name", tool)
        parameters = result.get("parameters", result)
        if "tool_name" not in result:
            tool_name = tool

        return PlanResult(
            tool_name=tool_name,
            parameters=parameters,
            domain=domain,
            intent=intent,
            confidence=confidence,
        )

    def plan_multi(self, query: str, memory_hint: str = "",
                   available_tools: Optional[list[str]] = None,
                   max_workers: int = 4) -> list[PlanResult]:
        """多意图规划：拆分 → 并发 plan。"""

        # 意图拆分
        messages = [
            {"role": "system", "content": intent_split_system_prompt()},
            {"role": "user", "content": intent_split_user_prompt(query, memory_hint)},
        ]
        split = self.client.complete(messages, guided_json=intent_split_schema())

        is_multi = split.get("multi", False)
        sub_queries = split.get("sub_queries", [query])

        if not sub_queries:
            sub_queries = [query]
        if not is_multi or len(sub_queries) <= 1:
            return [self.plan(query, memory_hint=memory_hint, available_tools=available_tools)]

        # 并发
        results: list[Optional[PlanResult]] = [None] * len(sub_queries)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(sub_queries))) as pool:
            futures = {
                pool.submit(self.plan, sq, memory_hint, available_tools): i
                for i, sq in enumerate(sub_queries)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = PlanResult(
                        tool_name="error",
                        parameters={"query": sub_queries[idx], "error": str(e)},
                        domain="unknown",
                        notes=[f"子请求[{idx}] 失败: {e}"],
                    )

        return [r for r in results if r is not None]
