#!/usr/bin/env python3
"""demo.py —— 展示四域（影视/少儿/有声/设备）的端到端 planner 效果。

Usage:
    python demo.py                              # 离线 stub，跑预设 query
    python demo.py --query "声音大一点"          # 离线 stub，跑自定义 query
    python demo.py --live --query "播放评书"     # 连 vLLM，跑自定义 query
    python demo.py --debug --query "刘德华电影"  # 离线 + 打印中间 messages/schema（摘要）
    python demo.py --verbose --query "刘德华电影"  # 离线 + 完整打印 LLM 输入输出
    python demo.py --live --debug               # 连 vLLM + debug 模式跑预设 query
    python demo.py --live --verbose             # 连 vLLM + verbose 模式跑预设 query
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner import Planner, VLLMClient, VLLMConfig
from planner.agent import PlannerAgent
from planner.grammar import build_audio_schema, build_device_schema, build_ir_schema, build_route_schema


def make_stub():
    """根据 schema title 决定回什么 JSON（覆盖四域）。"""
    def responder(messages, guided_json):
        title = (guided_json or {}).get("title", "")

        # 取最后一条 user message 来判断请求内容
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        if title == "planner_route":
            # 简单关键词路由
            if "听" in last_user or "评书" in last_user or "相声" in last_user or "有声" in last_user:
                return json.dumps({"domain": "audio", "intent": "audio_search",
                                   "tool": "audio_search", "confidence": 0.9})
            if any(k in last_user for k in ["音量", "声音", "静音", "关机",
                                             "亮度", "蓝牙", "WiFi", "暂停", "快进",
                                             "护眼", "氛围灯", "HDMI"]):
                if "暂停" in last_user or "快进" in last_user or "快退" in last_user:
                    tool = "playback_control"
                elif "关机" in last_user:
                    tool = "power_control"
                elif "亮度" in last_user:
                    tool = "image_quality_control"
                elif "蓝牙" in last_user:
                    tool = "bluetooth_control"
                elif "WiFi" in last_user:
                    tool = "network_control"
                elif "护眼" in last_user:
                    tool = "screen_safety_control"
                elif "氛围灯" in last_user:
                    tool = "ambient_light_control"
                elif "HDMI" in last_user:
                    tool = "signal_source_control"
                else:
                    tool = "volume_control"
                return json.dumps({"domain": "device", "intent": "device_control",
                                   "tool": tool, "confidence": 0.95})
            if "儿歌" in last_user or "启蒙" in last_user or "拼音" in last_user:
                return json.dumps({"domain": "educ", "intent": "search",
                                   "tool": "educ_search", "confidence": 0.85})
            # 默认 vod
            return json.dumps({"domain": "vod", "intent": "search",
                               "tool": "vod_search", "confidence": 0.9})

        if title == "audio_slot_fill":
            query = last_user.split("用户请求：")[-1].split("\n")[0].strip() if "用户请求" in last_user else last_user
            return json.dumps({"tool": "audio_search", "query": query,
                               "play_mode": "play", "screen_mode": "normal"})

        if title == "device_slot_fill":
            query_part = last_user
            if "用户请求：" in last_user:
                query_part = last_user.split("用户请求：")[-1].split("\n")[0].strip()
            if "暂停" in query_part:
                return json.dumps({"tool": "playback_control", "operation": "关闭", "object": "播放"})
            if "快进" in query_part:
                return json.dumps({"tool": "playback_control", "operation": "提高", "object": "播放进度", "value": "30秒"})
            if "音量" in query_part or "声音" in query_part:
                if "大" in query_part or "高" in query_part:
                    return json.dumps({"tool": "volume_control", "operation": "提高", "object": "音量"})
                elif "小" in query_part or "低" in query_part:
                    return json.dumps({"tool": "volume_control", "operation": "降低", "object": "音量"})
                else:
                    return json.dumps({"tool": "volume_control", "operation": "设置", "object": "音量", "value": "50"})
            if "亮度" in query_part:
                return json.dumps({"tool": "image_quality_control", "operation": "设置", "object": "亮度", "value": "50"})
            if "蓝牙" in query_part:
                return json.dumps({"tool": "bluetooth_control", "operation": "打开", "object": "蓝牙"})
            if "护眼" in query_part:
                return json.dumps({"tool": "screen_safety_control", "operation": "打开", "object": "护眼模式"})
            if "关机" in query_part:
                return json.dumps({"tool": "power_control", "operation": "关闭", "object": "电源"})
            return json.dumps({"tool": "system_settings_control", "operation": "打开", "object": "设置"})

        # IR 阶段（vod/educ）
        if "educ" in title:
            return json.dumps({
                "domain": "educ", "action": "search",
                "query": {"and": [
                    {"field": "children_second_genre", "value": "儿歌"},
                    {"field": "language", "value": "英语"},
                    {"field": "fee", "value": 0}]}
            })
        # 默认 vod IR
        return json.dumps({
            "domain": "vod", "action": "search",
            "query": {"and": [
                {"field": "actor", "value": "刘德华"},
                {"field": "category", "value": "电影"},
                {"field": "fee", "value": 0}]},
        }, ensure_ascii=False)

    return responder


DEMO_QUERIES = [
    "我想看刘德华的免费电影",
    "给孩子找免费的英语儿歌",
    "播放三国演义评书",
    "声音大一点",
    "暂停",
]


class DebugPlanner(Planner):
    """带 debug 输出的 Planner：打印每一轮的 messages 和 guided_json schema。"""

    def plan(self, query: str, memory_hint: str = ""):
        from planner.agent import PlannerAgent
        from planner.compiler import compile_audio, compile_device, compile_with_fallback
        from planner.ir import IRError, parse_ir

        agent = PlannerAgent(query, memory_hint=memory_hint, max_repairs=self.max_repairs)

        messages = [
            {"role": "system", "content": agent.system_prompt()},
            {"role": "user", "content": agent.first_observation()},
        ]

        # ---- Route ----
        route_schema = build_route_schema()
        self._debug_call("ROUTE", messages, route_schema)
        route_raw = self.client.complete_json(messages, guided_json=route_schema)
        self._debug_response("ROUTE", route_raw)
        messages.append({"role": "assistant", "content": json.dumps(route_raw, ensure_ascii=False)})
        step = agent.observe(route_raw)

        # Simple / route_end
        if step.done:
            return self._build_result(agent, step)

        # ---- Phase 2: audio / device / IR ----
        schema = self._get_phase_schema(step, agent)
        self._debug_call(step.phase.upper(), messages, schema)

        if step.phase == "ir":
            # IR loop with repair
            while not step.done:
                messages.append({"role": "user", "content": step.next_observation})
                ir_raw = self.client.complete_json(messages, guided_json=schema)
                self._debug_response("IR", ir_raw)
                messages.append({"role": "assistant", "content": json.dumps(ir_raw, ensure_ascii=False)})
                step = agent.observe(ir_raw)
                if not step.done and step.phase == "ir_repair":
                    self._debug_info("IR_REPAIR", {"errs": step.info.get("errs", [])})

            if not agent.ir_valid:
                raise IRError("IR 校验失败: " + "; ".join(agent.errs))
            ir = parse_ir(agent.final_ir)
            actual_tool, params = compile_with_fallback(ir, agent.routed_tool)
            from planner.harness import PlanResult
            return PlanResult(
                tool_name=actual_tool, parameters=params, domain=agent.domain,
                intent=agent.intent, ir=agent.final_ir, repairs=agent.repairs,
                route_confidence=agent.confidence,
                fallback_from=agent.routed_tool if actual_tool != agent.routed_tool else None,
            )
        else:
            # audio / device slot-fill
            messages.append({"role": "user", "content": step.next_observation})
            slot_raw = self.client.complete_json(messages, guided_json=schema)
            self._debug_response(step.phase.upper(), slot_raw)
            messages.append({"role": "assistant", "content": json.dumps(slot_raw, ensure_ascii=False)})
            step = agent.observe(slot_raw)

            slot_fill = agent.slot_fill or {}
            if agent.domain == "audio":
                tool_name, params = compile_audio(slot_fill)
            else:
                tool_name, params = compile_device(slot_fill)
            from planner.harness import PlanResult
            return PlanResult(
                tool_name=tool_name, parameters=params, domain=agent.domain or "device",
                intent=agent.intent, slot_fill=slot_fill, route_confidence=agent.confidence,
            )

    def _get_phase_schema(self, step, agent):
        if step.phase == "audio":
            return build_audio_schema()
        elif step.phase == "device":
            return build_device_schema()
        else:
            return build_ir_schema(agent.domain)

    def _build_result(self, agent, step):
        from planner.harness import PlanResult
        return PlanResult(
            tool_name=agent.routed_tool or "unknown", parameters={},
            domain=agent.domain or "unknown", intent=agent.intent,
            route_confidence=agent.confidence,
            notes=[f"route_end: {step.info.get('reason', '')}"],
        )

    def _debug_call(self, phase, messages, schema):
        print(f"\n  {'─'*50}")
        print(f"  [DEBUG {phase}] → 调用模型")
        print(f"  [DEBUG {phase}] messages ({len(messages)} 轮):")
        for i, m in enumerate(messages):
            role = m["role"]
            content = m["content"]
            preview = content[:120].replace("\n", "\\n") + ("..." if len(content) > 120 else "")
            print(f"    [{i}] {role}: {preview}")
        print(f"  [DEBUG {phase}] guided_json schema title: {schema.get('title', '?')}")
        print(f"  [DEBUG {phase}] schema keys: {list(schema.get('properties', {}).keys())}")

    def _debug_response(self, phase, response):
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                pass
        print(f"  [DEBUG {phase}] ← 模型输出: {json.dumps(response, ensure_ascii=False)}")

    def _debug_info(self, phase, info):
        print(f"  [DEBUG {phase}] info: {json.dumps(info, ensure_ascii=False)}")


class VerbosePlanner(Planner):
    """Verbose 模式的 Planner：完整打印每次 LLM 调用的 messages 全文、guided_json schema 全文、模型完整返回。"""

    def plan(self, query: str, memory_hint: str = ""):
        from planner.agent import PlannerAgent
        from planner.compiler import compile_audio, compile_device, compile_with_fallback
        from planner.ir import IRError, parse_ir

        agent = PlannerAgent(query, memory_hint=memory_hint, max_repairs=self.max_repairs)

        messages = [
            {"role": "system", "content": agent.system_prompt()},
            {"role": "user", "content": agent.first_observation()},
        ]

        # ---- Route ----
        route_schema = build_route_schema()
        self._verbose_call("ROUTE", messages, route_schema)
        route_raw = self.client.complete_json(messages, guided_json=route_schema)
        self._verbose_response("ROUTE", route_raw)
        messages.append({"role": "assistant", "content": json.dumps(route_raw, ensure_ascii=False)})
        step = agent.observe(route_raw)

        # Simple / route_end
        if step.done:
            return self._build_result(agent, step)

        # ---- Phase 2: audio / device / IR ----
        schema = self._get_phase_schema(step, agent)

        if step.phase == "ir":
            while not step.done:
                messages.append({"role": "user", "content": step.next_observation})
                self._verbose_call("IR", messages, schema)
                ir_raw = self.client.complete_json(messages, guided_json=schema)
                self._verbose_response("IR", ir_raw)
                messages.append({"role": "assistant", "content": json.dumps(ir_raw, ensure_ascii=False)})
                step = agent.observe(ir_raw)
                if not step.done and step.phase == "ir_repair":
                    self._verbose_info("IR_REPAIR", {"errs": step.info.get("errs", [])})

            if not agent.ir_valid:
                raise IRError("IR 校验失败: " + "; ".join(agent.errs))
            ir = parse_ir(agent.final_ir)
            actual_tool, params = compile_with_fallback(ir, agent.routed_tool)
            from planner.harness import PlanResult
            return PlanResult(
                tool_name=actual_tool, parameters=params, domain=agent.domain,
                intent=agent.intent, ir=agent.final_ir, repairs=agent.repairs,
                route_confidence=agent.confidence,
                fallback_from=agent.routed_tool if actual_tool != agent.routed_tool else None,
            )
        else:
            # audio / device slot-fill
            messages.append({"role": "user", "content": step.next_observation})
            self._verbose_call(step.phase.upper(), messages, schema)
            slot_raw = self.client.complete_json(messages, guided_json=schema)
            self._verbose_response(step.phase.upper(), slot_raw)
            messages.append({"role": "assistant", "content": json.dumps(slot_raw, ensure_ascii=False)})
            step = agent.observe(slot_raw)

            slot_fill = agent.slot_fill or {}
            if agent.domain == "audio":
                tool_name, params = compile_audio(slot_fill)
            else:
                tool_name, params = compile_device(slot_fill)
            from planner.harness import PlanResult
            return PlanResult(
                tool_name=tool_name, parameters=params, domain=agent.domain or "device",
                intent=agent.intent, slot_fill=slot_fill, route_confidence=agent.confidence,
            )

    def _get_phase_schema(self, step, agent):
        if step.phase == "audio":
            return build_audio_schema()
        elif step.phase == "device":
            return build_device_schema()
        else:
            return build_ir_schema(agent.domain)

    def _build_result(self, agent, step):
        from planner.harness import PlanResult
        return PlanResult(
            tool_name=agent.routed_tool or "unknown", parameters={},
            domain=agent.domain or "unknown", intent=agent.intent,
            route_confidence=agent.confidence,
            notes=[f"route_end: {step.info.get('reason', '')}"],
        )

    def _verbose_call(self, phase, messages, schema):
        print(f"\n{'━'*70}")
        print(f"  [VERBOSE {phase}] ═══ LLM 调用输入 ═══")
        print(f"{'━'*70}")
        print(f"\n  ── messages ({len(messages)} 轮) ──\n")
        for i, m in enumerate(messages):
            role = m["role"]
            content = m["content"]
            print(f"  ┌─ [{i}] role: {role}")
            print(f"  │  content:")
            for line in content.split("\n"):
                print(f"  │  {line}")
            print(f"  └{'─'*50}")
        print(f"\n  ── guided_json schema ──\n")
        print(json.dumps(schema, ensure_ascii=False, indent=2))
        print()

    def _verbose_response(self, phase, response):
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                pass
        print(f"\n{'─'*70}")
        print(f"  [VERBOSE {phase}] ═══ LLM 输出 ═══")
        print(f"{'─'*70}")
        print(json.dumps(response, ensure_ascii=False, indent=2))
        print()

    def _verbose_info(self, phase, info):
        print(f"\n  [VERBOSE {phase}] info:")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print()


def run_query(planner, query, debug=False):
    """运行单个 query 并输出结果。"""
    print(f"{'='*60}")
    print(f"用户：{query}")
    print(f"{'─'*60}")
    res = planner.plan(query)
    print(f"\n  ▶ domain:     {res.domain}")
    print(f"  ▶ intent:     {res.intent}")
    print(f"  ▶ tool:       {res.tool_name}")
    print(f"  ▶ parameters: {json.dumps(res.parameters, ensure_ascii=False, indent=4)}")
    if res.slot_fill:
        print(f"  ▶ slot_fill:  {json.dumps(res.slot_fill, ensure_ascii=False)}")
    if res.ir:
        print(f"  ▶ IR:         {json.dumps(res.ir, ensure_ascii=False)}")
    if res.fallback_from:
        print(f"  ▶ fallback:   {res.fallback_from} → {res.tool_name}")
    if res.repairs:
        print(f"  ▶ repairs:    {res.repairs}")
    if res.notes:
        print(f"  ▶ notes:      {res.notes}")
    print()


def main():
    parser = argparse.ArgumentParser(description="电视端 AI Planner demo（4域：影视/少儿/有声/设备）")
    parser.add_argument("--live", action="store_true", help="连接真实 vLLM（需 VLLM_BASE_URL / VLLM_MODEL）")
    parser.add_argument("--debug", action="store_true", help="打印中间过程（messages / schema / 模型输出）")
    parser.add_argument("--verbose", action="store_true",
                        help="完整打印每次 LLM 调用的输入输出（messages 全文 + schema 全文 + 模型完整返回）")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="自定义用户请求（不传则跑预设 demo queries）")
    parser.add_argument("--memory", "-m", type=str, default="",
                        help="对话上下文 memory_hint（可选）")
    args = parser.parse_args()

    # 构造 client
    if args.live:
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        model = os.environ.get("VLLM_MODEL", "your-30b-moe")
        client = VLLMClient(VLLMConfig(base_url=base_url, model=model))
        print(f"[LIVE] 连接 vLLM: {base_url} / {model}")
    else:
        client = VLLMClient(responder=make_stub())
        print("[STUB] 离线模式（stub responder）")

    # 构造 planner
    if args.verbose:
        planner = VerbosePlanner(client)
        print("[VERBOSE] 已开启 verbose 模式（完整打印 LLM 输入输出）")
    elif args.debug:
        planner = DebugPlanner(client)
        print("[DEBUG] 已开启 debug 模式")
    else:
        planner = Planner(client)

    print()

    # 确定要跑哪些 query
    queries = [args.query] if args.query else DEMO_QUERIES

    for query in queries:
        run_query(planner, query, debug=args.debug)


if __name__ == "__main__":
    main()
