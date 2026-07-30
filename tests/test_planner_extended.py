"""扩展离线测试：覆盖新增的 audio / device 域 + 路由扩展。

运行：  python tests/test_planner_extended.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.compiler import compile_audio, compile_device  # noqa: E402
from planner.grammar import (  # noqa: E402
    build_audio_schema,
    build_device_schema,
    build_route_schema,
)
from planner.agent import (  # noqa: E402
    AUDIO_TOOLS,
    DEVICE_TOOLS,
    IR_TOOLS,
    SIMPLE_TOOLS,
    PlannerAgent,
)
from planner.harness import Planner  # noqa: E402
from planner.vllm_client import VLLMClient  # noqa: E402


# ============================================================
# Route schema tests
# ============================================================

def test_route_schema_has_all_domains():
    s = build_route_schema()
    domains = s["properties"]["domain"]["enum"]
    assert "vod" in domains
    assert "educ" in domains
    assert "audio" in domains
    assert "device" in domains


def test_route_schema_has_device_tools():
    s = build_route_schema()
    tools = s["properties"]["tool"]["enum"]
    assert "volume_control" in tools
    assert "power_control" in tools
    assert "playback_control" in tools
    assert "ai_picture_sound_control" in tools


def test_route_schema_has_audio_tools():
    s = build_route_schema()
    tools = s["properties"]["tool"]["enum"]
    assert "audio_search" in tools
    assert "audio_chat_qa" in tools


def test_route_schema_has_audio_intents():
    s = build_route_schema()
    intents = s["properties"]["intent"]["enum"]
    assert "audio_search" in intents
    assert "audio_play" in intents
    assert "device_control" in intents


# ============================================================
# Audio schema tests
# ============================================================

def test_audio_schema_shape():
    s = build_audio_schema()
    assert s["title"] == "audio_slot_fill"
    assert "tool" in s["properties"]
    assert "query" in s["properties"]
    assert "play_mode" in s["properties"]
    assert s["properties"]["play_mode"]["enum"] == ["search", "play", "screen_off_play"]


def test_compile_audio_search():
    slot = {"tool": "audio_search", "query": "郭德纲的相声", "play_mode": "search"}
    tool, params = compile_audio(slot)
    assert tool == "audio_search"
    assert params["query"] == "郭德纲的相声"
    assert params["play_mode"] == "search"


def test_compile_audio_play():
    slot = {"tool": "audio_search", "query": "三国演义评书",
            "play_mode": "play", "screen_mode": "normal"}
    tool, params = compile_audio(slot)
    assert tool == "audio_search"
    assert params["play_mode"] == "play"
    assert params["screen_mode"] == "normal"


def test_compile_audio_screen_off():
    slot = {"tool": "audio_search", "query": "睡前故事",
            "play_mode": "screen_off_play", "screen_mode": "screen_standby"}
    tool, params = compile_audio(slot)
    assert params["play_mode"] == "screen_off_play"
    assert params["screen_mode"] == "screen_standby"


def test_compile_audio_chat_qa():
    slot = {"tool": "audio_chat_qa", "query": "这本书的作者是谁", "play_mode": "search"}
    tool, params = compile_audio(slot)
    assert tool == "audio_chat_qa"
    assert params["query"] == "这本书的作者是谁"


def test_compile_audio_invalid_tool_fallback():
    slot = {"tool": "invalid_tool", "query": "test", "play_mode": "search"}
    tool, params = compile_audio(slot)
    assert tool == "audio_search"  # fallback


# ============================================================
# Device schema tests
# ============================================================

def test_device_schema_shape():
    s = build_device_schema()
    assert s["title"] == "device_slot_fill"
    assert "tool" in s["properties"]
    assert "operation" in s["properties"]
    assert "object" in s["properties"]
    assert "value" in s["properties"]
    assert len(s["properties"]["tool"]["enum"]) == 20


def test_compile_device_volume():
    slot = {"tool": "volume_control", "operation": "提高", "object": "音量"}
    tool, params = compile_device(slot)
    assert tool == "volume_control"
    assert params["operation"] == "提高"
    assert params["object"] == "音量"
    assert "value" not in params


def test_compile_device_volume_with_value():
    slot = {"tool": "volume_control", "operation": "设置", "object": "音量", "value": "30"}
    tool, params = compile_device(slot)
    assert tool == "volume_control"
    assert params["value"] == "30"


def test_compile_device_power():
    slot = {"tool": "power_control", "operation": "关闭", "object": "电源"}
    tool, params = compile_device(slot)
    assert tool == "power_control"
    assert params["operation"] == "关闭"


def test_compile_device_invalid_tool_fallback():
    slot = {"tool": "invalid", "operation": "打开", "object": "蓝牙"}
    tool, params = compile_device(slot)
    assert tool == "system_settings_control"  # fallback


# ============================================================
# Agent state machine tests
# ============================================================

def test_agent_routes_to_audio():
    agent = PlannerAgent("我想听郭德纲的相声")
    obs = agent.first_observation()
    assert "用户请求" in obs

    route_output = {"domain": "audio", "intent": "audio_search",
                    "tool": "audio_search", "confidence": 0.9}
    step = agent.observe(route_output)
    assert not step.done
    assert step.phase == "audio"
    assert agent.phase == "audio"


def test_agent_routes_to_device():
    agent = PlannerAgent("声音大一点")
    route_output = {"domain": "device", "intent": "device_control",
                    "tool": "volume_control", "confidence": 0.95}
    step = agent.observe(route_output)
    assert not step.done
    assert step.phase == "device"
    assert agent.phase == "device"


def test_agent_audio_slot_fill():
    agent = PlannerAgent("播放三国演义评书")
    # Route
    route_output = {"domain": "audio", "intent": "audio_play",
                    "tool": "audio_search", "confidence": 0.9}
    step = agent.observe(route_output)
    assert step.phase == "audio"

    # Slot-fill
    slot_output = {"tool": "audio_search", "query": "三国演义评书",
                   "play_mode": "play", "screen_mode": "normal"}
    step = agent.observe(slot_output)
    assert step.done
    assert step.phase == "audio_done"
    assert agent.slot_fill == slot_output


def test_agent_device_slot_fill():
    agent = PlannerAgent("把音量调到30")
    # Route
    route_output = {"domain": "device", "intent": "device_control",
                    "tool": "volume_control", "confidence": 0.95}
    step = agent.observe(route_output)
    assert step.phase == "device"

    # Slot-fill
    slot_output = {"tool": "volume_control", "operation": "设置",
                   "object": "音量", "value": "30"}
    step = agent.observe(slot_output)
    assert step.done
    assert step.phase == "device_done"
    assert agent.slot_fill == slot_output


def test_agent_simple_tool_ends_immediately():
    agent = PlannerAgent("给我推荐点好看的")
    route_output = {"domain": "vod", "intent": "personalized",
                    "tool": "vod_personalized_search", "confidence": 0.9}
    step = agent.observe(route_output)
    assert step.done
    assert step.phase == "route_end"
    assert step.info.get("reason") == "simple_slot_fill"


def test_agent_ir_flow_still_works():
    agent = PlannerAgent("刘德华的免费电影")
    route_output = {"domain": "vod", "intent": "search",
                    "tool": "vod_search", "confidence": 0.9}
    step = agent.observe(route_output)
    assert not step.done
    assert step.phase == "ir"

    ir_output = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "actor", "value": "刘德华"},
            {"field": "category", "value": "电影"},
            {"field": "fee", "value": 0}]}
    }
    step = agent.observe(ir_output)
    assert step.done
    assert step.phase == "ir_ok"


# ============================================================
# Harness end-to-end tests with stubs
# ============================================================

def _make_audio_stub():
    """Stub that routes to audio then produces slot-fill."""
    call_count = [0]

    def responder(messages, guided_json):
        title = (guided_json or {}).get("title", "")
        call_count[0] += 1
        if title == "planner_route":
            return json.dumps({"domain": "audio", "intent": "audio_search",
                               "tool": "audio_search", "confidence": 0.9})
        if title == "audio_slot_fill":
            return json.dumps({"tool": "audio_search", "query": "郭德纲的相声",
                               "play_mode": "search", "screen_mode": "normal"})
        return json.dumps({})
    return responder


def test_harness_audio_end_to_end():
    planner = Planner(VLLMClient(responder=_make_audio_stub()))
    res = planner.plan("我想听郭德纲的相声")
    assert res.tool_name == "audio_search"
    assert res.domain == "audio"
    assert res.parameters["query"] == "郭德纲的相声"
    assert res.parameters["play_mode"] == "search"


def _make_device_stub():
    """Stub that routes to device then produces slot-fill."""
    def responder(messages, guided_json):
        title = (guided_json or {}).get("title", "")
        if title == "planner_route":
            return json.dumps({"domain": "device", "intent": "device_control",
                               "tool": "volume_control", "confidence": 0.95})
        if title == "device_slot_fill":
            return json.dumps({"tool": "volume_control", "operation": "提高",
                               "object": "音量"})
        return json.dumps({})
    return responder


def test_harness_device_end_to_end():
    planner = Planner(VLLMClient(responder=_make_device_stub()))
    res = planner.plan("声音大一点")
    assert res.tool_name == "volume_control"
    assert res.domain == "device"
    assert res.parameters["operation"] == "提高"
    assert res.parameters["object"] == "音量"


def _make_simple_stub():
    """Stub that routes to a simple tool (vod_history)."""
    def responder(messages, guided_json):
        title = (guided_json or {}).get("title", "")
        if title == "planner_route":
            return json.dumps({"domain": "vod", "intent": "history",
                               "tool": "vod_history", "confidence": 0.9})
        return json.dumps({})
    return responder


def test_harness_simple_tool():
    planner = Planner(VLLMClient(responder=_make_simple_stub()))
    res = planner.plan("我最近看了什么")
    assert res.tool_name == "vod_history"
    assert res.domain == "vod"
    assert res.parameters == {}  # 暂时为空，待扩展


# ============================================================
# Runner
# ============================================================

def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
