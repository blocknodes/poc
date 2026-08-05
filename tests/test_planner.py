"""planner 离线测试 —— 对齐 test_set 7.28 版本用例。

测试层次：
  1. registry 字段一致性
  2. IR 解析 + 校验
  3. nested 编译（vod_search / vod_search_all / vod_relate_search）
  4. flat 编译（best-effort）
  5. 工具选择（vod_search vs vod_search_all）
  6. agent 状态机
  7. 慢链路直传 query
"""
import json
import sys
sys.path.insert(0, ".")

from planner.registry import (
    FIELD_REGISTRY, VOD_SEARCH_FIELDS, VOD_RELATE_FIELDS,
    PLAYBACK_FIELDS, SORT_REGISTRY, get_field, field_names, Kind,
)
from planner.ir import parse_ir, validate_ir, IRError
from planner.compiler import (
    compile_nested, compile_relate, compile_flat_best_effort,
    compile_with_fallback, select_vod_search_tool, CompileError,
)
from planner.grammar import build_ir_schema, build_route_schema
from planner.agent import (
    PlannerAgent, IR_TOOLS, SIMPLE_TOOLS, SLOW_SEARCH_TOOLS,
    loads_lenient,
)


_pass = 0
_fail = 0


def _assert(cond, msg=""):
    global _pass, _fail
    if cond:
        _pass += 1
    else:
        _fail += 1
        frame = sys._getframe(1)
        print(f"  FAIL  [{frame.f_code.co_name}] {msg}")


# ===========================================================================
# 1. Registry 字段对齐 0725-v1 schema
# ===========================================================================
def test_nested_field_names_vod():
    """nested 后端的落地名应对齐 0725-v1 tool schema。"""
    # fee -> is_fee (not 'fee')
    _assert(get_field("fee").nested_name("vod") == "is_fee",
            "fee nested vod should be 'is_fee'")
    # age -> age_range
    _assert(get_field("age").nested_name("vod") == "age_range",
            "age nested vod should be 'age_range'")
    # release_year -> release_time
    _assert(get_field("release_year").nested_name("vod") == "release_time",
            "release_year nested vod should be 'release_time'")
    # rate -> rate
    _assert(get_field("rate").nested_name("vod") == "rate",
            "rate nested vod should be 'rate'")
    # is_over -> is_over
    _assert(get_field("is_over").nested_name("vod") == "is_over",
            "is_over nested vod should be 'is_over'")


def test_playback_fields():
    """playback 字段应含 series / video_index / voiceStartPos。"""
    _assert("series" in PLAYBACK_FIELDS, "series in playback")
    _assert("video_index" in PLAYBACK_FIELDS, "video_index in playback")
    _assert("voiceStartPos" in PLAYBACK_FIELDS, "voiceStartPos in playback")
    _assert("videoIndex" not in PLAYBACK_FIELDS, "videoIndex should NOT be in playback")


def test_vod_search_fields_subset():
    """vod_search 精简版字段集应为 10 个。"""
    _assert(len(VOD_SEARCH_FIELDS) == 10, f"expected 10, got {len(VOD_SEARCH_FIELDS)}")
    expected = {"title", "actor", "director", "entertainer", "prize",
                "role", "tag", "target", "definition", "category"}
    _assert(VOD_SEARCH_FIELDS == expected,
            f"mismatch: {VOD_SEARCH_FIELDS.symmetric_difference(expected)}")


# ===========================================================================
# 2. IR 解析 + 校验（对齐 test_set 用例）
# ===========================================================================
def test_ir_parse_play_with_playback():
    """test_set: '播放甄嬛传第二季第三集' → play, title + series + video_index"""
    ir_dict = {
        "domain": "vod", "action": "play",
        "query": {"field": "title", "value": "庆余年"},
        "playback": {"series": 2, "video_index": 3},
    }
    ir = parse_ir(ir_dict)
    _assert(ir.action == "play")
    _assert(ir.playback == {"series": 2, "video_index": 3})
    errs = validate_ir(ir)
    _assert(not errs, f"validate failed: {errs}")


def test_ir_validate_rejects_old_videoIndex():
    """旧字段 videoIndex 在新版应校验失败。"""
    ir_dict = {
        "domain": "vod", "action": "play",
        "query": {"field": "title", "value": "test"},
        "playback": {"videoIndex": 5},
    }
    ir = parse_ir(ir_dict)
    errs = validate_ir(ir)
    _assert(any("videoIndex" in e for e in errs),
            f"should reject videoIndex, got: {errs}")


def test_ir_validate_fee_status():
    """fee 字段 value 应为 0 或 1。"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "category", "value": "电视剧"},
            {"field": "fee", "value": 0},
        ]},
    }
    ir = parse_ir(ir_dict)
    errs = validate_ir(ir)
    _assert(not errs, f"validate failed: {errs}")


# ===========================================================================
# 3. nested 编译（对齐 test_set 期望参数）
# ===========================================================================
def test_compile_nested_play_with_video_index():
    """test_set: '放第1集哑巴新娘' → play, title=哑巴新娘, video_index=1"""
    ir_dict = {
        "domain": "vod", "action": "play",
        "query": {"field": "title", "value": "哑巴新娘"},
        "playback": {"video_index": 1},
    }
    ir = parse_ir(ir_dict)
    actual_tool, params = compile_with_fallback(ir, "vod_search", retext="放第1集哑巴新娘")
    _assert(actual_tool == "vod_search")
    _assert(params["action"] == "play")
    _assert(params["retext"] == "放第1集哑巴新娘")
    q = params["query"]["and"]
    _assert({"field": "title", "value": "哑巴新娘"} in q)
    _assert({"field": "video_index", "value": 1} in q)


def test_compile_nested_search_is_fee():
    """test_set: '免费的电视剧英雄泪' → search, title=英雄泪, category=电视剧, is_fee=0"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "title", "value": "英雄泪"},
            {"field": "category", "value": "电视剧"},
            {"field": "fee", "value": 0},
        ]},
    }
    ir = parse_ir(ir_dict)
    actual_tool, params = compile_with_fallback(ir, "vod_search", retext="免费的电视剧英雄泪")
    _assert(actual_tool == "vod_search")
    q = params["query"]["and"]
    _assert({"field": "is_fee", "value": 0} in q, f"expected is_fee=0 in {q}")
    _assert({"field": "category", "value": "电视剧"} in q)


def test_compile_nested_release_time():
    """test_set: '光线传媒2025年的电影' → release_time from/to"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "company", "value": "光线传媒"},
            {"field": "category", "value": "电影"},
            {"field": "release_year", "range": {"from": "20250101", "to": "20251231"}},
        ]},
    }
    ir = parse_ir(ir_dict)
    actual_tool, params = compile_with_fallback(ir, "vod_search", retext="光线传媒2025年有哪些电影")
    # company 不在 VOD_SEARCH_FIELDS → vod_search_all
    _assert(actual_tool == "vod_search_all",
            f"expected vod_search_all, got {actual_tool}")
    q = params["query"]["and"]
    _assert({"field": "release_time", "from": "20250101", "to": "20251231"} in q,
            f"expected release_time in {q}")


def test_compile_nested_vod_search_all():
    """test_set: 'BBC纪录片' → vod_search_all (company 字段)"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "company", "value": "BBC"},
            {"field": "category", "value": "纪录片"},
        ]},
    }
    ir = parse_ir(ir_dict)
    actual_tool, params = compile_with_fallback(ir, "vod_search", retext="BBC纪录片")
    _assert(actual_tool == "vod_search_all",
            f"company triggers search_all, got {actual_tool}")


def test_compile_nested_area_selects_search_all():
    """test_set: '我想看韩剧' → area=韩国, category=电视剧 → vod_search_all"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "area", "value": "韩国"},
            {"field": "category", "value": "电视剧"},
        ]},
    }
    ir = parse_ir(ir_dict)
    tool = select_vod_search_tool(ir)
    _assert(tool == "vod_search_all", f"area should trigger search_all, got {tool}")


def test_compile_nested_basic_selects_vod_search():
    """test_set: '播放大生意人' → title 字段 → vod_search（精简版够用）"""
    ir_dict = {
        "domain": "vod", "action": "play",
        "query": {"field": "title", "value": "大生意人"},
    }
    ir = parse_ir(ir_dict)
    tool = select_vod_search_tool(ir)
    _assert(tool == "vod_search", f"title-only should use vod_search, got {tool}")


# ===========================================================================
# 4. vod_relate_search 编译
# ===========================================================================
def test_compile_relate_search():
    """test_set: '和甄嬛传类似的电视剧' → vod_relate_search"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "title", "value": "甄嬛传"},
            {"field": "category", "value": "电视剧"},
        ]},
    }
    ir = parse_ir(ir_dict)
    actual_tool, params = compile_with_fallback(ir, "vod_relate_search", retext="和甄嬛传类似的电视剧")
    _assert(actual_tool == "vod_relate_search")
    _assert("query" in params)
    _assert("action" not in params, "relate should not have action")
    _assert("retext" not in params, "relate should not have retext")


# ===========================================================================
# 5. 慢链路直传 query
# ===========================================================================
def test_slow_search_passthrough():
    """vod_fuzzy_search 只传 query 原文。"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "title", "value": "射雕英雄传"},
            {"field": "tag", "value": "八三版"},
        ]},
    }
    ir = parse_ir(ir_dict)
    actual_tool, params = compile_with_fallback(
        ir, "vod_fuzzy_search", retext="射雕英雄传八三版的"
    )
    _assert(actual_tool == "vod_fuzzy_search")
    _assert(params == {"query": "射雕英雄传八三版的"},
            f"slow_search should only have query, got {params}")


# ===========================================================================
# 6. flat best-effort 编译
# ===========================================================================
def test_compile_flat_best_effort():
    """flat best-effort 应输出可表达的字段，跳过不可表达的。"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "actor", "values": ["刘德华", "吴京"], "op": "and"},
            {"field": "category", "value": "电影"},
            {"field": "fee", "value": 0},
            {"field": "release_year", "range": {"from": "20200101", "to": "*"}},
        ]},
        "sort": [{"key": "rate", "order": "desc"}],
    }
    ir = parse_ir(ir_dict)
    params = compile_flat_best_effort(ir)
    _assert(params.get("actor") == "刘德华 AND 吴京", f"actor={params.get('actor')}")
    _assert(params.get("category") == "电影")
    _assert(params.get("is_fee") == "0", f"is_fee={params.get('is_fee')}")
    _assert(params.get("date") == "20200101 TO *", f"date={params.get('date')}")
    _assert(params.get("rate") == "desc", f"rate={params.get('rate')}")


# ===========================================================================
# 7. Agent 状态机
# ===========================================================================
def test_agent_route_to_ir():
    """路由到 vod_search → 进入 IR 阶段。"""
    agent = PlannerAgent("刘德华的电影")
    step = agent.observe({"domain": "vod", "intent": "search",
                          "tool": "vod_search", "confidence": 0.95})
    _assert(not step.done)
    _assert(step.phase == "ir")
    _assert(agent.routed_tool == "vod_search")


def test_agent_route_to_slow_search():
    """路由到 vod_fuzzy_search → 直接结束（passthrough）。"""
    agent = PlannerAgent("骑着蓝色大鸟的人的电影")
    step = agent.observe({"domain": "vod", "intent": "slow_search",
                          "tool": "vod_fuzzy_search", "confidence": 0.9})
    _assert(step.done)
    _assert(step.info.get("reason") == "slow_search_passthrough")


def test_agent_route_to_relate():
    """路由到 vod_relate_search → 进入 IR 阶段。"""
    agent = PlannerAgent("类似流浪地球的电影")
    step = agent.observe({"domain": "vod", "intent": "relate",
                          "tool": "vod_relate_search", "confidence": 0.92})
    _assert(not step.done)
    _assert(step.phase == "ir")


def test_agent_route_to_personalized():
    """路由到 vod_personalized_search → 直接结束。"""
    agent = PlannerAgent("推荐我喜欢的电影")
    step = agent.observe({"domain": "vod", "intent": "personalized",
                          "tool": "vod_personalized_search", "confidence": 0.9})
    _assert(step.done)
    _assert(step.info.get("reason") == "simple_slot_fill")


def test_agent_route_to_history():
    """路由到 vod_history → 直接结束。"""
    agent = PlannerAgent("最近看过的电视剧")
    step = agent.observe({"domain": "vod", "intent": "history",
                          "tool": "vod_history", "confidence": 0.88})
    _assert(step.done)
    _assert(step.info.get("reason") == "simple_slot_fill")


def test_agent_ir_validate_and_repair():
    """IR 校验失败 → 自修复 → 再次提交合法 IR → 完成。"""
    agent = PlannerAgent("刘德华电影")
    # route
    agent.observe({"domain": "vod", "intent": "search",
                   "tool": "vod_search", "confidence": 0.95})
    # 第一次 IR 产出有错（field 不在 vod 域）
    bad_ir = {"domain": "vod", "query": {"field": "content_type", "value": "动画"}}
    step = agent.observe(bad_ir)
    _assert(not step.done)
    _assert(step.phase == "ir_repair")
    _assert(agent.repairs == 1)

    # 修复后产出正确 IR
    good_ir = {"domain": "vod", "query": {"field": "actor", "value": "刘德华"}}
    step = agent.observe(good_ir)
    _assert(step.done)
    _assert(step.phase == "ir_ok")


# ===========================================================================
# 8. Grammar schema 生成
# ===========================================================================
def test_ir_schema_vod_has_playback():
    """vod IR schema 应含 playback 字段（series/video_index/voiceStartPos）。"""
    schema = build_ir_schema("vod")
    pb = schema["properties"]["playback"]["properties"]
    _assert("series" in pb)
    _assert("video_index" in pb)
    _assert("voiceStartPos" in pb)


def test_route_schema_contains_new_tools():
    """路由 schema 应包含 0725-v1 新工具名。"""
    schema = build_route_schema()
    tools = schema["properties"]["tool"]["enum"]
    _assert("vod_search" in tools)
    _assert("vod_fuzzy_search" in tools)
    _assert("vod_relate_search" in tools)
    _assert("vod_personalized_search" in tools)
    _assert("vod_history" in tools)
    # 旧名应不存在
    _assert("vod_relate_recommend" not in tools,
            "old tool name vod_relate_recommend should be removed")
    _assert("vod_personalized_recommend" not in tools,
            "old tool name vod_personalized_recommend should be removed")
    _assert("vod_slow_search_data_search" not in tools,
            "old tool name vod_slow_search_data_search should be replaced by vod_fuzzy_search")


# ===========================================================================
# 9. 完整端到端编译对齐 test_set
# ===========================================================================
def test_e2e_play_series_video_index():
    """test_set: '我要看庆余年第二季第三集'"""
    ir_dict = {
        "domain": "vod", "action": "play",
        "query": {"field": "title", "value": "庆余年"},
        "playback": {"series": 2, "video_index": 3},
    }
    ir = parse_ir(ir_dict)
    actual_tool, params = compile_with_fallback(ir, "vod_search", retext="我要看庆余年第二季第三集")
    expected = {
        "action": "play",
        "retext": "我要看庆余年第二季第三集",
        "query": {"and": [
            {"field": "title", "value": "庆余年"},
            {"field": "series", "value": 2},
            {"field": "video_index", "value": 3},
        ]},
    }
    _assert(actual_tool == "vod_search")
    _assert(params["action"] == expected["action"])
    _assert(params["retext"] == expected["retext"])
    q = params["query"]["and"]
    for leaf in expected["query"]["and"]:
        _assert(leaf in q, f"missing leaf {leaf} in {q}")


def test_e2e_90s_hongkong_free_wuxia():
    """test_set: '90年代的香港免费武侠电视剧' → vod_search_all"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "area", "value": "香港"},
            {"field": "category", "value": "电视剧"},
            {"field": "tag", "value": "武侠"},
            {"field": "fee", "value": 0},
            {"field": "release_year", "range": {"from": "19900101", "to": "19991231"}},
        ]},
    }
    ir = parse_ir(ir_dict)
    errs = validate_ir(ir)
    _assert(not errs, f"validate: {errs}")
    actual_tool, params = compile_with_fallback(ir, "vod_search", retext="90年代的香港免费武侠电视剧")
    _assert(actual_tool == "vod_search_all",
            f"area triggers search_all, got {actual_tool}")
    q = params["query"]["and"]
    _assert({"field": "area", "value": "香港"} in q)
    _assert({"field": "is_fee", "value": 0} in q)
    _assert({"field": "release_time", "from": "19900101", "to": "19991231"} in q)


def test_e2e_not_logic():
    """test_set: '我想看电影，不要刘德华的，不要香港的，不要动作的' → NOT 节点"""
    ir_dict = {
        "domain": "vod", "action": "search",
        "query": {"and": [
            {"field": "category", "value": "电影"},
            {"not": {"field": "actor", "value": "刘德华"}},
            {"not": {"field": "area", "value": "香港"}},
            {"not": {"field": "tag", "value": "动作"}},
        ]},
    }
    ir = parse_ir(ir_dict)
    errs = validate_ir(ir)
    _assert(not errs, f"validate: {errs}")
    actual_tool, params = compile_with_fallback(ir, "vod_search", retext="我想看电影，不要刘德华的")
    _assert(actual_tool == "vod_search_all")  # area 触发 search_all
    q = params["query"]["and"]
    _assert({"not": {"field": "actor", "value": "刘德华"}} in q)
    _assert({"not": {"field": "area", "value": "香港"}} in q)


# ===========================================================================
# Runner
# ===========================================================================
def _run_all():
    global _pass, _fail
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            _fail += 1
            print(f"  ERR {fn.__name__}: {e}")
    print(f"\n{'='*40}")
    print(f"PASSED: {_pass}  FAILED: {_fail}")
    if _fail:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
