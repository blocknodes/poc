"""离线测试：registry / ir / compiler / grammar / harness(stub)。

运行：  python -m pytest tests/ -q      （或直接 python tests/test_planner.py）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.compiler import (  # noqa: E402
    CompileError, can_compile_flat, compile_flat, compile_nested,
    compile_with_fallback,
)
from planner.grammar import build_ir_schema, build_route_schema  # noqa: E402
from planner.ir import IR, And, Not, Or, SortItem, leaf, parse_ir, validate_ir  # noqa: E402
from planner.registry import field_names, Kind  # noqa: E402
from planner.harness import Planner  # noqa: E402
from planner.vllm_client import VLLMClient  # noqa: E402


def test_registry_counts():
    # 影视 24 精确维度、少儿 16 精确维度
    assert len(field_names("vod", Kind.EXACT)) == 24, field_names("vod", Kind.EXACT)
    assert len(field_names("educ", Kind.EXACT)) == 16, field_names("educ", Kind.EXACT)


def test_ir_validate_ok():
    ir = IR(domain="vod", query=And([
        leaf("actor", values=["刘德华", "吴京"], op="and"),
        leaf("category", value="电影"),
        Not(leaf("tag", value="恐怖")),
        leaf("rate", range={"from": 8, "to": "*"}),
        leaf("fee", value=0),
    ]), sort=[SortItem("rate", "desc")])
    assert validate_ir(ir) == []


def test_ir_validate_catches_errors():
    ir = IR(domain="educ", query=And([
        leaf("actor", value="刘德华"),          # actor 不在 educ
        leaf("fee", value=3),                    # 状态必须 0/1
        leaf("release_year", range={"from": "2020", "to": "*"}),  # 非 yyyyMMdd
    ]))
    errs = validate_ir(ir)
    assert any("actor" in e for e in errs)
    assert any("fee" in e for e in errs)
    assert any("release_year" in e for e in errs)


def test_compile_nested_vod():
    ir = IR(domain="vod", query=And([
        leaf("actor", values=["刘德华", "吴京"], op="and"),
        leaf("category", value="电影"),
        Not(leaf("tag", value="恐怖")),
        leaf("release_year", range={"from": "20200101", "to": "*"}),
        leaf("fee", value=0),
    ]), sort=[SortItem("rate", "desc")])
    params = compile_nested(ir)
    q = params["query"]["and"]
    assert {"field": "actor", "values": ["刘德华", "吴京"], "operator": "and"} in q
    assert {"field": "category", "value": "电影"} in q
    assert {"not": {"field": "tag", "value": "恐怖"}} in q
    assert {"field": "release_year", "from": "20200101", "to": "*"} in q
    assert {"field": "fee", "value": 0} in q          # vod 用 fee
    assert params["sort"] == {"rate": {"order": "desc"}}


def test_compile_nested_educ_fee_maps_to_is_fee():
    ir = IR(domain="educ", query=leaf("fee", value=1))
    params = compile_nested(ir)
    assert params["query"] == {"field": "is_fee", "value": 1}   # educ 映射 is_fee


def test_compile_flat_vod():
    ir = IR(domain="vod", query=And([
        leaf("actor", values=["刘德华", "吴京"], op="and"),
        leaf("category", value="电影"),
        Not(leaf("tag", value="恐怖")),
        leaf("release_year", range={"from": "20200101", "to": "*"}),
        leaf("fee", value=0),
    ]), sort=[SortItem("rate", "desc")])
    params = compile_flat(ir)
    assert params["actor"] == "刘德华 AND 吴京"
    assert params["category"] == "电影"
    assert params["tag"] == "NOT 恐怖"
    assert params["date"] == "20200101 TO *"
    assert params["is_fee"] == "0"
    assert params["rate"] == "desc"


def test_flat_rejects_cross_field_or():
    ir = IR(domain="vod", query=Or([
        leaf("actor", value="刘德华"),
        leaf("director", value="张艺谋"),
    ]))
    ok, reason = can_compile_flat(ir)
    assert not ok and "OR" in reason


def test_flat_rejects_educ_release_year():
    # educ 慢链路没有日期字段
    ir = IR(domain="educ", query=leaf("release_year", range={"from": "20200101", "to": "*"}))
    ok, _ = can_compile_flat(ir)
    assert not ok


def test_compile_with_fallback_keeps_slow_tool_best_effort():
    # 慢链路是语义后端：跨字段 OR 无法无损扁平化时，不再回退到精确检索，
    # 而是保持慢链路工具并 best-effort 摊平可表达的单字段叶子（布尔关系交还语义层）。
    ir = IR(domain="vod", query=Or([
        leaf("actor", value="刘德华"),
        leaf("director", value="张艺谋"),
    ]))
    tool, params = compile_with_fallback(ir, "vod_slow_search_data_search")
    assert tool == "vod_slow_search_data_search"     # 保持慢链路，不回退
    assert params.get("actor") == "刘德华"
    assert params.get("director") == "张艺谋"


def test_compile_flat_best_effort_drops_nonflat_fields():
    # title 无 flat 落地名 -> 跳过（交还原始 query 语义层），保持慢链路工具不回退。
    from planner.compiler import compile_flat_best_effort
    ir = IR(domain="vod", query=And([
        leaf("title", value="评书隋唐演义"),
        leaf("category", value="综艺"),
    ]))
    tool, params = compile_with_fallback(ir, "vod_slow_search_data_search")
    assert tool == "vod_slow_search_data_search"
    assert "title" not in params                     # 无 flat 落地名，已跳过
    assert params.get("category") == "综艺"


def test_grammar_schema_shape():
    s = build_ir_schema("vod")
    assert s["properties"]["domain"] == {"const": "vod"}
    assert "playback" in s["properties"]            # vod 有播放控制
    educ = build_ir_schema("educ")
    assert "playback" not in educ["properties"]     # educ 无
    # educ 的 leaf 精确枚举里不应含 actor
    exact_variant = educ["$defs"]["Leaf"]["oneOf"][0]
    assert "actor" not in exact_variant["properties"]["field"]["enum"]


def test_route_schema():
    s = build_route_schema()
    assert "vod_search" in s["properties"]["tool"]["enum"]
    assert "educ_search" in s["properties"]["tool"]["enum"]


# ---- 端到端：用 stub responder 模拟 vLLM（约束解码后模型输出合法 JSON）----
def _make_stub():
    def responder(messages, guided_json):
        # 依据 schema title 判定当前是路由还是 IR 阶段
        title = (guided_json or {}).get("title", "")
        if title == "planner_route":
            return json.dumps({"domain": "vod", "intent": "search",
                               "tool": "vod_search", "confidence": 0.9})
        # IR 阶段
        return json.dumps({
            "domain": "vod", "action": "search",
            "query": {"and": [
                {"field": "actor", "value": "刘德华"},
                {"field": "category", "value": "电影"},
                {"field": "fee", "value": 0}]},
        }, ensure_ascii=False)
    return responder


def test_harness_end_to_end():
    planner = Planner(VLLMClient(responder=_make_stub()))
    res = planner.plan("我想看刘德华的免费电影")
    assert res.tool_name == "vod_search"
    assert res.domain == "vod"
    q = res.parameters["query"]["and"]
    assert {"field": "actor", "value": "刘德华"} in q
    assert {"field": "fee", "value": 0} in q
    assert res.repairs == 0


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
