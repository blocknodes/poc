"""可运行 demo：展示一份 IR 如何编译到三个目标；以及连真实 vLLM 的用法。

离线运行（不连模型）：  python demo.py
连真实 vLLM：           设置环境变量后 python demo.py --live
    VLLM_BASE_URL=http://host:8000/v1  VLLM_MODEL=your-30b-moe
"""
import argparse
import json
import os

from planner.compiler import can_compile_flat, compile_nested, compile_with_fallback
from planner.ir import And, Not, SortItem, IR, leaf
from planner.harness import Planner
from planner.vllm_client import VLLMClient, VLLMConfig


def _p(title, obj):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def demo_compiler():
    # “刘德华和吴京都演、非恐怖、2020年后、免费、按评分降序”
    ir = IR(domain="vod", query=And([
        leaf("actor", values=["刘德华", "吴京"], op="and"),
        leaf("category", value="电影"),
        Not(leaf("tag", value="恐怖")),
        leaf("release_year", range={"from": "20200101", "to": "*"}),
        leaf("fee", value=0),
    ]), sort=[SortItem("rate", "desc")])

    _p("一份 IR", {"query": "见代码", "note": "domain=vod"})
    _p("编译 -> vod_search (nested)", compile_nested(ir))
    tool, params = compile_with_fallback(ir, "vod_slow_search_data_search")
    _p(f"编译 -> {tool} (flat)", params)

    # 演示 flat 不可编译 -> 回退
    from planner.ir import Or
    ir2 = IR(domain="vod", query=Or([leaf("actor", value="刘德华"),
                                     leaf("director", value="张艺谋")]))
    ok, reason = can_compile_flat(ir2)
    print(f"\n跨字段 OR 可编译到 flat? {ok}  ({reason})")
    tool2, params2 = compile_with_fallback(ir2, "vod_slow_search_data_search")
    _p(f"自动回退 -> {tool2}", params2)


def demo_live():
    cfg = VLLMConfig(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ.get("VLLM_MODEL", "qwen-30b-moe"),
    )
    planner = Planner(VLLMClient(cfg))
    for q in ["我想看刘德华的免费电影", "适合3到6岁女孩看的科普动画", "最近有什么好看的科幻片，评分高的"]:
        res = planner.plan(q)
        _p(f"query: {q}", {
            "tool": res.tool_name, "domain": res.domain,
            "repairs": res.repairs, "fallback_from": res.fallback_from,
            "parameters": res.parameters, "notes": res.notes,
        })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="连接真实 vLLM 接口")
    args = ap.parse_args()
    if args.live:
        demo_live()
    else:
        demo_compiler()
