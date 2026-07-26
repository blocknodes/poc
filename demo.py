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


class DebugVLLMClient(VLLMClient):
    """包装 VLLMClient，在每次 complete_json 前后打印完整的 LLM 输入与输出。"""

    _call_idx = 0

    def complete_json(self, messages, guided_json=None, *, max_tokens=None):
        DebugVLLMClient._call_idx += 1
        idx = DebugVLLMClient._call_idx
        print(f"\n{'='*60}")
        print(f"[DEBUG] LLM call #{idx}")
        print(f"{'='*60}")
        print(f"\n--- INPUT messages ({len(messages)} 条) ---")
        for i, m in enumerate(messages):
            role = m["role"]
            content = m["content"]
            print(f"  [{i}] role={role}")
            print(f"      {content}")
        if guided_json:
            print(f"\n--- guided_json schema ---")
            print(json.dumps(guided_json, ensure_ascii=False, indent=2))
        print(f"\n--- calling LLM ... ---")

        result = super().complete_json(messages, guided_json, max_tokens=max_tokens)

        print(f"\n--- OUTPUT (call #{idx}) ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"{'='*60}\n")
        return result


def demo_live(debug=False, query=None):
    cfg = VLLMConfig(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ.get("VLLM_MODEL", "qwen-30b-moe"),
    )
    client = DebugVLLMClient(cfg) if debug else VLLMClient(cfg)
    planner = Planner(client)
    queries = [query] if query else [
        "我想看电影，不要刘德华的，不要香港的，不要动作的",
        "我想看刘德华的免费电影",
        "适合3到6岁女孩看的科普动画",
        "最近有什么好看的科幻片，评分高的",
    ]
    for q in queries:
        res = planner.plan(q)
        _p(f"query: {q}", {
            "tool": res.tool_name, "domain": res.domain,
            "repairs": res.repairs, "fallback_from": res.fallback_from,
            "parameters": res.parameters, "notes": res.notes,
        })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="连接真实 vLLM 接口")
    ap.add_argument("--debug", action="store_true", help="打印每次 LLM 调用的完整输入 messages 与输出（需配合 --live）")
    ap.add_argument("--query", type=str, default=None, help="单条 query，不指定则跑默认批量用例")
    args = ap.parse_args()
    if args.live:
        demo_live(debug=args.debug, query=args.query)
    else:
        if args.debug or args.query:
            print("[WARN] --debug/--query 仅在 --live 模式下生效（离线 compiler demo 不走 LLM）")
        demo_compiler()
