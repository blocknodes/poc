#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 planner 的 benchmark CSV 转成 **SFT（监督微调）** 用的多轮对话 jsonl。

与 build_dataset.py（GRPO gym-env）的区别：SFT 需要把每一轮 assistant 的
**gold 答案**都填好（route JSON + IR JSON），让模型模仿；GRPO 数据只有 query +
gold 列、assistant 内容由 rollout 现场生成。

训推一致（关键）：system / user 文本**完全复用** planner.prompts 的
`route_system_prompt / route_observation / ir_observation`——和推理侧 harness、
以及 GRPO rollout 走的是同一份 prompt，逐 token 一致。SFT 只是把 assistant 轮
换成 gold。

轨迹形态（与推理/GRPO 一致）::

    system : route_system_prompt()          (ROUTE_SYSTEM + 路由 few-shot)
    user   : route_observation(query)
    asst   : gold 路由 JSON  {"domain","intent","tool","confidence"}
    [user  : ir_observation(query, domain)   (仅检索类且有可验证 gold IR 时)
     asst  : gold IR JSON]

gold IR 由 `gold_params`（编译后的后端形态）**反编译 + 编译回填验证**得到：
只有「反编译出的 IR 能 parse/validate、且用 gold_tool 编译回去与 gold_params
严格等价」的样本才补 IR 轮；否则只监督 route 轮（degrade 到 route-only，不产出
不可信的 IR target）。这样 SFT 数据 100% 可证明正确。

覆盖策略：
  * 所有行都监督 **route**（路由是主指标，~85%）。
  * 仅带 golden 参数、且反编译回填验证通过的行额外监督 **IR**。
  * 非检索类工具（clip/recommend/history 等，不在 IR_TOOLS）天然 route-only。
  * 慢链路（*_slow_search）编译走 flat 后端，与 nested 形态 gold 不等价 → 自动
    degrade 成 route-only（与 bench 对慢链路 param 不度量一致）。

用法：
    python build_sft_dataset.py -i ../benchmark_vod.csv -o data_sft --val-ratio 0.1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Any, Optional

# 复用同仓库 planner 包（train/ 上一级即 poc 根目录）
_POC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _POC_ROOT not in sys.path:
    sys.path.insert(0, _POC_ROOT)

from planner.agent import IR_TOOLS  # noqa: E402
from planner.compiler import compile_with_fallback  # noqa: E402
from planner.ir import parse_ir, validate_ir  # noqa: E402
from planner.prompts import (  # noqa: E402
    ir_observation,
    route_observation,
    route_system_prompt,
)
from planner.registry import FIELD_REGISTRY, SORT_REGISTRY  # noqa: E402


# ---------------------------------------------------------------------------
# route 目标：{"domain","intent","tool","confidence"}（与 prompts._r few-shot 对齐）
# ---------------------------------------------------------------------------
_INTENT_BY_TOOL = {
    "vod_search": "search",
    "educ_search": "search",
    "vod_slow_search_data_search": "slow_search",
    "educ_slow_search_data_search": "slow_search",
    "vod_clip_search": "clip",
    "vod_personalized_recommend": "personalized",
    "vod_relate_recommend": "relate",
    "vod_history": "history",
}


def _route_target(domain: str, tool: str) -> dict:
    intent = _INTENT_BY_TOOL.get(tool)
    if intent is None and "_" in tool:
        intent = tool.split("_", 1)[1]  # 兜底：去掉 domain 前缀
    return {"domain": domain, "intent": intent, "tool": tool, "confidence": 1.0}


# ---------------------------------------------------------------------------
# 参数等价（与 bench.params_equivalent 对齐：canon + 单元素 and/or 拍平）
# ---------------------------------------------------------------------------
def _canon(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _canon(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        items = [_canon(x) for x in obj]
        return sorted(items, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    return obj


def _unwrap_singletons(node: Any) -> Any:
    if isinstance(node, dict):
        for op in ("and", "or"):
            if op in node and isinstance(node[op], list):
                items = [_unwrap_singletons(x) for x in node[op]]
                return items[0] if len(items) == 1 else {op: items}
        return {k: _unwrap_singletons(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_unwrap_singletons(x) for x in node]
    return node


def _params_equivalent(a: Any, b: Any) -> bool:
    return _canon(_unwrap_singletons(a)) == _canon(_unwrap_singletons(b))


# ---------------------------------------------------------------------------
# 反编译：gold_params（后端 nested 形态）-> IR（逻辑字段名 + IR mini-DSL）
# ---------------------------------------------------------------------------
def _nested_field_reverse(domain: str) -> dict:
    """后端字段名 -> FieldSpec 的反向表（按 domain）。"""
    rev = {}
    for spec in FIELD_REGISTRY.values():
        nm = spec.nested_name(domain)
        if nm:
            rev[nm] = spec
    return rev


_SORT_NESTED_REVERSE = {s.nested_key: k for k, s in SORT_REGISTRY.items()}


def _decompile_node(node: Any, rev: dict) -> dict:
    if not isinstance(node, dict):
        raise ValueError("节点必须是对象")
    if "and" in node:
        return {"and": [_decompile_node(x, rev) for x in node["and"]]}
    if "or" in node:
        return {"or": [_decompile_node(x, rev) for x in node["or"]]}
    if "not" in node:
        return {"not": _decompile_node(node["not"], rev)}
    if "field" in node:
        spec = rev.get(node["field"])
        if spec is None:
            raise ValueError(f"未知后端字段 {node['field']!r}")
        canon = spec.canonical
        if "from" in node or "to" in node:          # RANGE leaf
            return {"field": canon, "range": {"from": node.get("from"), "to": node.get("to")}}
        if "values" in node:                          # EXACT 多值
            leaf = {"field": canon, "values": list(node["values"])}
            op = node.get("operator")
            if op and op != "or":
                leaf["op"] = op
            return leaf
        return {"field": canon, "value": node.get("value")}  # EXACT/STATUS 单值
    raise ValueError(f"无法识别的节点: {node}")


def _decompile_params_to_ir(domain: str, params: dict) -> dict:
    rev = _nested_field_reverse(domain)
    ir: dict[str, Any] = {"domain": domain, "action": "search",
                          "query": _decompile_node(params.get("query"), rev)}
    sort = params.get("sort")
    if isinstance(sort, dict) and sort:
        items = []
        for nk, spec in sort.items():
            key = _SORT_NESTED_REVERSE.get(nk)
            if key is None:
                raise ValueError(f"未知排序键 {nk!r}")
            items.append({"key": key, "order": (spec or {}).get("order", "desc")})
        ir["sort"] = items
    return ir


def build_gold_ir(domain: str, tool: str, params: dict) -> Optional[dict]:
    """反编译 + 编译回填验证。通过返回 gold IR，否则 None（该行 degrade 成 route-only）。"""
    try:
        ir = _decompile_params_to_ir(domain, params)
        obj = parse_ir(ir)
        if validate_ir(obj):
            return None
        _, compiled = compile_with_fallback(obj, tool)
    except Exception:
        return None
    return ir if _params_equivalent(compiled, params) else None


# ---------------------------------------------------------------------------
# 组装 SFT 样本
# ---------------------------------------------------------------------------
def build_sample(domain: str, query: str, tool: str, gp_raw: str) -> tuple[dict, bool]:
    msgs = [
        {"role": "system", "content": route_system_prompt()},
        {"role": "user", "content": route_observation(query)},
        {"role": "assistant", "content": json.dumps(_route_target(domain, tool), ensure_ascii=False)},
    ]
    has_ir = False
    if tool in IR_TOOLS and gp_raw:
        try:
            params = json.loads(gp_raw)
        except json.JSONDecodeError:
            params = None
        gold_ir = build_gold_ir(domain, tool, params) if isinstance(params, dict) else None
        if gold_ir is not None:
            msgs.append({"role": "user", "content": ir_observation(query, domain)})
            msgs.append({"role": "assistant", "content": json.dumps(gold_ir, ensure_ascii=False)})
            has_ir = True
    return {"messages": msgs}, has_ir


def load_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            query = (r.get("query") or "").strip()
            tool = (r.get("expected_tool") or "").strip()
            if not query or not tool:
                continue
            gp = (r.get("expected_params") or "").strip()
            rows.append({
                "domain": (r.get("domain") or "vod").strip(),
                "query": query,
                "expected_tool": tool,
                "expected_params": gp,
            })
    return rows


def write_jsonl(path: str, samples: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build SFT dataset for planner (route + verified IR)")
    ap.add_argument("--input", "-i", default="../benchmark_vod.csv", help="benchmark CSV 路径")
    ap.add_argument("--out-dir", "-o", default="data_sft", help="输出目录")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_rows(args.input)
    random.Random(args.seed).shuffle(rows)

    samples, n_ir = [], 0
    for r in rows:
        s, has_ir = build_sample(r["domain"], r["query"], r["expected_tool"], r["expected_params"])
        samples.append(s)
        n_ir += int(has_ir)

    n_val = int(len(samples) * args.val_ratio)
    val, train = samples[:n_val], samples[n_val:]

    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    print(f"train: {len(train)} -> {os.path.join(args.out_dir, 'train.jsonl')}")
    if val:
        write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
        print(f"val  : {len(val)} -> {os.path.join(args.out_dir, 'val.jsonl')}")

    # 覆盖率速览
    withp = sum(1 for r in rows if r["expected_params"] and r["expected_tool"] in IR_TOOLS)
    print(f"\n样本总数        : {len(samples)}（全部监督 route）")
    print(f"带 IR 监督轮     : {n_ir}")
    print(f"有 golden 且检索类: {withp}  -> 反编译回填验证通过 {n_ir} / 失败 {withp - n_ir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
