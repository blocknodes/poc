#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planner 评测脚本。

输入：benchmark_vod.csv（由 tools/clean_vod_benchmark.py 生成）
逐条跑 Planner.plan(query)，与 expected_tool / expected_params 对比，分层打分：
  - 路由准确率  route_acc     ：路由原始选中的工具是否命中（忽略 flat->nested 回退）
  - 最终工具准确率 final_acc   ：最终落地工具是否 == expected_tool
  - 参数等价率  param_acc      ：仅统计有 golden expected_params 的行，做 JSON 语义等价
  - 结构合法率  valid_rate     ：未抛异常（路由+IR+校验+编译全流程跑通）的比例

连真实 vLLM（参考 demo.py --live）：
    VLLM_BASE_URL=http://host:8000/v1 VLLM_MODEL=your-30b-moe \
        python bench.py --input benchmark_vod.csv --live

冒烟（校验 benchmark 能加载、看分布，不连模型）：
    python bench.py --input benchmark_vod.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

from planner.harness import Planner
from planner.vllm_client import VLLMClient, VLLMConfig


# ----------------------------- 数据结构 -----------------------------
@dataclass
class Case:
    id: str
    domain: str
    query: str
    intent: str
    expected_tool: str
    expected_params: Optional[dict]  # 解析后的 golden 参数，可能为 None
    note: str


@dataclass
class RowResult:
    case: Case
    pred_tool: Optional[str] = None
    route_tool: Optional[str] = None       # 回退前路由选的工具
    pred_params: Optional[dict] = None
    ok: bool = False                       # 全流程未抛异常
    route_hit: bool = False
    final_hit: bool = False
    param_checked: bool = False
    param_hit: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0


# ----------------------------- 工具函数 -----------------------------
def load_cases(path: str) -> list[Case]:
    cases: list[Case] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("expected_params") or "").strip()
            try:
                params = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                params = None  # golden 本身非法则不参与参数打分
            cases.append(Case(
                id=row.get("id", ""),
                domain=row.get("domain", "vod"),
                query=row.get("query", "").strip(),
                intent=row.get("intent", "").strip(),
                expected_tool=row.get("expected_tool", "").strip(),
                expected_params=params,
                note=row.get("note", "").strip(),
            ))
    return cases


def _canon(obj: Any) -> Any:
    """归一化 JSON：dict 按 key 排序，list 内元素按序列化排序 —— 用于语义等价比较。"""
    if isinstance(obj, dict):
        return {k: _canon(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        canon_items = [_canon(x) for x in obj]
        return sorted(canon_items, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    return obj


def _unwrap_singletons(node: Any) -> Any:
    """把单元素的 and/or 拍平：{"and":[X]} -> X。

    gold 里常把单字段包成 `and[单]`，而模型可能直接输出裸字段；两者语义相同。
    与训练侧 reward 的指纹归一化保持一致，避免「内容全对但结构包裹不同」被误判为失败。
    """
    if isinstance(node, dict):
        for op in ("and", "or"):
            if op in node and isinstance(node[op], list):
                items = [_unwrap_singletons(x) for x in node[op]]
                return items[0] if len(items) == 1 else {op: items}
        return {k: _unwrap_singletons(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_unwrap_singletons(x) for x in node]
    return node


def params_equivalent(a: Any, b: Any) -> bool:
    return _canon(_unwrap_singletons(a)) == _canon(_unwrap_singletons(b))


# ----------------------------- 单条评测 -----------------------------
def run_case(planner: Planner, case: Case) -> RowResult:
    r = RowResult(case=case)
    t0 = time.perf_counter()
    try:
        res = planner.plan(case.query)
        r.ok = True
        r.pred_tool = res.tool_name
        r.route_tool = res.fallback_from or res.tool_name
        r.pred_params = res.parameters

        r.final_hit = (res.tool_name == case.expected_tool)
        r.route_hit = (r.route_tool == case.expected_tool)

        if case.expected_params is not None:
            r.param_checked = True
            r.param_hit = bool(res.parameters) and params_equivalent(res.parameters, case.expected_params)
    except Exception as e:  # 路由/IR/校验/编译任一环节失败
        r.error = f"{type(e).__name__}: {e}"
    finally:
        r.latency_ms = (time.perf_counter() - t0) * 1000
    return r


# ----------------------------- 汇总打印 -----------------------------
def summarize(results: list[RowResult]) -> dict:
    n = len(results)
    ok = sum(r.ok for r in results)
    route_hit = sum(r.route_hit for r in results)
    final_hit = sum(r.final_hit for r in results)
    param_total = sum(r.param_checked for r in results)
    param_hit = sum(r.param_hit for r in results)
    lat = sorted(r.latency_ms for r in results if r.ok)

    def pct(a, b):
        return f"{(a / b * 100):5.1f}%" if b else "  n/a"

    print("\n" + "=" * 60)
    print(f"总用例        : {n}")
    print(f"结构合法率    : {pct(ok, n)}  ({ok}/{n})  流程未抛异常")
    print(f"路由准确率    : {pct(route_hit, n)}  ({route_hit}/{n})  回退前工具命中")
    print(f"最终工具准确率: {pct(final_hit, n)}  ({final_hit}/{n})")
    print(f"参数等价率    : {pct(param_hit, param_total)}  ({param_hit}/{param_total})  仅统计有 golden 参数的行")
    if lat:
        p50 = lat[len(lat) // 2]
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        print(f"延迟          : p50={p50:.0f}ms  p95={p95:.0f}ms")

    # 按 expected_tool 分层
    by_tool: dict[str, list[RowResult]] = defaultdict(list)
    for r in results:
        by_tool[r.case.expected_tool].append(r)
    print("\n按 expected_tool 分层（路由命中 / 最终命中 / 参数命中）:")
    print(f"  {'tool':<32}{'N':>5}{'route':>9}{'final':>9}{'param':>9}")
    for tool in sorted(by_tool, key=lambda t: -len(by_tool[t])):
        rs = by_tool[tool]
        pt = sum(x.param_checked for x in rs)
        line = (f"  {tool:<32}{len(rs):>5}"
                f"{pct(sum(x.route_hit for x in rs), len(rs)):>9}"
                f"{pct(sum(x.final_hit for x in rs), len(rs)):>9}"
                f"{pct(sum(x.param_hit for x in rs), pt):>9}")
        print(line)
    print("=" * 60)

    return {
        "total": n, "valid": ok, "route_hit": route_hit,
        "final_hit": final_hit, "param_total": param_total, "param_hit": param_hit,
    }


def dump_details(results: list[RowResult], path: str) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "query", "expected_tool", "route_tool", "pred_tool",
                    "route_hit", "final_hit", "param_checked", "param_hit",
                    "error", "pred_params", "expected_params", "latency_ms"])
        for r in results:
            c = r.case
            w.writerow([
                c.id, c.query, c.expected_tool, r.route_tool or "", r.pred_tool or "",
                int(r.route_hit), int(r.final_hit), int(r.param_checked), int(r.param_hit),
                r.error or "",
                json.dumps(r.pred_params, ensure_ascii=False) if r.pred_params else "",
                json.dumps(c.expected_params, ensure_ascii=False) if c.expected_params else "",
                f"{r.latency_ms:.0f}",
            ])
    print(f"\n逐条明细已写入: {path}")


# ----------------------------- 实时统计 -----------------------------
class LiveStats:
    """线程安全的累计统计，随每条用例完成实时刷新到终端。"""

    def __init__(self, total: int):
        self.total = total
        self.lock = threading.Lock()
        self.done = 0
        self.valid = 0
        self.tool_hit = 0        # 最终工具命中
        self.route_hit = 0       # 回退前路由命中
        self.param_total = 0
        self.param_hit = 0
        self.overall_hit = 0     # 端到端：工具命中 且（无 golden 参数 或 参数命中）

    def update(self, r: "RowResult") -> None:
        with self.lock:
            self.done += 1
            self.valid += r.ok
            self.tool_hit += r.final_hit
            self.route_hit += r.route_hit
            if r.param_checked:
                self.param_total += 1
                self.param_hit += r.param_hit
            if r.final_hit and (not r.param_checked or r.param_hit):
                self.overall_hit += 1
            self._render()

    def _render(self) -> None:
        def pct(a, b):
            return f"{(a / b * 100):5.1f}%" if b else "  n/a"
        line = (f"\r[{self.done}/{self.total}] "
                f"tool={pct(self.tool_hit, self.done)} "
                f"route={pct(self.route_hit, self.done)} "
                f"param={pct(self.param_hit, self.param_total)} "
                f"overall={pct(self.overall_hit, self.done)} "
                f"valid={pct(self.valid, self.done)}")
        sys.stderr.write(line)
        sys.stderr.flush()


# ----------------------------- 入口 -----------------------------
def build_planner() -> Planner:
    cfg = VLLMConfig(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ.get("VLLM_MODEL", "qwen-30b-moe"),
    )
    # request_format 兼容老服务：VLLM_REQUEST_FORMAT=guided
    if os.environ.get("VLLM_REQUEST_FORMAT"):
        cfg.request_format = os.environ["VLLM_REQUEST_FORMAT"]
    return Planner(VLLMClient(cfg))


def main() -> int:
    ap = argparse.ArgumentParser(description="Planner benchmark runner")
    ap.add_argument("--input", "-i", default="benchmark_vod.csv", help="benchmark CSV 路径")
    ap.add_argument("--live", action="store_true", help="连接真实 vLLM（否则只做 dry-run）")
    ap.add_argument("--dry-run", action="store_true", help="只加载 benchmark 并打印分布，不跑模型")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--workers", type=int, default=1, help="并发请求数")
    ap.add_argument("--output", "-o", default="bench_results.csv", help="逐条明细输出路径")
    args = ap.parse_args()

    cases = load_cases(args.input)
    if args.limit > 0:
        cases = cases[: args.limit]
    print(f"加载 {len(cases)} 条用例 <- {args.input}")

    if args.dry_run or not args.live:
        dist = Counter(c.expected_tool for c in cases)
        withp = sum(c.expected_params is not None for c in cases)
        print("\nexpected_tool 分布:")
        for k, v in dist.most_common():
            print(f"  {k:<32}{v:>5}")
        print(f"带 golden 参数: {withp}/{len(cases)}")
        if not args.live:
            print("\n(未加 --live，仅 dry-run。加 --live 并设置 "
                  "VLLM_BASE_URL / VLLM_MODEL 连接模型评测)")
        return 0

    planner = build_planner()
    results: list[RowResult] = [None] * len(cases)  # type: ignore
    live = LiveStats(len(cases))

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_case, planner, c): i for i, c in enumerate(cases)}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                live.update(results[i])
    else:
        for i, c in enumerate(cases):
            results[i] = run_case(planner, c)
            live.update(results[i])
    sys.stderr.write("\n")
    sys.stderr.flush()

    summarize(results)
    dump_details(results, args.output)

    errs = [r for r in results if r.error]
    if errs:
        print(f"\n失败样例（前 5）:")
        for r in errs[:5]:
            print(f"  [{r.case.id}] {r.case.query[:30]} -> {r.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
