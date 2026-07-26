#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练侧 benchmark —— 评测训练产出 checkpoint 的路由/参数指标。

数据源：train/data/val.jsonl（与 GRPO 训练同源），格式：
  {"query": "...", "gold_tool": "vod_search", "gold_params": "{...}", "gold_domain": "vod", ...}

也支持 benchmark_vod.csv（与 poc/bench.py 相同格式）。

用法：
  # 连已部署的 vLLM（与 bench.py 一致）
  VLLM_BASE_URL=http://host:8000/v1 VLLM_MODEL=planner-35b-a3b \
      python benchmark.py --input data/val.jsonl --live

  # 自动从 checkpoint 起临时 vLLM 服务、跑评测、再关服务
  python benchmark.py --checkpoint megatron_output/v10-.../checkpoint-140-merged --auto-serve

  # 批量评测多个 checkpoint（按 step 排序输出对比表）
  python benchmark.py --checkpoint-dir megatron_output/v10-... --auto-serve

  # 只跑路由（route_only 模式，与训练一致，快速验证路由能力）
  python benchmark.py --input data/val.jsonl --live --route-only

指标（与 poc/bench.py 对齐）：
  - route_acc   : 路由命中率（回退前工具是否 == gold_tool）
  - final_acc   : 最终落地工具准确率
  - param_acc   : 参数等价率（仅统计有 gold_params 的行）
  - valid_rate  : 全流程未抛异常的比例
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

# 让 import planner 包生效
_POC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _POC_ROOT not in sys.path:
    sys.path.insert(0, _POC_ROOT)

from planner.harness import Planner
from planner.vllm_client import VLLMClient, VLLMConfig


# ===========================================================================
# 数据加载
# ===========================================================================
@dataclass
class Case:
    id: str
    domain: str
    query: str
    gold_tool: str
    gold_params: Optional[dict]


def load_jsonl(path: str) -> list[Case]:
    """加载 train/data/val.jsonl 格式。"""
    cases = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            raw_params = row.get("gold_params", "")
            params = None
            if raw_params:
                if isinstance(raw_params, dict):
                    params = raw_params
                elif isinstance(raw_params, str) and raw_params.strip():
                    try:
                        params = json.loads(raw_params)
                    except json.JSONDecodeError:
                        pass
            cases.append(Case(
                id=str(i),
                domain=row.get("gold_domain", row.get("domain", "vod")),
                query=row.get("query", ""),
                gold_tool=row.get("gold_tool", row.get("expected_tool", "")),
                gold_params=params,
            ))
    return cases


def load_csv(path: str) -> list[Case]:
    """加载 benchmark_vod.csv 格式（兼容 poc/bench.py）。"""
    import csv
    cases = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("expected_params") or "").strip()
            try:
                params = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                params = None
            cases.append(Case(
                id=row.get("id", ""),
                domain=row.get("domain", "vod"),
                query=row.get("query", "").strip(),
                gold_tool=row.get("expected_tool", "").strip(),
                gold_params=params,
            ))
    return cases


def load_cases(path: str) -> list[Case]:
    if path.endswith(".csv"):
        return load_csv(path)
    return load_jsonl(path)


# ===========================================================================
# 等价比较（与 planner_plugin.py / bench.py 对齐）
# ===========================================================================
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


def params_equivalent(a: Any, b: Any) -> bool:
    return _canon(_unwrap_singletons(a)) == _canon(_unwrap_singletons(b))


# ===========================================================================
# 单条评测
# ===========================================================================
@dataclass
class Result:
    case: Case
    ok: bool = False
    route_tool: Optional[str] = None
    pred_tool: Optional[str] = None
    pred_params: Optional[dict] = None
    route_hit: bool = False
    final_hit: bool = False
    param_checked: bool = False
    param_hit: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0


def run_case(planner: Planner, case: Case) -> Result:
    r = Result(case=case)
    t0 = time.perf_counter()
    try:
        res = planner.plan(case.query)
        r.ok = True
        r.pred_tool = res.tool_name
        r.route_tool = res.fallback_from or res.tool_name
        r.pred_params = res.parameters
        r.route_hit = (r.route_tool == case.gold_tool)
        r.final_hit = (res.tool_name == case.gold_tool)
        if case.gold_params is not None:
            r.param_checked = True
            r.param_hit = bool(res.parameters) and params_equivalent(res.parameters, case.gold_params)
    except Exception as e:
        r.error = f"{type(e).__name__}: {e}"
    finally:
        r.latency_ms = (time.perf_counter() - t0) * 1000
    return r


# ===========================================================================
# 汇总
# ===========================================================================
def pct(a: int, b: int) -> str:
    return f"{(a / b * 100):5.1f}%" if b else "  n/a"


def summarize(results: list[Result], label: str = "") -> dict:
    n = len(results)
    ok = sum(r.ok for r in results)
    route_hit = sum(r.route_hit for r in results)
    final_hit = sum(r.final_hit for r in results)
    param_total = sum(r.param_checked for r in results)
    param_hit = sum(r.param_hit for r in results)

    header = f"  [{label}]" if label else ""
    print(f"\n{'='*60}{header}")
    print(f"总用例        : {n}")
    print(f"结构合法率    : {pct(ok, n)}  ({ok}/{n})")
    print(f"路由准确率    : {pct(route_hit, n)}  ({route_hit}/{n})")
    print(f"最终工具准确率: {pct(final_hit, n)}  ({final_hit}/{n})")
    print(f"参数等价率    : {pct(param_hit, param_total)}  ({param_hit}/{param_total})")

    # 按 gold_tool 分层
    by_tool: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_tool[r.case.gold_tool].append(r)
    print(f"\n  {'tool':<32}{'N':>5}{'route':>9}{'final':>9}{'param':>9}")
    for tool in sorted(by_tool, key=lambda t: -len(by_tool[t])):
        rs = by_tool[tool]
        pt = sum(x.param_checked for x in rs)
        print(f"  {tool:<32}{len(rs):>5}"
              f"{pct(sum(x.route_hit for x in rs), len(rs)):>9}"
              f"{pct(sum(x.final_hit for x in rs), len(rs)):>9}"
              f"{pct(sum(x.param_hit for x in rs), pt):>9}")
    print("=" * 60)

    # 打印前 5 条错误
    errs = [r for r in results if r.error]
    if errs:
        print(f"  失败样例（前 5）:")
        for r in errs[:5]:
            print(f"    [{r.case.id}] {r.case.query[:40]} -> {r.error[:80]}")

    return {
        "label": label, "total": n, "valid": ok,
        "route_acc": route_hit / n if n else 0,
        "final_acc": final_hit / n if n else 0,
        "param_acc": param_hit / param_total if param_total else 0,
    }


# ===========================================================================
# 自动起/停 vLLM 服务
# ===========================================================================
def start_vllm_server(model_path: str, port: int = 8100, gpus: str = "0,1,2,3",
                      tp: Optional[int] = None) -> subprocess.Popen:
    """后台起 vLLM server，返回 Popen 对象。"""
    if tp is None:
        tp = len(gpus.split(","))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    cmd = [
        "vllm", "serve", model_path,
        "--served-model-name", "bench-model",
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--enable-expert-parallel",
    ]
    print(f">> 启动 vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # 等待服务就绪
    _wait_for_server(f"http://127.0.0.1:{port}/v1/models", timeout=300)
    return proc


def _wait_for_server(url: str, timeout: float = 300):
    """轮询等待 vLLM 服务就绪。"""
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f">> vLLM 服务就绪 ({time.time()-t0:.0f}s)")
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM 服务 {url} 在 {timeout}s 内未就绪")


def stop_server(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(">> vLLM 服务已停止")


# ===========================================================================
# 批量 checkpoint 发现
# ===========================================================================
def find_merged_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    """在 ckpt_dir 下找所有 checkpoint-*-merged 目录，返回 [(step, path)] 按 step 排序。"""
    import re
    results = []
    for entry in os.listdir(ckpt_dir):
        m = re.match(r"checkpoint-(\d+)-merged$", entry)
        if m:
            step = int(m.group(1))
            full = os.path.join(ckpt_dir, entry)
            if os.path.isdir(full):
                results.append((step, full))
    results.sort()
    return results


# ===========================================================================
# 入口
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="训练侧 benchmark：评测 checkpoint 的路由/参数指标")
    ap.add_argument("--input", "-i", default="data/val.jsonl",
                    help="评测数据（val.jsonl 或 benchmark_vod.csv）")
    ap.add_argument("--live", action="store_true",
                    help="连接已运行的 vLLM 服务（VLLM_BASE_URL / VLLM_MODEL）")
    ap.add_argument("--route-only", action="store_true",
                    help="只评路由（不走 IR 生成/编译，等价于训练时 route_only 模式）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--workers", type=int, default=1, help="并发请求数")

    # 自动起服务相关
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="已合并的 HF checkpoint 路径（自动起 vLLM、跑评测、再关）")
    ap.add_argument("--checkpoint-dir", type=str, default=None,
                    help="训练输出目录，批量评测所有 checkpoint-*-merged")
    ap.add_argument("--auto-serve", action="store_true",
                    help="配合 --checkpoint / --checkpoint-dir 自动起停 vLLM")
    ap.add_argument("--serve-port", type=int, default=8100, help="临时 vLLM 端口")
    ap.add_argument("--serve-gpus", type=str, default="0,1,2,3", help="临时 vLLM 用的 GPU")
    ap.add_argument("--serve-tp", type=int, default=None, help="临时 vLLM 张量并行度")

    ap.add_argument("--dry-run", action="store_true",
                    help="只加载数据并打印分布，不连模型")
    ap.add_argument("--output", "-o", default=None,
                    help="汇总结果输出路径（JSON），默认打印到 stdout")
    args = ap.parse_args()

    # 加载评测集
    cases = load_cases(args.input)
    if args.limit > 0:
        cases = cases[:args.limit]
    print(f"加载 {len(cases)} 条评测用例 <- {args.input}")
    has_params = sum(c.gold_params is not None for c in cases)
    print(f"  带 gold_params: {has_params}/{len(cases)}")

    if args.dry_run:
        from collections import Counter
        dist = Counter(c.gold_tool for c in cases)
        print(f"\ngold_tool 分布:")
        for k, v in dist.most_common():
            print(f"  {k:<32}{v:>5}")
        return 0

    # 确定要评测的 checkpoint 列表
    checkpoints: list[tuple[str, str]] = []  # [(label, model_path)]
    if args.checkpoint_dir and args.auto_serve:
        merged = find_merged_checkpoints(args.checkpoint_dir)
        if not merged:
            print(f"错误：{args.checkpoint_dir} 下没有 checkpoint-*-merged 目录", file=sys.stderr)
            sys.exit(1)
        checkpoints = [(f"step-{step}", path) for step, path in merged]
        print(f"发现 {len(checkpoints)} 个 checkpoint: {[l for l,_ in checkpoints]}")
    elif args.checkpoint and args.auto_serve:
        label = os.path.basename(args.checkpoint.rstrip("/"))
        checkpoints = [(label, args.checkpoint)]
    elif args.live:
        checkpoints = [("live", "")]
    else:
        print("请指定 --live（连已有服务）或 --checkpoint/--checkpoint-dir --auto-serve（自动起服务）")
        sys.exit(1)

    # route_only 模式下设置环境变量（影响 PlannerAgent）
    if args.route_only:
        os.environ["PLANNER_ROUTE_ONLY"] = "1"

    all_summaries = []

    for label, model_path in checkpoints:
        proc = None
        try:
            if model_path:
                # 自动起 vLLM
                proc = start_vllm_server(model_path, port=args.serve_port,
                                         gpus=args.serve_gpus, tp=args.serve_tp)
                base_url = f"http://127.0.0.1:{args.serve_port}/v1"
                model_name = "bench-model"
            else:
                base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
                model_name = os.environ.get("VLLM_MODEL", "qwen-30b-moe")

            cfg = VLLMConfig(base_url=base_url, model=model_name)
            if os.environ.get("VLLM_REQUEST_FORMAT"):
                cfg.request_format = os.environ["VLLM_REQUEST_FORMAT"]
            planner = Planner(VLLMClient(cfg))

            # 跑评测（支持并发）
            import threading
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results: list[Optional[Result]] = [None] * len(cases)
            lock = threading.Lock()
            done_count = [0]
            route_count = [0]

            def _progress():
                sys.stderr.write(f"\r  [{label}] {done_count[0]}/{len(cases)}"
                                 f"  route={pct(route_count[0], done_count[0])}")
                sys.stderr.flush()

            if args.workers > 1:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(run_case, planner, c): i for i, c in enumerate(cases)}
                    for fut in as_completed(futs):
                        i = futs[fut]
                        r = fut.result()
                        results[i] = r
                        with lock:
                            done_count[0] += 1
                            route_count[0] += r.route_hit
                            _progress()
            else:
                for i, case in enumerate(cases):
                    r = run_case(planner, case)
                    results[i] = r
                    done_count[0] += 1
                    route_count[0] += r.route_hit
                    _progress()

            sys.stderr.write("\n")

            summary = summarize(results, label=label)
            all_summaries.append(summary)

        finally:
            if proc:
                stop_server(proc)

    # 输出对比表
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print("对比汇总:")
        print(f"  {'checkpoint':<30}{'route_acc':>10}{'final_acc':>10}{'param_acc':>10}")
        for s in all_summaries:
            print(f"  {s['label']:<30}{s['route_acc']*100:>9.1f}%{s['final_acc']*100:>9.1f}%"
                  f"{s['param_acc']*100:>9.1f}%")
        print("=" * 60)

    # 写入文件
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"\n汇总已写入: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
