#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 planner 的 benchmark CSV 转成 GRPO gym-env 训练用的 jsonl。

输入 CSV 列（见 benchmark_vod.csv）：
    id, domain, query, intent, expected_tool, expected_params, note

输出 jsonl 每行（ms-swift GRPO + gym env 的数据契约）：
    {
      "messages":   [{"role":"user","content": <query>}],   # 占位，env.reset 会重建
      "query":      <query>,                                  # env.reset 读它
      "gold_domain":<domain>,                                 # reward 用
      "gold_tool":  <expected_tool>,                          # reward 用
      "gold_params":<expected_params 原始 JSON 字符串>,       # reward 用（可能为空串）
      "env_config": {"name": "planner_env", "max_repairs": 2}
    }

透传列（query/gold_*）经 `--vllm_server_pass_dataset true` 进入 rollout 的 data_dict，
并在打分阶段作为 batched kwargs 传给 reward 函数。

用法：
    python build_dataset.py --input ../benchmark_vod.csv \
        --out-dir data --val-ratio 0.1 --max-repairs 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Any


def load_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            query = (r.get("query") or "").strip()
            if not query:
                continue
            # gold_params 保留为原始 JSON 字符串（可能为空）；校验一下能否解析
            gp_raw = (r.get("expected_params") or "").strip()
            if gp_raw:
                try:
                    json.loads(gp_raw)
                except json.JSONDecodeError:
                    gp_raw = ""  # golden 本身非法则丢弃参数，避免污染 reward
            rows.append({
                "domain": (r.get("domain") or "vod").strip(),
                "query": query,
                "expected_tool": (r.get("expected_tool") or "").strip(),
                "expected_params": gp_raw,
            })
    return rows


def to_sample(row: dict[str, Any], max_repairs: int) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": row["query"]}],
        "query": row["query"],
        "gold_domain": row["domain"],
        "gold_tool": row["expected_tool"],
        "gold_params": row["expected_params"],
        "env_config": {"name": "planner_env", "max_repairs": max_repairs},
    }


def write_jsonl(path: str, samples: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build GRPO gym-env dataset for planner")
    ap.add_argument("--input", "-i", default="../benchmark_vod.csv", help="benchmark CSV 路径")
    ap.add_argument("--out-dir", "-o", default="data", help="输出目录")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    ap.add_argument("--max-repairs", type=int, default=2, help="env 自修复轮数上限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_rows(args.input)
    random.Random(args.seed).shuffle(rows)
    samples = [to_sample(r, args.max_repairs) for r in rows]

    n_val = int(len(samples) * args.val_ratio)
    val, train = samples[:n_val], samples[n_val:]

    train_path = os.path.join(args.out_dir, "train.jsonl")
    write_jsonl(train_path, train)
    print(f"train: {len(train)} -> {train_path}")
    if val:
        val_path = os.path.join(args.out_dir, "val.jsonl")
        write_jsonl(val_path, val)
        print(f"val  : {len(val)} -> {val_path}")

    # 分布速览
    from collections import Counter
    dist = Counter(s["gold_tool"] for s in samples)
    withp = sum(1 for s in samples if s["gold_params"])
    print("\ngold_tool 分布:")
    for k, v in dist.most_common():
        print(f"  {k:<34}{v:>5}")
    print(f"带 golden 参数: {withp}/{len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
