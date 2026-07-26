#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 LLM（OpenAI 接口）批量生成 planner 训练数据。

思路：给 LLM 一个 seed query 或 intent 分类 + 工具列表，让它一次性产出 N 条
(query, gold_tool, gold_params) 三元组，然后过一遍 IR 校验确保 gold_params 合法。

支持两种生成模式：
  1. expand —— 基于已有 benchmark 样本做扩写变换（改述/加条件/换主体），保留 tool+结构
  2. create —— 按意图/工具分布从头生成全新 query + 标注

输出与 build_dataset.py 产出格式一致（可直接拼入 train.jsonl）。

用法：
  # 基于 benchmark 扩写（默认），每条扩 5 条，用 gpt-4o
  python generate_data.py --mode expand \
      --input ../benchmark_vod.csv \
      --api-base https://api.openai.com/v1 \
      --model gpt-4o \
      --api-key $OPENAI_API_KEY \
      --expand-n 5 \
      --output data/generated_train.jsonl

  # 按意图分布从头生成 200 条
  python generate_data.py --mode create \
      --total 200 \
      --api-base https://api.openai.com/v1 \
      --model gpt-4o \
      --api-key $OPENAI_API_KEY \
      --output data/generated_train.jsonl

  # 用本地 vLLM / 兼容接口
  python generate_data.py --mode expand \
      --input ../benchmark_vod.csv \
      --api-base http://localhost:8000/v1 \
      --model qwen-72b \
      --api-key EMPTY \
      --output data/generated_train.jsonl

  # 合并到训练集
  cat data/train.jsonl data/generated_train.jsonl > data/train_combined.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

# planner 包
_POC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _POC_ROOT not in sys.path:
    sys.path.insert(0, _POC_ROOT)

from planner.ir import IRError, parse_ir, validate_ir
from planner.registry import FIELD_REGISTRY, field_names, SORT_REGISTRY

try:
    import requests
except ImportError:
    print("需要安装 requests: pip install requests", file=sys.stderr)
    sys.exit(1)


# ===========================================================================
# LLM 客户端
# ===========================================================================
@dataclass
class LLMConfig:
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    api_key: str = ""
    temperature: float = 0.9
    max_tokens: int = 4096
    timeout: float = 120.0


def llm_chat(config: LLMConfig, messages: list[dict], temperature: Optional[float] = None,
             max_retries: int = 5) -> str:
    """调用 OpenAI 兼容 /chat/completions 接口，返回 assistant 文本。自动重试 429/5xx。"""
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature if temperature is not None else config.temperature,
        "max_tokens": config.max_tokens,
    }
    for attempt in range(max_retries):
        resp = requests.post(
            f"{config.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=config.timeout,
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(2 ** attempt + random.random(), 30)
            sys.stderr.write(f"\n  [retry] {resp.status_code}, wait {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            sys.stderr.flush()
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    # 最后一次失败直接抛
    resp.raise_for_status()
    return ""


# ===========================================================================
# 工具/意图定义（与 prompts.py 对齐）
# ===========================================================================
TOOL_INTENT_MAP = {
    "vod_search": [
        "vod_keyword_search",       # 明确名称/演员/分类/地区
        "vod_quality_search",       # 热度/口碑/排序
    ],
    "vod_slow_search_data_search": [
        "vod_vague_search",         # 场景/情绪/人群/模糊描述
        "vod_episodes_qa",          # 剧集问答（几集/更新状态）
    ],
    "vod_clip_search": [
        "vod_clip_lines_search",    # 台词搜索
        "vod_clip_plot_search",     # 剧情描述找片
        "vod_clip_source_search",   # 名场面找出处
    ],
    "vod_personalized_recommend": ["vod_personalized_recommend"],
    "vod_relate_recommend": ["vod_relate_recommend"],
    "vod_history": ["vod_history"],
}

# 所有合法 field（vod domain）
VOD_FIELDS = field_names("vod")
VOD_SORT_KEYS = list(SORT_REGISTRY.keys())

# IR 结构示例（供 LLM 参考）
IR_EXAMPLE = """\
IR 结构：{"domain":"vod","query":<node>,"sort":[...]（可选）}
node 可以是:
  {"field":"F","value":"V"}                         精确单值
  {"field":"F","values":["V1","V2"],"op":"and|or"}  同字段多值
  {"field":"F","value":0}                           状态字段(fee:0=免费,1=付费; is_over:0=连载,1=完结)
  {"field":"F","range":{"from":"X","to":"Y"}}       范围（开区间用"*"）；release_year 用 yyyyMMdd
  {"and":[node,...]} / {"or":[node,...]} / {"not":node}  布尔组合

可用 field（vod domain）：""" + ", ".join(VOD_FIELDS) + """
sort key: """ + ", ".join(VOD_SORT_KEYS)


# ===========================================================================
# Mode: expand —— 基于已有样本扩写
# ===========================================================================
EXPAND_SYSTEM = f"""你是一个电视端搜索数据标注员。给你一条已有的 (query, tool, params) 样本，
请生成 {{n}} 条**语义相近但表达不同**的变体，要求：
  1. 保持 tool 和逻辑意图一致（如条件组合方式、排除、排序）。
  2. 变换方式：改述/换主体（换演员/片名/分类/年份等）/加减细节条件/口语化/方言式。
  3. 如果原样本有 params (IR)，也输出对应变体的新 params (IR)。
  4. 如果原样本没有 params（非检索类工具或 slow_search），只输出 query 和 tool。

{IR_EXAMPLE}

输出格式（JSON array，严格只输出 JSON，不要其他文字）：
[
  {{"query": "...", "tool": "...", "params": {{...}} 或 null}},
  ...
]"""

EXPAND_USER_TEMPLATE = """原样本：
query: {query}
tool: {tool}
params: {params}

请生成 {n} 条变体（JSON array）："""


def expand_one(config: LLMConfig, seed: dict, n: int) -> list[dict]:
    """基于一个 seed 样本，让 LLM 扩写 n 条变体。"""
    params_str = seed.get("expected_params") or "null"
    user_msg = EXPAND_USER_TEMPLATE.format(
        query=seed["query"], tool=seed["expected_tool"], params=params_str, n=n
    )
    messages = [
        {"role": "system", "content": EXPAND_SYSTEM.format(n=n)},
        {"role": "user", "content": user_msg},
    ]
    raw = llm_chat(config, messages)
    return _parse_json_array(raw)


# ===========================================================================
# Mode: create —— 按意图分布从头生成
# ===========================================================================
CREATE_SYSTEM = f"""你是一个电视端搜索数据标注员。请按照要求为指定 tool 生成用户查询样本。

工具列表与含义：
- vod_search：可结构化的明确检索（名称/演员/分类/地区/年份/免费/热度排序等）
- vod_slow_search_data_search：需要语义理解的模糊检索（场景/情绪/人群推荐/剧集问答/出品方/奖项/待播等）
- vod_clip_search：台词搜索/剧情描述找片/名场面找出处
- vod_personalized_recommend：纯个性化推荐（无具体条件）
- vod_relate_recommend：找类似的（"类似XX的"）
- vod_history：看观看历史

{IR_EXAMPLE}

要求：
  1. query 是真实用户在电视语音遥控器上会说的自然语言（中文、口语化、可包含语气词）。
  2. 对 vod_search，必须输出合法的 params (IR)，条件组合要多样（单条件/多条件/排除/排序/范围）。
  3. 对 vod_slow_search_data_search，params 设为 null（这类查询无法直接结构化）。
  4. 对 clip/personalized/relate/history，params 设为 null。
  5. 每条样本都要独立、不重复、覆盖不同题材/演员/分类/年份等。

输出格式（JSON array，严格只输出 JSON，不要其他文字）：
[
  {{"query": "...", "tool": "...", "params": {{...}} 或 null}},
  ...
]"""

CREATE_USER_TEMPLATE = """请为 tool="{tool}" 生成 {n} 条样本（JSON array）。
意图方向参考：{intent_hint}
要求表达多样化、条件组合丰富，尽量覆盖不同场景。"""


def create_batch(config: LLMConfig, tool: str, n: int, intent_hint: str = "") -> list[dict]:
    """为指定 tool 从头生成 n 条样本。"""
    user_msg = CREATE_USER_TEMPLATE.format(tool=tool, n=n, intent_hint=intent_hint)
    messages = [
        {"role": "system", "content": CREATE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    raw = llm_chat(config, messages)
    return _parse_json_array(raw)


# ===========================================================================
# JSON 解析 & IR 校验
# ===========================================================================
def _parse_json_array(text: str) -> list[dict]:
    """从 LLM 输出中提取 JSON array。"""
    text = text.strip()
    # 剥离 markdown 代码块
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    # 找 [ ... ]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        arr = json.loads(text[start:end + 1])
        if isinstance(arr, list):
            return arr
    except json.JSONDecodeError:
        pass
    return []


def validate_and_fix(item: dict, domain: str = "vod") -> Optional[dict]:
    """校验一条生成结果，返回合法的训练样本 dict 或 None。"""
    query = (item.get("query") or "").strip()
    tool = (item.get("tool") or "").strip()
    if not query or not tool:
        return None

    # 校验 params（IR）
    params_raw = item.get("params")
    gold_params = ""
    if params_raw and isinstance(params_raw, dict):
        # 确保有 domain
        if "domain" not in params_raw:
            params_raw["domain"] = domain
        # 尝试 parse + validate
        try:
            ir = parse_ir(params_raw)
            errs = validate_ir(ir)
            if not errs:
                gold_params = json.dumps(params_raw, ensure_ascii=False)
            else:
                # 尝试修复常见问题：缺 domain
                params_raw["domain"] = domain
                ir = parse_ir(params_raw)
                errs = validate_ir(ir)
                if not errs:
                    gold_params = json.dumps(params_raw, ensure_ascii=False)
                # else: 放弃这条的 params，只保留 query+tool
        except (IRError, KeyError, TypeError):
            pass  # params 非法，只保留 query+tool
    elif params_raw and isinstance(params_raw, str):
        try:
            p = json.loads(params_raw)
            return validate_and_fix({**item, "params": p}, domain)
        except json.JSONDecodeError:
            pass

    return {
        "messages": [{"role": "user", "content": query}],
        "query": query,
        "gold_domain": domain,
        "gold_tool": tool,
        "gold_params": gold_params,
        "env_config": {"name": "planner_env", "max_repairs": 2},
    }


# ===========================================================================
# 数据加载
# ===========================================================================
def load_seeds(path: str) -> list[dict]:
    """加载 benchmark CSV 作为 seed 样本。"""
    seeds = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            query = (r.get("query") or "").strip()
            if not query:
                continue
            seeds.append({
                "query": query,
                "domain": (r.get("domain") or "vod").strip(),
                "expected_tool": (r.get("expected_tool") or "").strip(),
                "expected_params": (r.get("expected_params") or "").strip(),
                "intent": (r.get("intent") or "").strip(),
            })
    return seeds


# ===========================================================================
# 主流程
# ===========================================================================
def run_expand(config: LLMConfig, seeds: list[dict], expand_n: int,
               workers: int, limit: int) -> list[dict]:
    """expand 模式：基于 seeds 扩写。"""
    if limit > 0:
        seeds = seeds[:limit]

    results: list[dict] = []
    total = len(seeds)
    done = [0]

    def _task(seed):
        try:
            items = expand_one(config, seed, expand_n)
            valid = []
            for it in items:
                # 继承原样本的 tool（如果 LLM 没改的话）
                if not it.get("tool"):
                    it["tool"] = seed["expected_tool"]
                s = validate_and_fix(it, seed.get("domain", "vod"))
                if s:
                    valid.append(s)
            return valid
        except Exception as e:
            sys.stderr.write(f"\n  [warn] expand failed: {e}\n")
            return []

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_task, s): i for i, s in enumerate(seeds)}
            for fut in as_completed(futs):
                batch = fut.result()
                results.extend(batch)
                done[0] += 1
                sys.stderr.write(f"\r  expand: {done[0]}/{total}  generated: {len(results)}")
                sys.stderr.flush()
    else:
        for i, seed in enumerate(seeds):
            batch = _task(seed)
            results.extend(batch)
            done[0] += 1
            sys.stderr.write(f"\r  expand: {done[0]}/{total}  generated: {len(results)}")
            sys.stderr.flush()

    sys.stderr.write("\n")
    return results


def run_create(config: LLMConfig, total_target: int, workers: int) -> list[dict]:
    """create 模式：按工具分布从头生成。"""
    # 按比例分配各 tool 的条数
    distribution = {
        "vod_search": 0.45,
        "vod_slow_search_data_search": 0.30,
        "vod_clip_search": 0.15,
        "vod_personalized_recommend": 0.03,
        "vod_relate_recommend": 0.04,
        "vod_history": 0.03,
    }

    tasks: list[tuple[str, int, str]] = []  # (tool, n_per_batch, intent_hint)
    for tool, ratio in distribution.items():
        n_total = max(1, int(total_target * ratio))
        intents = TOOL_INTENT_MAP.get(tool, [tool])
        hint = "/".join(intents)
        # 每 batch 最多 15 条（LLM 输出稳定性）
        batch_size = 15
        while n_total > 0:
            n = min(batch_size, n_total)
            tasks.append((tool, n, hint))
            n_total -= n

    random.shuffle(tasks)
    results: list[dict] = []
    done = [0]
    total_tasks = len(tasks)

    def _task(tool, n, hint):
        try:
            items = create_batch(config, tool, n, hint)
            valid = []
            for it in items:
                if not it.get("tool"):
                    it["tool"] = tool
                s = validate_and_fix(it)
                if s:
                    valid.append(s)
            return valid
        except Exception as e:
            sys.stderr.write(f"\n  [warn] create failed ({tool}): {e}\n")
            return []

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_task, t, n, h): i for i, (t, n, h) in enumerate(tasks)}
            for fut in as_completed(futs):
                batch = fut.result()
                results.extend(batch)
                done[0] += 1
                sys.stderr.write(f"\r  create: {done[0]}/{total_tasks}  generated: {len(results)}")
                sys.stderr.flush()
    else:
        for t, n, h in tasks:
            batch = _task(t, n, h)
            results.extend(batch)
            done[0] += 1
            sys.stderr.write(f"\r  create: {done[0]}/{total_tasks}  generated: {len(results)}")
            sys.stderr.flush()

    sys.stderr.write("\n")
    return results


def write_jsonl(path: str, samples: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="用 LLM 生成 planner 训练数据")
    ap.add_argument("--mode", choices=["expand", "create"], default="expand",
                    help="生成模式：expand=基于已有样本扩写，create=按意图分布从头生成")
    ap.add_argument("--input", "-i", default="../benchmark_vod.csv",
                    help="seed 数据（expand 模式用，benchmark CSV 或 jsonl）")
    ap.add_argument("--output", "-o", default="data/generated_train.jsonl",
                    help="输出路径")
    ap.add_argument("--total", type=int, default=200,
                    help="create 模式目标总数")
    ap.add_argument("--expand-n", type=int, default=5,
                    help="expand 模式每条 seed 扩写条数")
    ap.add_argument("--limit", type=int, default=0,
                    help="expand 模式只取前 N 个 seed（0=全部）")
    ap.add_argument("--workers", type=int, default=4,
                    help="并发请求数")

    # LLM 配置
    ap.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "http://aibi-model.hisense.com:8086/v1"),
                    help="OpenAI 兼容接口地址")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "hx-deepseek-v4-flash"),
                    help="模型名")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "sk-tZKuOAXymN2Vq0w4gzeBEQ"),
                    help="API key")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="生成温度")

    # 去重 & 合并
    ap.add_argument("--dedup", action="store_true",
                    help="对输出做 query 去重")
    ap.add_argument("--append", action="store_true",
                    help="追加到已有输出文件而非覆盖")

    args = ap.parse_args()

    config = LLMConfig(
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    if not config.api_key:
        print("错误：需要设置 --api-key 或 OPENAI_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)

    print(f"模式: {args.mode}")
    print(f"LLM: {config.api_base}  model={config.model}  temperature={config.temperature}")
    print(f"并发: {args.workers}")

    t0 = time.time()

    if args.mode == "expand":
        seeds = load_seeds(args.input)
        print(f"加载 {len(seeds)} 个 seed <- {args.input}")
        results = run_expand(config, seeds, args.expand_n, args.workers, args.limit)
    else:
        print(f"目标生成: {args.total} 条")
        results = run_create(config, args.total, args.workers)

    elapsed = time.time() - t0
    print(f"\n生成完成: {len(results)} 条有效样本  ({elapsed:.1f}s)")

    # 去重
    if args.dedup:
        seen = set()
        deduped = []
        for s in results:
            q = s["query"]
            if q not in seen:
                seen.add(q)
                deduped.append(s)
        print(f"去重: {len(results)} -> {len(deduped)}")
        results = deduped

    # 统计
    from collections import Counter
    dist = Counter(s["gold_tool"] for s in results)
    has_params = sum(1 for s in results if s["gold_params"])
    print(f"\ngold_tool 分布:")
    for k, v in dist.most_common():
        print(f"  {k:<34}{v:>5}")
    print(f"带合法 params: {has_params}/{len(results)}")

    # 写入
    if args.append and os.path.exists(args.output):
        # 追加模式
        with open(args.output, "a", encoding="utf-8") as f:
            for s in results:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\n已追加到: {args.output}")
    else:
        write_jsonl(args.output, results)
        print(f"\n已写入: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
