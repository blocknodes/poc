#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把「影视用例.csv」清洗成可评测的 benchmark_vod.csv。

清洗规则：
- 只保留有效评测所需字段：id / domain / query / intent / expected_tool / expected_params / note
- domain 统一为 vod（源里全是「影视」）
- query 去零宽字符(\\u200c 等)、去首尾空白、内部换行折叠为空格
- intent(智能体意图) 去掉括号说明，只保留意图名
- expected_tool 归一：慢链路的 construction+search 组合统一为 vod_slow_search_data_search，
  多值/换行只取归一后的主工具
- expected_params 若有则解析后压紧输出(合法 JSON，ensure_ascii=False)，无则留空
- 丢弃无 query 或无 expected_tool 的行（无法评分）
- 按 (query, expected_tool, expected_params) 去重
"""
import csv
import json
import re
import sys
from pathlib import Path

SRC = Path("影视用例.csv")
DST = Path("benchmark_vod.csv")

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


def clean_query(q: str) -> str:
    q = q.translate(ZERO_WIDTH)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def norm_tool(t: str) -> str:
    t = t.strip()
    if not t:
        return ""
    parts = [p.strip() for p in re.split(r"[\n、,]", t) if p.strip()]
    if any("slow_search" in p for p in parts):
        return "vod_slow_search_data_search"
    return parts[0]


def norm_intent(i: str) -> str:
    return re.split(r"[（(]", i.strip())[0].strip()


def norm_params(p: str) -> str:
    p = p.strip()
    if not p:
        return ""
    try:
        obj = json.loads(p)
    except json.JSONDecodeError:
        return p  # 保留原样，交由后续人工修复
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def main() -> int:
    rows = list(csv.reader(SRC.open(encoding="utf-8-sig")))
    data = rows[1:]

    out, seen = [], set()
    dropped_no_query = dropped_no_tool = deduped = 0

    for r in data:
        cell = lambda idx: r[idx].strip() if len(r) > idx else ""
        query = clean_query(cell(1))
        tool = norm_tool(cell(5))
        if not query:
            dropped_no_query += 1
            continue
        if not tool:
            dropped_no_tool += 1
            continue
        intent = norm_intent(cell(3))
        params = norm_params(cell(6))
        note = cell(4)

        key = (query, tool, params)
        if key in seen:
            deduped += 1
            continue
        seen.add(key)
        out.append(["vod", query, intent, tool, params, note])

    with DST.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "domain", "query", "intent", "expected_tool",
                    "expected_params", "note"])
        for i, row in enumerate(out, 1):
            w.writerow([i, *row])

    print(f"源数据行            : {len(data)}")
    print(f"丢弃(无 query)      : {dropped_no_query}")
    print(f"丢弃(无 expected_tool): {dropped_no_tool}")
    print(f"去重                : {deduped}")
    print(f"写入 {DST}          : {len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
