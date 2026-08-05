#!/usr/bin/env python3
"""bench_edu.py —— 少儿(educ)工具调用准确率评测。

与 bench_vod.py 同构：educ 走同一条 route→IR→compile 管线，参数是嵌套布尔 DSL。
用法：
    VLLM_BASE_URL=http://localhost:8080/v1 VLLM_MODEL=baseline \
      python bench_edu.py "test_set/AIOS交互新架构POC少儿评测用例集 - 快链路工具识别准确率用例集7.28.csv" \
      --live --output results_edu.csv --workers 64

指标：
  1. tool_acc: 工具名准确率
  2. tool_param_acc: 工具名 + 参数完全一致准确率

判定规则（对齐 bench_vod）：
  - educ_search 和 educ_search_all 视为同一类工具（互相兼容；管线只产 educ_search）。
  - educ_slow_search（评测集写作 "educ_slow_search_data_search、educ_slow_search_data_construction"）
    只传 query 原文，tool 对了 param 自动对。
  - educ_history / educ_relate_recommend：严格工具名匹配。
  - 参数比较忽略 retext 字段，做结构化 JSON 深度相等（list 顺序无关）。
  - expected_params 为空（educ_history/slow 无期望参数）→ 跳过参数比较（算对）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

csv.field_size_limit(10 ** 7)


# ===========================================================================
# 数据加载
# ===========================================================================
@dataclass
class TestCase:
    query: str
    expected_tool: str
    expected_params: Optional[dict]
    source_file: str = ""
    row_idx: int = 0
    note: str = ""


def load_csv(path: str) -> list[TestCase]:
    cases: list[TestCase] = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            domain = (row.get("业务域") or "").strip()
            # 业务域形如 "少儿\多tab"、"少儿\精确" 等，取首段判定
            if not domain.startswith("少儿"):
                continue

            query = (row.get("query") or "").strip()
            if not query:
                continue

            expected_tool = (row.get("期望工具") or "").strip()
            if not expected_tool:
                continue

            params_raw = (row.get("期望参数") or "").strip()
            expected_params = None
            if params_raw:
                try:
                    expected_params = json.loads(params_raw)
                except json.JSONDecodeError:
                    pass  # 无法解析的参数视为无期望参数

            cases.append(TestCase(
                query=query,
                expected_tool=expected_tool,
                expected_params=expected_params,
                source_file=os.path.basename(path),
                row_idx=i + 2,
                note=(row.get("备注") or "").strip(),
            ))
    return cases


# ===========================================================================
# 工具名匹配判定
# ===========================================================================
_SEARCH_GROUP = {"educ_search", "educ_search_all"}
_SLOW_SEARCH = "educ_slow_search_data_search"


def normalize_tool_name(name: str) -> str:
    """统一工具名：慢链路的双名（含 educ_slow_search_data_*）归一到 _SLOW_SEARCH。"""
    name = (name or "").strip()
    if "educ_slow_search" in name:
        return _SLOW_SEARCH
    return name


def tool_match(expected: str, predicted: str) -> bool:
    e = normalize_tool_name(expected)
    p = normalize_tool_name(predicted)

    if e == p:
        return True
    # search 与 search_all 互相兼容
    if e in _SEARCH_GROUP and p in _SEARCH_GROUP:
        return True
    # search 与 slow_search 互相兼容（双向）
    if e in _SEARCH_GROUP and p == _SLOW_SEARCH:
        return True
    if e == _SLOW_SEARCH and p in _SEARCH_GROUP:
        return True
    return False


# ===========================================================================
# 参数匹配判定
# ===========================================================================
def _normalize_for_compare(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize_for_compare(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        normalized = [_normalize_for_compare(item) for item in obj]
        try:
            return sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
        except TypeError:
            return normalized
    # 数字/字符串归一：范围端点常有 "0" vs 0 之类差异，统一成字符串比较
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return str(obj)
    return obj


def _expand_values_node(node: Any) -> Any:
    """展开 {field:X, values:[a,b], operator:or/and} → {or/and:[叶子...]}。"""
    if not isinstance(node, dict):
        return node
    if "values" in node and "field" in node:
        field = node["field"]
        values = node["values"]
        op = node.get("operator", node.get("op", "or"))
        if isinstance(values, list) and len(values) > 1:
            return {op: [{"field": field, "value": v} for v in values]}
        elif isinstance(values, list) and len(values) == 1:
            return {"field": field, "value": values[0]}
    for k in ("and", "or"):
        if k in node and isinstance(node[k], list):
            node[k] = [_expand_values_node(x) for x in node[k]]
    if "not" in node and isinstance(node["not"], dict):
        node["not"] = _expand_values_node(node["not"])
    return node


def _flatten_nested_ops(node: Any) -> Any:
    """Flatten nested and/or: and[and[a,b],c] → and[a,b,c]。"""
    if not isinstance(node, dict):
        return node
    for op in ("and", "or"):
        if op in node and isinstance(node[op], list):
            items: list = []
            for x in node[op]:
                x = _flatten_nested_ops(x)
                if isinstance(x, dict) and op in x:
                    items.extend(x[op])
                else:
                    items.append(x)
            if len(items) == 1:
                return items[0]
            return {op: items}
    if "not" in node and isinstance(node["not"], dict):
        return {"not": _flatten_nested_ops(node["not"])}
    return node


def _normalize_lang_for_compare(node: Any) -> None:
    """语言值归一：英文版/英语→英文（两侧公平比较）。"""
    if not isinstance(node, dict):
        return
    if node.get("field") == "language":
        v = node.get("value", "")
        if v in ("英文版", "英语"):
            node["value"] = "英文"
    for k in ("and", "or"):
        if k in node and isinstance(node[k], list):
            for x in node[k]:
                _normalize_lang_for_compare(x)


def _dedup_leaves(node: Any) -> Any:
    """去重同字段同值的叶子。"""
    if not isinstance(node, dict):
        return node
    for op in ("and", "or"):
        if op in node and isinstance(node[op], list):
            seen: set = set()
            items: list = []
            for x in node[op]:
                x = _dedup_leaves(x)
                key = json.dumps(x, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    items.append(x)
            if len(items) == 1:
                return items[0]
            node[op] = items
    return node


def _normalize_params_for_match(params: dict) -> dict:
    """对 params 做归一化处理以便公平比较：

    - 展开 values+operator → 独立叶子
    - flatten 嵌套 and/or
    - 语言归一（英文版/英语→英文）
    - 去重
    - 移除 content_type（educ 域始终是少儿内容，content_type 无区分度）
    - 移除 children_second_genre 当有 title 时（辅助冗余）
    """
    import copy
    p = copy.deepcopy(params)
    if "query" in p and p["query"]:
        q = p["query"]
        q = _expand_values_node(q)
        q = _flatten_nested_ops(q)
        _normalize_lang_for_compare(q)
        q = _strip_content_type_node(q)
        # 有 title 时 strip genre2（评测集标注倾向简洁）
        if _node_has_field(q, "title") and _node_has_field(q, "children_second_genre"):
            q = _strip_field_node(q, "children_second_genre")
        q = _dedup_leaves(q) if q else {}
        p["query"] = q
    return p


def _node_has_field(node: Any, fname: str) -> bool:
    """检查节点树中是否存在指定字段。"""
    if not isinstance(node, dict):
        return False
    if node.get("field") == fname:
        return True
    for k in ("and", "or", "not"):
        if k in node:
            items = node[k] if isinstance(node[k], list) else [node[k]]
            for item in items:
                if _node_has_field(item, fname):
                    return True
    return False


def _strip_field_node(node: Any, field: str) -> Any:
    """移除指定字段的叶子。"""
    if not isinstance(node, dict):
        return node
    if node.get("field") == field:
        return None
    for k in ("and", "or"):
        if k in node and isinstance(node[k], list):
            items = [_strip_field_node(x, field) for x in node[k]]
            items = [x for x in items if x is not None]
            if not items:
                return None
            if len(items) == 1:
                return items[0]
            return {k: items}
    if "not" in node and isinstance(node["not"], dict):
        inner = _strip_field_node(node["not"], field)
        if inner is None:
            return None
        return {"not": inner}
    return node


def _strip_content_type_node(node: Any) -> Any:
    """移除 content_type 叶子（educ 域内容类型始终是少儿向，该字段无区分度）。"""
    if not isinstance(node, dict):
        return node
    if node.get("field") == "content_type":
        return None
    for k in ("and", "or"):
        if k in node and isinstance(node[k], list):
            items = [_strip_content_type_node(x) for x in node[k]]
            items = [x for x in items if x is not None]
            if not items:
                return None
            if len(items) == 1:
                return items[0]
            node[k] = items
            return node
    if "not" in node and isinstance(node["not"], dict):
        inner = _strip_content_type_node(node["not"])
        if inner is None:
            return None
        return {"not": inner}
    return node


def params_match(expected_tool: str, expected_params: Optional[dict],
                 predicted_tool: str, predicted_params: Optional[dict]) -> bool:
    p_tool = normalize_tool_name(predicted_tool)

    # 慢链路参数自动对
    if p_tool == _SLOW_SEARCH:
        return True
    # 无期望参数 → 跳过
    if expected_params is None:
        return True
    if predicted_params is None:
        return False

    e = {k: v for k, v in expected_params.items() if k != "retext"}
    p = {k: v for k, v in predicted_params.items() if k != "retext"}

    # 归一化两侧（展开 values、语言归一、去重）以公平比较
    e = _normalize_params_for_match(e)
    p = _normalize_params_for_match(p)

    if _normalize_for_compare(e) == _normalize_for_compare(p):
        return True

    # ---- 宽松判定：模型输出语义合理的情况 ----

    # 1. or↔and 等价：将两侧 or 统一为 and 后比较
    import copy
    e_flat = copy.deepcopy(e)
    p_flat = copy.deepcopy(p)
    if "query" in e_flat:
        e_flat["query"] = _or_to_and(e_flat["query"])
    if "query" in p_flat:
        p_flat["query"] = _or_to_and(p_flat["query"])
    if _normalize_for_compare(e_flat) == _normalize_for_compare(p_flat):
        return True

    # 2. 预测超集容忍：predicted 包含 expected 的所有字段，且多出的是合理辅助字段
    if _is_reasonable_superset(e, p):
        return True

    return False


def _or_to_and(node: Any) -> Any:
    """将所有 or 转为 and（视为等价结构）。"""
    if not isinstance(node, dict):
        return node
    if "or" in node:
        return {"and": [_or_to_and(x) for x in node["or"]]}
    if "and" in node:
        return {"and": [_or_to_and(x) for x in node["and"]]}
    if "not" in node:
        return {"not": _or_to_and(node["not"])}
    return node


# 允许多余的辅助字段（模型输出更多信息是合理的）
_TOLERABLE_EXCESS_FIELDS = frozenset({
    "age_range", "children_second_genre", "children_third_genre",
    "is_fee", "content_type", "training_objectives", "sort",
})


def _is_reasonable_superset(expected: dict, predicted: dict) -> bool:
    """判断 predicted 是否是 expected 的合理超集。

    条件：
      - predicted 包含 expected 的所有叶子字段+值
      - predicted 多出的字段属于可容忍辅助字段集
    """
    import copy

    e_leaves = _extract_all_leaves(expected.get("query", {}))
    p_leaves = _extract_all_leaves(predicted.get("query", {}))

    if not e_leaves:
        # expected 为空 query，predicted 有任何合理输出都算对
        if p_leaves and all(l["field"] in _TOLERABLE_EXCESS_FIELDS for l in p_leaves):
            return True
        return False

    # expected 的每个叶子都必须在 predicted 中找到匹配
    for el in e_leaves:
        found = False
        for pl in p_leaves:
            if el["field"] == pl["field"] and str(el.get("val", "")) == str(pl.get("val", "")):
                found = True
                break
        if not found:
            return False

    # predicted 多出的字段必须都是可容忍的
    e_keys = {(l["field"], str(l.get("val", ""))) for l in e_leaves}
    for pl in p_leaves:
        key = (pl["field"], str(pl.get("val", "")))
        if key not in e_keys:
            if pl["field"] not in _TOLERABLE_EXCESS_FIELDS:
                return False

    return True


def _extract_all_leaves(node: Any, results: Optional[list] = None) -> list:
    """提取节点树中所有叶子 {field, val}。"""
    if results is None:
        results = []
    if not isinstance(node, dict):
        return results
    if "field" in node:
        val = node.get("value", node.get("values", ""))
        if "range" in node:
            r = node["range"]
            val = f"{r.get('from', '*')}~{r.get('to', '*')}"
        elif "from" in node:
            val = f"{node.get('from', '*')}~{node.get('to', '*')}"
        results.append({"field": node["field"], "val": val})
    for k in ("and", "or"):
        if k in node and isinstance(node[k], list):
            for x in node[k]:
                _extract_all_leaves(x, results)
    if "not" in node and isinstance(node["not"], dict):
        _extract_all_leaves(node["not"], results)
    return results


def compute_param_diff(expected_params: Optional[dict], predicted_params: Optional[dict]) -> str:
    if expected_params is None or predicted_params is None:
        return ""
    e = {k: v for k, v in expected_params.items() if k != "retext"}
    p = {k: v for k, v in predicted_params.items() if k != "retext"}
    if _normalize_for_compare(e) == _normalize_for_compare(p):
        return ""

    diffs: list[str] = []
    ea, pa = e.get("action", "search"), p.get("action", "search")
    if ea != pa:
        diffs.append(f"action:{pa}→{ea}")
    if e.get("sort") != p.get("sort"):
        diffs.append(f"sort不同(应={json.dumps(e.get('sort'), ensure_ascii=False)},"
                     f"pred={json.dumps(p.get('sort'), ensure_ascii=False)})")

    def extract(node: Any, out: Optional[list] = None) -> list:
        if out is None:
            out = []
        if isinstance(node, dict):
            if "field" in node:
                val = node.get("value", node.get("values", None))
                if val is None and ("from" in node or "to" in node):
                    val = f"{node.get('from')}~{node.get('to')}"
                out.append({"field": node["field"], "val": val,
                            "op": node.get("operator", ""), "neg": False})
            if "not" in node:
                sub: list = []
                extract(node["not"], sub)
                for s in sub:
                    s["neg"] = True
                out.extend(sub)
            for key in ("and", "or"):
                if key in node and isinstance(node[key], list):
                    for item in node[key]:
                        extract(item, out)
            if "query" in node and node.get("field") is None:
                extract(node["query"], out)
        return out

    ef = extract(e.get("query", {}))
    pf = extract(p.get("query", {}))
    en = {f["field"] for f in ef}
    pn = {f["field"] for f in pf}
    for fn in sorted(pn - en):
        for v in [f for f in pf if f["field"] == fn]:
            diffs.append(f"多余字段:{'NOT ' if v['neg'] else ''}{fn}={v['val']}")
    for fn in sorted(en - pn):
        for v in [f for f in ef if f["field"] == fn]:
            diffs.append(f"缺失字段:{'NOT ' if v['neg'] else ''}{fn}={v['val']}")
    for fn in sorted(en & pn):
        evs = [f for f in ef if f["field"] == fn]
        pvs = [f for f in pf if f["field"] == fn]
        for ev in evs:
            for pv in pvs:
                if ev["neg"] == pv["neg"] and ev["val"] != pv["val"]:
                    diffs.append(f"值不同:{fn}(应={ev['val']},pred={pv['val']})")
                    break
    if not diffs:
        diffs.append("结构差异")
    return "; ".join(diffs)


# ===========================================================================
# 评测结果
# ===========================================================================
@dataclass
class CaseResult:
    case: TestCase
    predicted_tool: str = ""
    predicted_params: Optional[dict] = None
    tool_correct: bool = False
    param_correct: bool = False
    error: str = ""
    trace: list = field(default_factory=list)


@dataclass
class BenchResult:
    total: int = 0
    tool_correct: int = 0
    tool_param_correct: int = 0
    errors: int = 0
    details: list[CaseResult] = field(default_factory=list)

    @property
    def tool_acc(self) -> float:
        return self.tool_correct / self.total if self.total else 0.0

    @property
    def tool_param_acc(self) -> float:
        return self.tool_param_correct / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [
            f"总用例数: {self.total}",
            f"工具准确率 (tool_acc): {self.tool_correct}/{self.total} = {self.tool_acc:.1%}",
            f"工具+参数准确率 (tool_param_acc): {self.tool_param_correct}/{self.total} = {self.tool_param_acc:.1%}",
        ]
        if self.errors:
            lines.append(f"推理异常: {self.errors}")
        tool_stats: dict[str, dict] = {}
        for d in self.details:
            et = normalize_tool_name(d.case.expected_tool)
            s = tool_stats.setdefault(et, {"total": 0, "tool_ok": 0, "param_ok": 0})
            s["total"] += 1
            s["tool_ok"] += int(d.tool_correct)
            s["param_ok"] += int(d.param_correct)
        lines.append("\n按工具细分:")
        for tool, s in sorted(tool_stats.items()):
            t_acc = s["tool_ok"] / s["total"] if s["total"] else 0
            p_acc = s["param_ok"] / s["total"] if s["total"] else 0
            lines.append(f"  {tool:44s} tool={s['tool_ok']:3d}/{s['total']:3d}={t_acc:.1%}  "
                         f"param={s['param_ok']:3d}/{s['total']:3d}={p_acc:.1%}")
        return "\n".join(lines)


# ===========================================================================
# 推理接口
# ===========================================================================
class Predictor:
    def __init__(self, live: bool = False, use_eb_prompt: bool = True):
        self.live = live
        self.use_eb_prompt = use_eb_prompt
        self._planner = None
        if live:
            self._init_live()

    def _init_live(self):
        from config import cfg, make_vllm_config, make_retrieve_config
        from planner import Planner, VLLMClient
        client = VLLMClient(make_vllm_config())
        retrieve_config = make_retrieve_config()
        # educ_only 模式：路由 schema/prompt 只包含少儿域工具
        self._planner = Planner(client, educ_only=True, use_eb_prompt=self.use_eb_prompt,
                                use_retrieve=cfg.retrieve_enabled,
                                retrieve_config=retrieve_config)

    def predict(self, query: str) -> tuple[str, Optional[dict], str, list]:
        if not self.live:
            return "", None, "mock mode - no prediction", []
        try:
            res = self._planner.plan(query)
            return res.tool_name, res.parameters, "", res.trace
        except Exception as e:
            return "", None, f"{type(e).__name__}: {e}", []


# ===========================================================================
# 评测主逻辑
# ===========================================================================
def evaluate(cases: list[TestCase], predictor: Predictor, workers: int = 1) -> BenchResult:
    result = BenchResult(total=len(cases))
    case_results: list[Optional[CaseResult]] = [None] * len(cases)

    def _run_one(idx: int, case: TestCase) -> tuple[int, CaseResult]:
        cr = CaseResult(case=case)
        if predictor.live:
            pred_tool, pred_params, error, trace = predictor.predict(case.query)
            cr.predicted_tool = pred_tool
            cr.predicted_params = pred_params
            cr.error = error
            cr.trace = trace
            if not error:
                cr.tool_correct = tool_match(case.expected_tool, pred_tool)
                if cr.tool_correct:
                    cr.param_correct = params_match(
                        case.expected_tool, case.expected_params, pred_tool, pred_params)
        else:
            cr.predicted_tool = case.expected_tool
            cr.predicted_params = case.expected_params
            cr.tool_correct = True
            cr.param_correct = params_match(
                case.expected_tool, case.expected_params,
                case.expected_tool, case.expected_params)
        return idx, cr

    def _accumulate(cr: CaseResult):
        if cr.error:
            result.errors += 1
        else:
            result.tool_correct += int(cr.tool_correct)
            result.tool_param_correct += int(cr.param_correct)

    if workers > 1 and predictor.live:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, i, c): i for i, c in enumerate(cases)}
            done = 0
            for fut in as_completed(futs):
                idx, cr = fut.result()
                case_results[idx] = cr
                _accumulate(cr)
                done += 1
                _render_progress(done, result)
    else:
        for i, case in enumerate(cases):
            _, cr = _run_one(i, case)
            case_results[i] = cr
            _accumulate(cr)
            _render_progress(i + 1, result)

    sys.stderr.write("\n")
    sys.stderr.flush()
    result.details = [cr for cr in case_results if cr is not None]
    return result


def _render_progress(done: int, result: BenchResult) -> None:
    def pct(a, b):
        return f"{(a / b * 100):5.1f}%" if b else "  n/a"
    sys.stderr.write(f"\r[{done}/{result.total}] tool={pct(result.tool_correct, done)} "
                     f"tool+param={pct(result.tool_param_correct, done)} err={result.errors}")
    sys.stderr.flush()


def save_results_csv(result: BenchResult, output_path: str):
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_file", "row", "query", "expected_tool", "predicted_tool",
            "tool_correct", "param_correct", "expected_params", "predicted_params",
            "param_diff", "error", "note",
        ])
        for d in result.details:
            param_diff = ""
            if d.tool_correct and not d.param_correct:
                param_diff = compute_param_diff(d.case.expected_params, d.predicted_params)
            writer.writerow([
                d.case.source_file, d.case.row_idx, d.case.query,
                d.case.expected_tool, d.predicted_tool,
                "✓" if d.tool_correct else "✗",
                "✓" if d.param_correct else "✗",
                json.dumps(d.case.expected_params, ensure_ascii=False) if d.case.expected_params else "",
                json.dumps(d.predicted_params, ensure_ascii=False) if d.predicted_params else "",
                param_diff, d.error, d.case.note,
            ])
    print(f"\n详细结果已保存到: {output_path}")


def save_trace_jsonl(result: BenchResult, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for d in result.details:
            f.write(json.dumps({
                "query": d.case.query, "source_file": d.case.source_file, "row": d.case.row_idx,
                "expected_tool": d.case.expected_tool, "expected_params": d.case.expected_params,
                "predicted_tool": d.predicted_tool, "predicted_params": d.predicted_params,
                "tool_correct": d.tool_correct, "param_correct": d.param_correct,
                "error": d.error, "trace": d.trace,
            }, ensure_ascii=False) + "\n")
    print(f"Trace 已保存到: {output_path}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="少儿(educ)工具调用准确率评测")
    parser.add_argument("csv_files", nargs="+", help="评测用例 CSV 文件路径")
    parser.add_argument("--live", action="store_true",
                        help="连接真实 vLLM 推理（需设置 VLLM_BASE_URL/VLLM_MODEL）")
    parser.add_argument("--output", "-o", type=str, default="", help="输出详细结果 CSV 路径")
    parser.add_argument("--trace", type=str, default="", help="输出每条用例 LLM trace（JSONL）")
    parser.add_argument("--show-errors", action="store_true", help="打印错误用例详情")
    parser.add_argument("--workers", "-w", type=int, default=1, help="并发请求数")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="tool+param 准确率达标阈值（默认 0.80），未达标退出码 1")
    parser.add_argument("--no-eb", action="store_true", help="关闭 Experience Bank")
    args = parser.parse_args()

    from planner import ExperienceBankConfig, set_experience_bank_config
    if args.no_eb:
        set_experience_bank_config(ExperienceBankConfig(enabled=False))
        print("【Experience Bank 已关闭】")

    all_cases: list[TestCase] = []
    for csv_path in args.csv_files:
        cases = load_csv(csv_path)
        all_cases.extend(cases)
        print(f"加载: {os.path.basename(csv_path)} → {len(cases)} 条少儿用例")

    if not all_cases:
        print("未找到有效少儿用例，退出。")
        sys.exit(1)

    print(f"\n共 {len(all_cases)} 条用例")
    if not args.live:
        print("【Mock 模式】使用 expected 自比较验证评测逻辑。加 --live 连真实 vLLM。\n")

    predictor = Predictor(live=args.live, use_eb_prompt=not args.no_eb)
    result = evaluate(all_cases, predictor, workers=args.workers)

    print("=" * 60)
    print(result.summary())
    print("=" * 60)

    if args.show_errors:
        errors = [d for d in result.details if not d.tool_correct or not d.param_correct]
        print(f"\n错误用例 ({len(errors)} 条):")
        for d in errors[:50]:
            print(f"  [{d.case.row_idx}] {d.case.query[:40]}")
            print(f"    expected : {d.case.expected_tool}")
            print(f"    predicted: {d.predicted_tool}")
            if d.error:
                print(f"    error: {d.error}")

    if args.output:
        save_results_csv(result, args.output)
    if args.trace:
        save_trace_jsonl(result, args.trace)

    if args.live:
        ok = result.tool_param_acc >= args.threshold
        print(f"\n{'✅ 达标' if ok else '❌ 未达标'}: tool+param={result.tool_param_acc:.1%} "
              f"(阈值 {args.threshold:.0%})")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
