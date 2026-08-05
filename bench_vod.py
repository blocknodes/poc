#!/usr/bin/env python3
"""bench_vod.py —— 影视工具调用准确率评测。

用法：
    # 评测单个 CSV
    python bench_vod.py test_set/AIOS交互新架构POC影视评测用例集\ \ -\ 快链路工具识别准确率用例集（7.28版本）.csv

    # 评测多个 CSV（结果合并）
    python bench_vod.py test_set/*.csv

    # 连真实 vLLM（默认用 mock 离线模式）
    VLLM_BASE_URL=http://host:8000/v1 VLLM_MODEL=your-30b-moe python bench_vod.py test_set/*.csv --live

    # 输出详细结果到 CSV
    python bench_vod.py test_set/*.csv --output results.csv

指标：
  1. tool_acc: 工具名准确率
  2. tool_param_acc: 工具名 + 参数完全一致准确率

判定规则：
  - vod_search 和 vod_search_all 视为同一类工具（互相兼容）。
  - vod_slow_search_data_search / vod_fuzzy_search 包含 vod_search / vod_search_all 的能力：
    * 如果 expected=vod_search/vod_search_all 但 predicted=vod_slow_search_data_search 或 vod_fuzzy_search，tool 算对。
    * 反之不行（expected=vod_slow_search/vod_fuzzy_search 但 predicted=vod_search，算错）。
  - vod_slow_search_data_search / vod_fuzzy_search 的参数只有 query（原文），所以 tool 对了 param 就自动对。
  - vod_personalized_search / vod_history：参数比较用 JSON 深度相等。
  - vod_search / vod_search_all / vod_relate_search：参数比较用结构化 JSON 深度相等
    （忽略 retext 字段，因为 retext 是原始文本，模型不产出）。
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


def _read_rows(path: str):
    """读取 CSV 或 xlsx 文件，返回 DictReader 兼容的 dict 迭代器。"""
    # 检测是否为 xlsx（PK magic bytes）
    with open(path, "rb") as fb:
        magic = fb.read(4)
    if magic == b"PK\x03\x04":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(next(rows_iter))]
        for row in rows_iter:
            yield dict(zip(headers, [str(v).strip() if v is not None else "" for v in row]))
        wb.close()
    else:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            yield from reader


def load_csv(path: str) -> list[TestCase]:
    """从评测用例 CSV/xlsx 加载测试用例。"""
    cases: list[TestCase] = []
    for i, row in enumerate(_read_rows(path)):
        domain = (row.get("业务域") or "").strip()
        if domain and domain != "影视":
            continue  # 只处理影视域

        query = (row.get("query") or "").strip()
        if not query:
            continue

        expected_tool = (row.get("期望工具") or "").strip()
        if not expected_tool:
            continue

        params_raw = (row.get("期望参数（对于慢链路部分语料 也给出了如果进入快链路的参数）") or "").strip()
        expected_params = None
        if params_raw:
            try:
                expected_params = json.loads(params_raw)
            except json.JSONDecodeError:
                pass  # 无法解析的参数跳过

        cases.append(TestCase(
            query=query,
            expected_tool=expected_tool,
            expected_params=expected_params,
            source_file=os.path.basename(path),
            row_idx=i + 2,  # CSV 行号（1-indexed header + 1-indexed data）
            note=(row.get("说明") or "").strip(),
        ))
    return cases


# ===========================================================================
# 工具名匹配判定
# ===========================================================================
# vod_search 和 vod_search_all 互相兼容
_SEARCH_GROUP = {"vod_search", "vod_search_all"}

# vod_fuzzy_search（原 vod_slow_search_data_search）包含 search 的能力
_FUZZY_SEARCH = "vod_fuzzy_search"

# 在 test_set 中可能写作旧名 vod_slow_search / vod_slow_search_data_search
_FUZZY_SEARCH_ALIASES = {"vod_slow_search", "vod_slow_search_data_search", "vod_fuzzy_search"}


def normalize_tool_name(name: str) -> str:
    """统一工具名格式。"""
    name = name.strip()
    if name in _FUZZY_SEARCH_ALIASES:
        return _FUZZY_SEARCH
    return name


def tool_match(expected: str, predicted: str) -> bool:
    """判定工具名是否匹配。

    规则：
      1. vod_search ↔ vod_search_all 互相兼容
      2. expected=vod_search/vod_search_all, predicted=vod_fuzzy_search → 算对
         （fuzzy_search 包含 search 能力）
      3. 其他必须严格相等
    """
    e = normalize_tool_name(expected)
    p = normalize_tool_name(predicted)

    if e == p:
        return True

    # search 和 search_all 互相兼容
    if e in _SEARCH_GROUP and p in _SEARCH_GROUP:
        return True

    # fuzzy_search 包含 search/search_all（expected 是 search，predicted 是 fuzzy_search → 对）
    if e in _SEARCH_GROUP and p == _FUZZY_SEARCH:
        return True

    return False


# ===========================================================================
# 参数匹配判定
# ===========================================================================
def _normalize_for_compare(obj: Any) -> Any:
    """递归归一化 JSON 对象以便深度比较。

    - dict: 按 key 排序，递归
    - list: 排序后比较（顺序无关，适用于 and/or 节点的 items）
    - 其他: 原样
    """
    if isinstance(obj, dict):
        return {k: _normalize_for_compare(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        normalized = [_normalize_for_compare(item) for item in obj]
        # 用 json.dumps 序列化后排序（确保 dict 元素也能排序）
        try:
            return sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
        except TypeError:
            return normalized
    return obj


def params_match(expected_tool: str, expected_params: Optional[dict],
                 predicted_tool: str, predicted_params: Optional[dict]) -> bool:
    """判定参数是否匹配。

    规则：
      1. 如果 predicted_tool 是 vod_fuzzy_search，参数自动算对
         （因为 fuzzy_search 只传 query 原文，参数一定对）。
      2. 对于其他工具，忽略 retext 字段后做深度 JSON 相等比较。
      3. 如果 expected_params 为 None（CSV 中无期望参数），跳过参数比较（算对）。
    """
    p_tool = normalize_tool_name(predicted_tool)

    # fuzzy_search 参数自动对
    if p_tool == _FUZZY_SEARCH:
        return True

    # 无期望参数 → 跳过
    if expected_params is None:
        return True

    if predicted_params is None:
        return False

    # 忽略 retext 字段
    e = {k: v for k, v in expected_params.items() if k != "retext"}
    p = {k: v for k, v in predicted_params.items() if k != "retext"}

    return _normalize_for_compare(e) == _normalize_for_compare(p)


def compute_param_diff(expected_params: Optional[dict], predicted_params: Optional[dict]) -> str:
    """计算 expected 和 predicted params 之间的差异描述。

    返回人类可读的差异描述字符串，空字符串表示一致或无法比较。
    """
    if expected_params is None or predicted_params is None:
        return ""

    e = {k: v for k, v in expected_params.items() if k != "retext"}
    p = {k: v for k, v in predicted_params.items() if k != "retext"}

    if _normalize_for_compare(e) == _normalize_for_compare(p):
        return ""

    diffs: list[str] = []

    # 1. action 差异
    ea = e.get("action", "search")
    pa = p.get("action", "search")
    if ea != pa:
        diffs.append(f"action:{pa}→{ea}")

    # 2. sort 差异
    es = e.get("sort")
    ps = p.get("sort")
    if es != ps:
        if es and not ps:
            diffs.append(f"sort缺失(应={json.dumps(es, ensure_ascii=False)})")
        elif ps and not es:
            diffs.append(f"sort多余(pred={json.dumps(ps, ensure_ascii=False)})")
        elif es != ps:
            diffs.append(f"sort不同(应={json.dumps(es, ensure_ascii=False)},pred={json.dumps(ps, ensure_ascii=False)})")

    # 3. query 差异（字段级）
    def extract_field_values(node: Any, results: Optional[list] = None) -> list:
        """提取所有 (field, value/values, operator, negated) 元组。"""
        if results is None:
            results = []
        if isinstance(node, dict):
            if "field" in node:
                val = node.get("value", node.get("values", None))
                op = node.get("operator", "")
                results.append({"field": node["field"], "val": val, "op": op, "neg": False})
            if "not" in node:
                sub = []
                extract_field_values(node["not"], sub)
                for s in sub:
                    s["neg"] = True
                results.extend(sub)
            for key in ("and", "or"):
                if key in node and isinstance(node[key], list):
                    for item in node[key]:
                        extract_field_values(item, results)
            if "query" in node and node.get("field") is None:
                extract_field_values(node["query"], results)
        return results

    e_fields = extract_field_values(e.get("query", {}))
    p_fields = extract_field_values(p.get("query", {}))

    e_field_names = {f["field"] for f in e_fields}
    p_field_names = {f["field"] for f in p_fields}

    # 多余字段
    extra = p_field_names - e_field_names
    if extra:
        for fn in sorted(extra):
            vals = [f for f in p_fields if f["field"] == fn]
            for v in vals:
                neg_str = "NOT " if v["neg"] else ""
                diffs.append(f"多余字段:{neg_str}{fn}={v['val']}")

    # 缺失字段
    missing = e_field_names - p_field_names
    if missing:
        for fn in sorted(missing):
            vals = [f for f in e_fields if f["field"] == fn]
            for v in vals:
                neg_str = "NOT " if v["neg"] else ""
                diffs.append(f"缺失字段:{neg_str}{fn}={v['val']}")

    # 共有字段值差异
    common = e_field_names & p_field_names
    for fn in sorted(common):
        e_vals = [f for f in e_fields if f["field"] == fn]
        p_vals = [f for f in p_fields if f["field"] == fn]
        for ev in e_vals:
            # 找对应的预测值
            matched = False
            for pv in p_vals:
                if ev["neg"] == pv["neg"]:
                    if ev["val"] != pv["val"]:
                        diffs.append(f"值不同:{fn}(应={ev['val']},pred={pv['val']})")
                    elif ev["op"] != pv["op"] and (ev["op"] or pv["op"]):
                        diffs.append(f"op不同:{fn}(应={ev['op']},pred={pv['op']})")
                    matched = True
                    break
            if not matched and ev not in p_vals:
                # neg 不同
                for pv in p_vals:
                    if ev["field"] == pv["field"] and ev["val"] == pv["val"] and ev["neg"] != pv["neg"]:
                        neg_e = "NOT " if ev["neg"] else ""
                        neg_p = "NOT " if pv["neg"] else ""
                        diffs.append(f"否定不同:{neg_e}{fn}={ev['val']} vs {neg_p}{fn}={pv['val']}")

    # 4. playback 差异
    ep = e.get("playback", {})
    pp = p.get("playback", {})
    if ep != pp:
        if ep and not pp:
            diffs.append(f"playback缺失(应={ep})")
        elif pp and not ep:
            diffs.append(f"playback多余(pred={pp})")
        else:
            for k in set(list(ep.keys()) + list(pp.keys())):
                ev = ep.get(k)
                pv = pp.get(k)
                if ev != pv:
                    diffs.append(f"playback.{k}:应={ev},pred={pv}")

    if not diffs:
        # 兜底：给出 JSON diff
        diffs.append(f"结构差异")

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
    trace: list = field(default_factory=list)  # LLM 调用轨迹


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

        # 按工具分组统计
        from collections import Counter
        tool_stats: dict[str, dict] = {}
        for d in self.details:
            et = normalize_tool_name(d.case.expected_tool)
            if et not in tool_stats:
                tool_stats[et] = {"total": 0, "tool_ok": 0, "param_ok": 0}
            tool_stats[et]["total"] += 1
            if d.tool_correct:
                tool_stats[et]["tool_ok"] += 1
            if d.param_correct:
                tool_stats[et]["param_ok"] += 1

        lines.append("\n按工具细分:")
        for tool, s in sorted(tool_stats.items()):
            t_acc = s["tool_ok"] / s["total"] if s["total"] else 0
            p_acc = s["param_ok"] / s["total"] if s["total"] else 0
            lines.append(f"  {tool:40s} tool={s['tool_ok']:3d}/{s['total']:3d}={t_acc:.1%}  "
                         f"param={s['param_ok']:3d}/{s['total']:3d}={p_acc:.1%}")

        return "\n".join(lines)


# ===========================================================================
# 推理接口（支持 live / mock 模式）
# ===========================================================================
class Predictor:
    """调用 planner 进行预测。"""

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
        self._planner = Planner(client, vod_only=True,
                                use_eb_prompt=self.use_eb_prompt,
                                use_retrieve=cfg.retrieve_enabled,
                                retrieve_config=retrieve_config)

    def predict(self, query: str) -> tuple[str, Optional[dict], str, list]:
        """返回 (tool_name, params, error_msg, trace)。"""
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
    """逐条评测，返回 BenchResult。"""
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
                        case.expected_tool, case.expected_params,
                        pred_tool, pred_params,
                    )
        else:
            cr.predicted_tool = case.expected_tool
            cr.predicted_params = case.expected_params
            cr.tool_correct = tool_match(case.expected_tool, case.expected_tool)
            if cr.tool_correct:
                cr.param_correct = params_match(
                    case.expected_tool, case.expected_params,
                    case.expected_tool, case.expected_params,
                )
        return idx, cr

    def _accumulate(cr: CaseResult):
        if cr.error:
            result.errors += 1
        else:
            if cr.tool_correct:
                result.tool_correct += 1
            if cr.param_correct:
                result.tool_param_correct += 1

    if workers > 1 and predictor.live:
        # 并发模式
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_one, i, c): i for i, c in enumerate(cases)}
            done_count = 0
            for fut in as_completed(futs):
                idx, cr = fut.result()
                case_results[idx] = cr
                _accumulate(cr)
                done_count += 1
                _render_progress(done_count, result)
    else:
        # 顺序模式
        for i, case in enumerate(cases):
            _, cr = _run_one(i, case)
            case_results[i] = cr
            _accumulate(cr)
            _render_progress(i + 1, result)

    # 换行结束进度条
    sys.stderr.write("\n")
    sys.stderr.flush()

    result.details = [cr for cr in case_results if cr is not None]
    return result


def _render_progress(done: int, result: BenchResult) -> None:
    """实时刷新行内进度条。"""
    def pct(a, b):
        return f"{(a / b * 100):5.1f}%" if b else "  n/a"

    line = (f"\r[{done}/{result.total}] "
            f"tool={pct(result.tool_correct, done)} "
            f"tool+param={pct(result.tool_param_correct, done)} "
            f"err={result.errors}")
    sys.stderr.write(line)
    sys.stderr.flush()


def save_results_csv(result: BenchResult, output_path: str):
    """导出详细结果到 CSV。"""
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_file", "row", "query", "expected_tool", "predicted_tool",
            "tool_correct", "param_correct", "expected_params", "predicted_params",
            "param_diff", "error", "note",
        ])
        for d in result.details:
            # 计算参数差异描述
            param_diff = ""
            if d.tool_correct and not d.param_correct:
                param_diff = compute_param_diff(d.case.expected_params, d.predicted_params)

            writer.writerow([
                d.case.source_file,
                d.case.row_idx,
                d.case.query,
                d.case.expected_tool,
                d.predicted_tool,
                "✓" if d.tool_correct else "✗",
                "✓" if d.param_correct else "✗",
                json.dumps(d.case.expected_params, ensure_ascii=False) if d.case.expected_params else "",
                json.dumps(d.predicted_params, ensure_ascii=False) if d.predicted_params else "",
                param_diff,
                d.error,
                d.case.note,
            ])
    print(f"\n详细结果已保存到: {output_path}")


def save_trace_jsonl(result: BenchResult, output_path: str):
    """导出每条用例的 LLM trace 到 JSONL 文件。

    每行一个 JSON 对象，包含：
      - query: 用户请求
      - expected_tool: 期望工具
      - predicted_tool: 预测工具
      - tool_correct: 工具是否正确
      - param_correct: 参数是否正确
      - trace: LLM 调用轨迹数组，每个元素包含 stage/messages/output 等
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for d in result.details:
            record = {
                "query": d.case.query,
                "source_file": d.case.source_file,
                "row": d.case.row_idx,
                "expected_tool": d.case.expected_tool,
                "expected_params": d.case.expected_params,
                "predicted_tool": d.predicted_tool,
                "predicted_params": d.predicted_params,
                "tool_correct": d.tool_correct,
                "param_correct": d.param_correct,
                "error": d.error,
                "trace": d.trace,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nTrace 已保存到: {output_path}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="影视工具调用准确率评测")
    parser.add_argument("csv_files", nargs="+", help="评测用例 CSV 文件路径")
    parser.add_argument("--live", action="store_true",
                        help="连接真实 vLLM 推理（需设置 VLLM_BASE_URL/VLLM_MODEL）")
    parser.add_argument("--output", "-o", type=str, default="",
                        help="输出详细结果的 CSV 路径")
    parser.add_argument("--trace", type=str, default="",
                        help="输出每条用例的 LLM trace（JSONL 格式）")
    parser.add_argument("--show-errors", action="store_true",
                        help="打印所有错误用例详情")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="并发请求数（默认 1，live 模式下有效）")

    # Experience Bank 控制
    eb_group = parser.add_argument_group("Experience Bank",
                                         "控制 Experience Bank 各层规则开关")
    eb_group.add_argument("--no-eb", action="store_true",
                          help="完全关闭 Experience Bank（compiler层+prompt层）")
    eb_group.add_argument("--no-eb-compiler", action="store_true",
                          help="关闭 Experience Bank compiler 层（保留 prompt 层）")
    eb_group.add_argument("--no-eb-prompt", action="store_true",
                          help="关闭 Experience Bank prompt 层（保留 compiler 层）")
    eb_group.add_argument("--no-action-override", action="store_true",
                          help="关闭 action 动词覆盖规则")
    eb_group.add_argument("--no-field-remap", action="store_true",
                          help="关闭字段名修正（好莱坞→company 等）")
    eb_group.add_argument("--no-value-normalize", action="store_true",
                          help="关闭值归一化（人名间隔符/大小写等）")
    eb_group.add_argument("--no-sort-control", action="store_true",
                          help="关闭 sort 精细控制")
    eb_group.add_argument("--no-query-text-complete", action="store_true",
                          help="关闭 query-text 感知补全（影片→category 等）")
    eb_group.add_argument("--no-title-process", action="store_true",
                          help="关闭 title 处理（去后缀/拆数字）")
    eb_group.add_argument("--no-time-convert", action="store_true",
                          help="关闭时间转换（去年/今年→绝对年份）")
    eb_group.add_argument("--no-name-normalize", action="store_true",
                          help="关闭名称映射（奖项/tag/company 归一）")
    eb_group.add_argument("--no-structure-simplify", action="store_true",
                          help="关闭结构简化（and解包/not fee归一）")

    args = parser.parse_args()

    # ---- 配置 Experience Bank ----
    from planner import ExperienceBankConfig, set_experience_bank_config

    if args.no_eb:
        # 完全关闭
        set_experience_bank_config(ExperienceBankConfig(enabled=False))
        print("【Experience Bank 已关闭（compiler+prompt）】")
    elif args.no_eb_compiler:
        set_experience_bank_config(ExperienceBankConfig(enabled=False))
        print("【Experience Bank compiler 层已关闭】")
    else:
        # 按单项开关配置
        cfg = ExperienceBankConfig(
            enabled=True,
            action_override=not args.no_action_override,
            field_remap=not args.no_field_remap,
            value_normalize=not args.no_value_normalize,
            sort_control=not args.no_sort_control,
            query_text_complete=not args.no_query_text_complete,
            title_process=not args.no_title_process,
            time_convert=not args.no_time_convert,
            name_normalize=not args.no_name_normalize,
            structure_simplify=not args.no_structure_simplify,
        )
        set_experience_bank_config(cfg)
        # 打印关闭了哪些
        disabled = []
        if args.no_action_override: disabled.append("action_override")
        if args.no_field_remap: disabled.append("field_remap")
        if args.no_value_normalize: disabled.append("value_normalize")
        if args.no_sort_control: disabled.append("sort_control")
        if args.no_query_text_complete: disabled.append("query_text_complete")
        if args.no_title_process: disabled.append("title_process")
        if args.no_time_convert: disabled.append("time_convert")
        if args.no_name_normalize: disabled.append("name_normalize")
        if args.no_structure_simplify: disabled.append("structure_simplify")
        if disabled:
            print(f"【Experience Bank 部分关闭: {', '.join(disabled)}】")

    # prompt 层控制标记（Predictor 内部使用）
    use_eb_prompt = not args.no_eb and not args.no_eb_prompt

    # 加载用例
    all_cases: list[TestCase] = []
    for csv_path in args.csv_files:
        cases = load_csv(csv_path)
        all_cases.extend(cases)
        print(f"加载: {os.path.basename(csv_path)} → {len(cases)} 条影视用例")

    if not all_cases:
        print("未找到有效用例，退出。")
        sys.exit(1)

    print(f"\n共 {len(all_cases)} 条用例")
    if not args.live:
        print("【Mock 模式】使用 expected 自比较验证评测逻辑。加 --live 连真实 vLLM。\n")

    # 评测
    predictor = Predictor(live=args.live, use_eb_prompt=use_eb_prompt)
    result = evaluate(all_cases, predictor, workers=args.workers)

    # 输出
    print("=" * 60)
    print(result.summary())
    print("=" * 60)

    if args.show_errors:
        errors = [d for d in result.details if not d.tool_correct or not d.param_correct]
        if errors:
            print(f"\n错误用例 ({len(errors)} 条):")
            for d in errors[:50]:  # 最多显示 50 条
                print(f"  [{d.case.row_idx}] {d.case.query[:40]}")
                print(f"    expected: {d.case.expected_tool}")
                print(f"    predicted: {d.predicted_tool}")
                if d.error:
                    print(f"    error: {d.error}")
                print()

    if args.output:
        save_results_csv(result, args.output)

    if args.trace:
        save_trace_jsonl(result, args.trace)

    # 退出码：非 100% 时退出 1（方便 CI）
    if result.tool_acc < 1.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
