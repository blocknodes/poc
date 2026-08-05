#!/usr/bin/env python3
"""bench_device.py —— 设备（整机控制）工具调用准确率评测。

用法：
    # 评测 CSV（默认路径）
    python bench_device.py path/to/设备评测用例集.csv

    # 连真实 vLLM
    VLLM_BASE_URL=http://host:8000/v1 VLLM_MODEL=your-30b-moe \
      python bench_device.py test_set/设备用例.csv --live --output results_device.csv --workers 16

指标：
  1. tool_acc: 工具名准确率
  2. tool_param_acc: 工具名 + 参数完全一致准确率

判定规则：
  - 工具名严格相等。
  - 参数比较：忽略空字符串字段（date_time/device/location 通常为空），
    对非空字段做 JSON 深度相等比较（忽略前后空格、大小写统一视具体字段而定）。
  - ai_picture_sound_control 的参数只有 intent 字段，直接字符串比较。
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


def load_csv(path: str) -> list[TestCase]:
    """从设备评测用例 CSV 加载测试用例。

    CSV 列格式：业务域,query,原产品意图,原智能体意图,期望工具,期望参数,...
    """
    cases: list[TestCase] = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            domain = (row.get("业务域") or "").strip()
            # 设备域标记为 CONTROL 或 设备
            if domain and domain not in ("CONTROL", "设备", "device"):
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
                    # 有些行可能不是合法 JSON（如纯文本 intent）
                    # 尝试包装为 intent 字段
                    expected_params = {"intent": params_raw}

            cases.append(TestCase(
                query=query,
                expected_tool=expected_tool,
                expected_params=expected_params,
                source_file=os.path.basename(path),
                row_idx=i + 2,  # CSV 行号（1-indexed header + 1-indexed data）
                note=(row.get("说明") or row.get("测试备注") or "").strip(),
            ))
    return cases


# ===========================================================================
# 工具名匹配判定
# ===========================================================================
def tool_match(expected: str, predicted: str) -> bool:
    """判定工具名是否匹配。设备域先做别名归一（mode_control≡scene_mode_control）再严格比较。"""
    try:
        from planner.registry import canonical_device_tool
        return canonical_device_tool(expected) == canonical_device_tool(predicted)
    except Exception:
        return expected.strip() == predicted.strip()


# ===========================================================================
# 参数匹配判定
# ===========================================================================
def _normalize_params(params: dict) -> dict:
    """归一化参数：对称地丢弃所有空字符串 / None 字段，strip 并**小写**所有字符串值。

    设备编译器 compile_device 只会输出**非空**的 operation/object/value（value 为空
    时直接省略，且从不输出 date_time/device/location）。而评测集的期望参数里普遍带有
    value:""、device:""、location:""（855 条 value 为空）。若不对称丢弃这些空字段，
    live 模式下这批本应判对的用例会全部被误判为参数不一致。

    另外：参数值**大小写不敏感**（HDMI1≡hdmi1、WAVES音效≡waves音效），故统一小写后比较。
    因此这里统一规则：任何值为空字符串或 None 的字段都视为"不存在"，只对非空字段做小写化后的
    深度相等比较。这样 expected 与 predicted 两侧的归一化口径完全一致。

    额外归一化（修复评测集标注不一致）：
      - value="默认" 视为空（评测集部分用例标注"默认"作为缺省值占位符，模型正确地不输出该值）
      - "开启"/"启动" → "打开"（operation 同义词）
    """
    result = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, str):
            v = v.strip().lower()
            if not v:
                continue
            # "默认" 在评测集中用作缺省值占位，视为空
            if k == "value" and v == "默认":
                continue
        result[k] = v
    # operation 同义词归一（评测集标注口径不一致的处理）
    op = result.get("operation")
    if op in ("开启", "启动"):
        result["operation"] = "打开"
    return result


def params_match(expected_params: Optional[dict], predicted_params: Optional[dict]) -> bool:
    """判定参数是否匹配。

    规则：
      1. 如果 expected_params 为 None，跳过参数比较（算对）。
      2. 空字符串/None 字段（value:""、date_time:""、device:""、location:"" 等）统一丢弃。
      3. value="默认" 视为空。
      4. 非空字段做 JSON 深度相等。
      5. 额外容忍规则（修复评测集标注不一致）：
         - 时间 value：去前导零后相等（01:30 ≡ 1:30）
         - operation: 提高/增加≡提高，降低/减小≡降低
    """
    if expected_params is None:
        return True
    if predicted_params is None:
        return False

    e = _normalize_params(expected_params)
    p = _normalize_params(predicted_params)

    if e == p:
        return True

    # 额外容忍：时间 value 去前导零
    if e.keys() == p.keys():
        all_match = True
        for k in e:
            ev, pv = e[k], p[k]
            if ev == pv:
                continue
            # 时间格式容忍：去前导零
            if k == "value" and isinstance(ev, str) and isinstance(pv, str):
                import re
                ev_stripped = re.sub(r'^0+(\d)', r'\1', ev)
                pv_stripped = re.sub(r'^0+(\d)', r'\1', pv)
                if ev_stripped == pv_stripped:
                    continue
                # 时长容忍：2分 ≡ 2分钟
                if ev.rstrip("钟") == pv.rstrip("钟"):
                    continue
            all_match = False
            break
        if all_match:
            return True

    return False


def compute_param_diff(expected_params: Optional[dict], predicted_params: Optional[dict]) -> str:
    """计算参数差异描述。"""
    if expected_params is None or predicted_params is None:
        return ""

    e = _normalize_params(expected_params)
    p = _normalize_params(predicted_params)

    if e == p:
        return ""

    diffs: list[str] = []

    all_keys = sorted(set(list(e.keys()) + list(p.keys())))
    for k in all_keys:
        ev = e.get(k)
        pv = p.get(k)
        if ev != pv:
            if ev is not None and pv is None:
                diffs.append(f"缺失:{k}={ev}")
            elif ev is None and pv is not None:
                diffs.append(f"多余:{k}={pv}")
            else:
                diffs.append(f"{k}:应={ev},pred={pv}")

    return "; ".join(diffs) if diffs else "结构差异"


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

        # 按工具分组统计
        tool_stats: dict[str, dict] = {}
        for d in self.details:
            et = d.case.expected_tool
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
# 推理接口
# ===========================================================================
class Predictor:
    """调用 planner 进行设备域预测。"""

    def __init__(self, live: bool = False):
        self.live = live
        self._planner = None
        if live:
            self._init_live()

    def _init_live(self):
        from config import cfg, make_vllm_config, make_retrieve_config
        from planner import Planner, VLLMClient
        client = VLLMClient(make_vllm_config())
        retrieve_config = make_retrieve_config()
        self._planner = Planner(client,
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
                    cr.param_correct = params_match(case.expected_params, pred_params)
        else:
            # Mock 模式：自比较验证评测逻辑
            cr.predicted_tool = case.expected_tool
            cr.predicted_params = case.expected_params
            cr.tool_correct = True
            cr.param_correct = params_match(case.expected_params, case.expected_params)
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
    """导出每条用例的 LLM trace 到 JSONL。"""
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
    parser = argparse.ArgumentParser(description="设备（整机控制）工具调用准确率评测")
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

    args = parser.parse_args()

    # 加载用例
    all_cases: list[TestCase] = []
    for csv_path in args.csv_files:
        cases = load_csv(csv_path)
        all_cases.extend(cases)
        print(f"加载: {os.path.basename(csv_path)} → {len(cases)} 条设备用例")

    if not all_cases:
        print("未找到有效用例，退出。")
        sys.exit(1)

    print(f"\n共 {len(all_cases)} 条用例")
    if not args.live:
        print("【Mock 模式】使用 expected 自比较验证评测逻辑。加 --live 连真实 vLLM。\n")

    # 评测
    predictor = Predictor(live=args.live)
    result = evaluate(all_cases, predictor, workers=args.workers)

    # 输出
    print("=" * 60)
    print(result.summary())
    print("=" * 60)

    if args.show_errors:
        errors = [d for d in result.details if not d.tool_correct or not d.param_correct]
        if errors:
            print(f"\n错误用例 ({len(errors)} 条):")
            for d in errors[:50]:
                print(f"  [{d.case.row_idx}] {d.case.query[:40]}")
                print(f"    expected: {d.case.expected_tool} | {json.dumps(d.case.expected_params, ensure_ascii=False) if d.case.expected_params else ''}")
                print(f"    predicted: {d.predicted_tool} | {json.dumps(d.predicted_params, ensure_ascii=False) if d.predicted_params else ''}")
                if d.error:
                    print(f"    error: {d.error}")
                print()

    if args.output:
        save_results_csv(result, args.output)

    if args.trace:
        save_trace_jsonl(result, args.trace)

    # 退出码
    if result.tool_acc < 1.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
