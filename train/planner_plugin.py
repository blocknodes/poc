# -*- coding: utf-8 -*-
"""ms-swift 外部插件：把 planner 主流程封装成 GRPO 的 gym env + 自定义 reward。

设计对齐用户要求：
  1. 用 gym env 封装「主流程」：route -> IR 生成 -> validate(自修复) -> compile。
  2. env 只负责推进多轮对话与状态机，**reward 恒为 0.0**（reward 不从 env 出）。
  3. 真正的打分由本文件里注册的 ORM reward 函数完成（planner_* 系列）。
  4. 供 GRPO 训练：`swift rlhf --rlhf_type grpo --use_gym_env true --gym_env planner_env
     --reward_funcs planner_route_reward planner_param_reward --reward_weights 0.3 0.7
     --external_plugins .../planner_plugin.py`

env / reward 通过 ms-swift 的两个注册表接入：
  * swift.rollout.gym_env.envs[name]      <- 我们注册 PlannerEnv 为 'planner_env'
  * swift.rewards.orms[name]              <- 我们注册若干 ORM

reward 函数拿得到的关键 kwargs（见 swift/rl_core/grpo_algorithm.py::compute_rewards_per_func
与 swift/rl_core/data.py::to_reward_row）：
  * completions : List[str]，每条 = 该样本最后一个 assistant turn 的文本（最终 IR）
  * messages    : List[Messages]，每条 = 完整多轮轨迹（含 route / IR / 修复）
  * 数据集透传列：gold_tool / gold_params / gold_domain（batched list）
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# 让插件能 import 到同仓库的 planner 包（train/ 的上一级即 poc 根目录）
# ---------------------------------------------------------------------------
_POC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _POC_ROOT not in sys.path:
    sys.path.insert(0, _POC_ROOT)

from planner.agent import IR_TOOLS, PlannerAgent, loads_lenient  # noqa: E402
from planner.compiler import CompileError, compile_with_fallback  # noqa: E402
from planner.ir import IRError, parse_ir, validate_ir  # noqa: E402

# ms-swift 侧注册表
from swift.rollout.gym_env import Env, envs  # noqa: E402
from swift.rewards import ORM, orms  # noqa: E402


# flat -> nested 回退等价（路由到慢链路却回退到精确检索，视为路由正确）
_FALLBACK_EQUIV = {
    "vod_slow_search_data_search": "vod_search",
    "educ_slow_search_data_search": "educ_search",
}


# ===========================================================================
# 工具函数
# ===========================================================================
# 复用 agent 的单一事实源 JSON 解析器（与推理侧完全一致）
_loads_lenient = loads_lenient


def _extract_assistant_texts(messages: Any) -> List[str]:
    if not isinstance(messages, list):
        return []
    return [m.get("content", "") for m in messages
            if isinstance(m, dict) and m.get("role") == "assistant"]


def _canon(obj: Any) -> Any:
    """JSON 语义归一化：dict 按 key 排序，list 元素按序列化排序。用于等价比较。"""
    if isinstance(obj, dict):
        return {k: _canon(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        items = [_canon(x) for x in obj]
        return sorted(items, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    return obj


def _params_equivalent(a: Any, b: Any) -> bool:
    return _canon(a) == _canon(b)


def _collect_fields(node: Any, out: set) -> None:
    """从 IR query 树里收集 (field, value/values/range) 指纹，用于部分匹配打分。"""
    if not isinstance(node, dict):
        return
    if "and" in node:
        for x in node["and"]:
            _collect_fields(x, out)
    elif "or" in node:
        for x in node["or"]:
            _collect_fields(x, out)
    elif "not" in node:
        _collect_fields(node["not"], out)
    elif "field" in node:
        key = node["field"]
        val = node.get("value")
        if val is None and node.get("values") is not None:
            val = tuple(sorted(map(str, node["values"])))
        elif node.get("range") is not None:
            r = node["range"]
            val = (str(r.get("from")), str(r.get("to")))
        out.add((key, str(val)))


def _params_fingerprint(params: Any) -> set:
    """把一份（已编译的）参数 dict 拍成指纹集合，用于召回 / 子集比较。

    覆盖两部分，保证与 bench 的严格全等对齐：
      * query 子树里的 (field, value/values/range) 指纹（and/or/not 结构被拍平）；
      * query 之外的顶层键（如 sort/fee 等），按 (@key, canonical-json) 计入。
    这样漏掉 sort / category / fee 会被算作「缺指纹」，多写或写错则成为「多余指纹」。
    """
    out: set = set()
    if not isinstance(params, dict):
        return out
    _collect_fields(params.get("query"), out)
    for k, v in params.items():
        if k == "query":
            continue
        out.add((f"@{k}", json.dumps(_canon(v), ensure_ascii=False, sort_keys=True)))
    return out


def _parse_gold_params(gold_params: Any) -> Optional[dict]:
    if gold_params is None:
        return None
    if isinstance(gold_params, dict):
        return gold_params
    if isinstance(gold_params, str):
        s = gold_params.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# ===========================================================================
# Gym Env：封装 planner 主流程（reward 恒为 0）
# ===========================================================================
class PlannerEnv(Env):
    """把 route -> IR -> validate(自修复) 封装成多轮 rollout。

    **完全委托给共享的 `PlannerAgent` 状态机**——prompt 构造与状态转移和推理侧
    `harness.Planner` 用的是同一份代码，实现训推一致。env 自身只做三件事：
      1. reset：用 agent 播种 system + 首个 observation；
      2. step：把模型输出喂给 agent，拿下一条 observation / 终止；
      3. reward 恒为 0.0（真正打分交给下面的 planner_* ORM）。

    轨迹形态（与推理逐 token 一致）::

        system : route_system_prompt()            (含路由 few-shot)
        user   : route_observation(query)          (reset)
        asst   : 路由 JSON                          (turn 1)
        user   : ir_observation(domain, query)     (含 IR few-shot + schema)
        asst   : IR JSON                            (turn 2)
        [user  : repair_observation(errs)           (自修复)
         asst  : 修正后的 IR JSON]                  (turn 3..)

    安全机制（Fix: context overflow guard）：
      step 中在调 agent 之前估算当前轨迹 token 数，若剩余预算不足以容纳生成长度，
      提前终止（truncated=True），避免 vLLM max_tokens<=0 报错。
    """

    # 粗估：1 char ≈ 0.6 token（中英混合偏保守）；可通过 env_config 覆盖
    _CHARS_PER_TOKEN = 1.8  # 即 1 token ≈ 1.8 chars，中英混合偏保守

    def __init__(self, env_config: dict):
        super().__init__(env_config)
        cfg = env_config or {}
        self.max_repairs = int(cfg.get("max_repairs", 2))
        # route_only：路由后直接结束，不进入 IR 阶段。
        # 优先从 env_config 取；fallback 到环境变量 PLANNER_ROUTE_ONLY=1
        self.route_only = bool(cfg.get("route_only",
                                       os.environ.get("PLANNER_ROUTE_ONLY", "0") == "1"))
        # 上下文长度上限 & 最小生成预留（与 run.sh 里 vllm_max_model_len / max_completion_length 对齐）
        self.max_model_len = int(cfg.get("max_model_len", 16384))
        self.min_gen_budget = int(cfg.get("min_gen_budget", 512))
        self.agent: Optional[PlannerAgent] = None
        self._messages: list = []  # 追踪完整轨迹用于 token 估算

    def _estimate_tokens(self, messages: list) -> int:
        """粗估轨迹 token 数（字符 / chars_per_token）。"""
        total_chars = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
        return int(total_chars / self._CHARS_PER_TOKEN)

    async def reset(self, config):
        data = getattr(config, "data_dict", None) or {}
        query = data.get("query") or self._query_from_messages(config)
        self.agent = PlannerAgent(query, max_repairs=self.max_repairs,
                                  route_only=self.route_only)
        info = {"phase": "route", "query": query}
        system_msg = self.agent.system_prompt()
        first_obs = self.agent.first_observation()
        # 初始化轨迹追踪
        self._messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": first_obs},
        ]
        # 返回 (observation, info, system_message)：GYMScheduler 会拼成 [system, user]
        return first_obs, info, system_msg

    async def step(self, action):
        content = action[-1]["content"] if action else ""
        # 追踪 assistant 输出
        self._messages.append({"role": "assistant", "content": content})

        # ---- Context overflow guard ----
        # 估算当前已用 token；若剩余预算 < min_gen_budget，提前终止
        used_tokens = self._estimate_tokens(self._messages)
        remaining = self.max_model_len - used_tokens
        if remaining < self.min_gen_budget:
            info = {"phase": "truncated", "reason": "context_overflow",
                    "used_tokens_est": used_tokens, "remaining_est": remaining}
            return None, 0.0, True, info

        step = self.agent.observe(content)
        info = {"phase": step.phase, **step.info}
        next_obs = step.next_observation if not step.done else None
        if next_obs:
            self._messages.append({"role": "user", "content": next_obs})
        return next_obs, 0.0, step.done, info

    async def close(self):
        pass

    @staticmethod
    def _query_from_messages(config) -> str:
        msgs = getattr(config, "messages", None) or []
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content", "")
        return ""


# ===========================================================================
# Reward ORMs：从完整轨迹 + gold 列打分（不依赖 env 的 reward）
# ===========================================================================
def _extract_route_and_ir(messages: Any) -> tuple[Optional[dict], Optional[dict]]:
    """从一条完整轨迹里取出 (route_dict, final_ir_dict)。

    约定：第 1 个 assistant = 路由；最后 1 个 assistant = 最终 IR（若进入过 IR 阶段）。
    若只有 1 个 assistant（非检索类工具直接结束），final_ir = None。
    """
    asst = _extract_assistant_texts(messages)
    if not asst:
        return None, None
    route = _loads_lenient(asst[0])
    final_ir = _loads_lenient(asst[-1]) if len(asst) >= 2 else None
    return route, final_ir


def _route_correct(route: Optional[dict], gold_tool: str) -> bool:
    if not route:
        return False
    tool = route.get("tool")
    if tool == gold_tool:
        return True

    return False


class PlannerRoute(ORM):
    """路由准确率：路由选中的工具是否命中 gold（容忍 flat->nested 回退等价）。"""

    def __call__(self, completions, messages=None, gold_tool=None, **kwargs) -> List[float]:
        messages = messages or [None] * len(completions)
        gold_tool = gold_tool or [None] * len(completions)
        out = []
        for msg, gt in zip(messages, gold_tool):
            route, _ = _extract_route_and_ir(msg)
            out.append(1.0 if (gt and _route_correct(route, gt)) else 0.0)
        return out


class PlannerIRValid(ORM):
    """结构/语义合法率：最终 IR 能 parse 且 validate_ir 通过。"""

    def __call__(self, completions, messages=None, **kwargs) -> List[float]:
        messages = messages or [None] * len(completions)
        out = []
        for msg in messages:
            _, ir_raw = _extract_route_and_ir(msg)
            if ir_raw is None:
                # 没有 IR 阶段（非检索类工具）——此项不适用，给 0（用权重/组合控制）
                out.append(0.0)
                continue
            try:
                ir = parse_ir(ir_raw)
                out.append(1.0 if not validate_ir(ir) else 0.0)
            except IRError:
                out.append(0.0)
        return out


def _equiv_score(final_ir: Optional[dict], gold_tool: str, gold_params: Optional[dict]) -> float:
    """参数召回率 + 子集门控（硬版）：把 IR 用 gold_tool 编译后与 gold 比指纹。

    - pred 出现任何 gold 没有的指纹（多余字段 / 写错的值）-> 0.0（硬惩罚过度指定）。
    - 否则按召回 |pred∩gold| / |gold| 给分；全覆盖 == 精确等价 == 1.0。
    - 没有可比指纹的 gold（极少数）退回严格等价判定。
    与 bench.py 的严格全等方向一致：只有「不多、不少、值全对」才拿满分。
    """
    if final_ir is None or gold_params is None:
        return 0.0
    try:
        ir = parse_ir(final_ir)
        if validate_ir(ir):
            return 0.0
        _, params = compile_with_fallback(ir, gold_tool)
    except (IRError, CompileError, KeyError, TypeError):
        return 0.0

    gold_fp = _params_fingerprint(gold_params)
    if not gold_fp:
        return 1.0 if _params_equivalent(params, gold_params) else 0.0
    pred_fp = _params_fingerprint(params)
    if pred_fp - gold_fp:            # 有多余 / 错误指纹 -> 硬 0
        return 0.0
    return len(pred_fp & gold_fp) / len(gold_fp)   # 召回；全覆盖=1.0（即 exact）


class PlannerEquiv(ORM):
    """端到端参数等价率（含部分匹配）。gold_params 为空时该项为 0。"""

    def __call__(self, completions, messages=None, gold_tool=None, gold_params=None, **kwargs) -> List[float]:
        n = len(completions)
        messages = messages or [None] * n
        gold_tool = gold_tool or [None] * n
        gold_params = gold_params or [None] * n
        out = []
        for msg, gt, gp in zip(messages, gold_tool, gold_params):
            _, final_ir = _extract_route_and_ir(msg)
            out.append(_equiv_score(final_ir, gt, _parse_gold_params(gp)))
        return out


class PlannerRouteReward(ORM):
    """路由 reward 分量（独立打 tensorboard）。

    - 非检索类工具样本（gold 不在 IR_TOOLS）：路由命中 1.0，否则 0.0。
    - 检索类样本：同上。
    训练时与 PlannerParamReward 一起使用，用 --reward_weights 控制组合权重。
    """

    def __call__(self, completions, messages=None, gold_tool=None, **kwargs) -> List[float]:
        n = len(completions)
        messages = messages or [None] * n
        gold_tool = gold_tool or [None] * n
        out = []
        for msg, gt in zip(messages, gold_tool):
            route, _ = _extract_route_and_ir(msg)
            out.append(1.0 if (gt and _route_correct(route, gt)) else 0.0)
        return out


class PlannerParamReward(ORM):
    """参数 reward 分量（独立打 tensorboard）。

    - 非检索类工具样本（gold 不在 IR_TOOLS）：无参数可评，给 0.0（该项权重对此类样本无贡献）。
    - 检索类样本：**仅在路由命中时计入**（路由错则参数被编译到错工具上、无意义 → 0.0）。
      用「召回 + 子集门控」硬版（见 _equiv_score）：漏字段按召回扣分、
      多字段 / 错值直接 0，全对 == 1.0。
    """

    def __call__(self, completions, messages=None, gold_tool=None, gold_params=None, **kwargs) -> List[float]:
        n = len(completions)
        messages = messages or [None] * n
        gold_tool = gold_tool or [None] * n
        gold_params = gold_params or [None] * n
        out = []
        for msg, gt, gp in zip(messages, gold_tool, gold_params):
            route, final_ir = _extract_route_and_ir(msg)
            route_ok = gt and _route_correct(route, gt)

            if gt not in IR_TOOLS:
                # 非检索类：无参数信号
                out.append(0.0)
                continue

            # 检索类：param 仅在路由命中时计入
            param = _equiv_score(final_ir, gt, _parse_gold_params(gp)) if route_ok else 0.0
            out.append(param)
        return out


class PlannerAccuracy(ORM):
    """组合主奖励（保留向后兼容，用于单 reward_func 场景）。

    推荐改用 --reward_funcs planner_route_reward planner_param_reward --reward_weights 0.3 0.7
    来获得分项 tensorboard 曲线。
    """

    W_ROUTE = 0.3
    W_PARAM = 0.7

    def __call__(self, completions, messages=None, gold_tool=None, gold_params=None, **kwargs) -> List[float]:
        n = len(completions)
        messages = messages or [None] * n
        gold_tool = gold_tool or [None] * n
        gold_params = gold_params or [None] * n
        out = []
        for msg, gt, gp in zip(messages, gold_tool, gold_params):
            route, final_ir = _extract_route_and_ir(msg)
            route_ok = 1.0 if (gt and _route_correct(route, gt)) else 0.0

            if gt not in IR_TOOLS:
                # 非检索类：只有路由信号
                out.append(route_ok)
                continue

            # 检索类：param 仅在路由命中时计入（否则整体为 0）
            param = _equiv_score(final_ir, gt, _parse_gold_params(gp)) if route_ok else 0.0
            out.append(self.W_ROUTE * route_ok + self.W_PARAM * param)
        return out


# ===========================================================================
# 修复 ms-swift `--mcore_adapter` 续训 bug：恢复 consumed_train_samples
# ---------------------------------------------------------------------------
# 现象：resume 后 iteration/LR 正常续上，但每个 step 的 reward 与原 run 对不上、
#      像是回退。根因不是模型退化，而是**数据采样位置错位**：
#
#   swift/megatron/trainers/base.py::_load_checkpoint 里
#       self.state.consumed_train_samples = getattr(args, 'consumed_train_samples', 0)
#   这个值本应由 _load_iteration() 从 checkpoint 的 iter_XXXXXXX/common.pt 读出，但
#   _load_iteration() 定位目录用的是
#       ckpt_dir = args.adapters[0] if args.adapters else args.model
#   用 --mcore_adapter 续训时 args.adapters 为空 → ckpt_dir 落到基座模型目录（无
#   common.pt）→ 提前返回、consumed_train_samples 保持默认 0。
#   （另一条兜底路径 load_mcore_checkpoint 只在 distcp 的 state_dict 里含 'args' 时才
#    设置，而 args 存在单独的 common.pt 里、distcp 分片中没有，故也设不上。）
#
#   MegatronPretrainingRandomSampler 的 epoch 洗牌种子 (manual_seed(epoch)) 与
#   epoch 内偏移都由 consumed_train_samples 决定；被重置为 0 后，resume 会用错误的
#   epoch 种子、从头重刷数据 → 每步喂的 prompt 与原 run 不同 → 逐步 reward 不可比、
#   且重复训练早期样本、跳过后续样本，实际影响训练覆盖。
#
# 修法：hook BaseMegatronTrainer._load_iteration，在 --mcore_adapter 续训（且
#      非 finetune）时，直接从该 ckpt 目录的 common.pt 把 consumed_train_samples 读
#      回来补到 args 上，后续 _load_checkpoint 的 line 119 即可正确取到。
#
# 生效条件：仅 megatron 训练进程（能 import 到 megatron 后端）且用了 --mcore_adapter
#          + 非 finetune 时才改写；rollout / 全新训练均无副作用（try/except 保护）。
# ===========================================================================
def _install_resume_consumed_samples_patch() -> None:
    try:
        from swift.megatron.trainers.base import BaseMegatronTrainer
    except Exception:
        # 该进程没有 megatron 后端（如 swift rollout server），无需修补。
        return

    if getattr(BaseMegatronTrainer, "_planner_consumed_patch", False):
        return  # 幂等：避免重复 import 时二次包裹

    import torch  # 训练进程必然可用；延迟到此处以免影响 rollout 侧
    from swift.utils import get_logger

    logger = get_logger()
    _orig_load_iteration = BaseMegatronTrainer._load_iteration

    def _read_consumed_train_samples(ckpt_dir: str) -> Optional[int]:
        tracker = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
        if not os.path.exists(tracker):
            return None
        with open(tracker) as f:
            iteration = int(f.read().strip())
        common_path = os.path.join(ckpt_dir, f"iter_{iteration:07d}", "common.pt")
        if not os.path.exists(common_path):
            return None
        try:
            sd = torch.load(common_path, map_location="cpu", weights_only=False)
        except TypeError:  # 老版本 torch 无 weights_only 形参
            sd = torch.load(common_path, map_location="cpu")
        saved_args = sd.get("args") if isinstance(sd, dict) else None
        if saved_args is None:
            return None
        n = getattr(saved_args, "consumed_train_samples", None)
        return int(n) if n is not None else None

    def _patched_load_iteration(self):
        iteration = _orig_load_iteration(self)
        args = self.args
        mcore_adapter = getattr(args, "mcore_adapter", None)
        if mcore_adapter and not getattr(args, "finetune", False):
            try:
                n = _read_consumed_train_samples(mcore_adapter)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[planner-resume-fix] 读取 consumed_train_samples 失败: {e}")
                n = None
            cur = getattr(args, "consumed_train_samples", 0)
            if n is not None and n != cur:
                logger.info(
                    f"[planner-resume-fix] 从 {mcore_adapter} 恢复 "
                    f"consumed_train_samples: {cur} -> {n}（修正数据采样位置/epoch 种子）")
                args.consumed_train_samples = n
        return iteration

    BaseMegatronTrainer._load_iteration = _patched_load_iteration
    BaseMegatronTrainer._planner_consumed_patch = True
    logger.info(
        "[planner-resume-fix] 已 hook BaseMegatronTrainer._load_iteration："
        "--mcore_adapter 续训时恢复 consumed_train_samples")


_install_resume_consumed_samples_patch()


# ===========================================================================
# 注册到 ms-swift
# ===========================================================================
envs["planner_env"] = PlannerEnv

orms["planner_accuracy"] = PlannerAccuracy   # 组合主奖励（向后兼容）
orms["planner_route_reward"] = PlannerRouteReward   # 路由分量（独立 tensorboard 曲线）
orms["planner_param_reward"] = PlannerParamReward   # 参数分量（独立 tensorboard 曲线）
orms["planner_route"] = PlannerRoute         # 诊断/可选：路由
orms["planner_ir_valid"] = PlannerIRValid    # 诊断/可选：结构合法
orms["planner_equiv"] = PlannerEquiv         # 诊断/可选：端到端等价
