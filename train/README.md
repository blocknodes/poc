# train/ —— 用 gym env 封装 planner 主流程，接入 ms-swift 做 GRPO（Megatron 后端）

把 `planner` 的**主流程**（route → IR 生成 → validate 自修复 → compile）封装成一个
ms-swift 的 **gym env**，用 **Megatron 后端的 GRPO**（`megatron rlhf --rlhf_type grpo`）
训练 `qwen3.5-35B-A`（MoE）。

关键取舍（对齐需求）：

- **reward 不从 env 出**：`PlannerEnv` 每一步都返回 `reward=0.0`，只负责推进多轮对话与状态机。
  真正的打分完全由本目录注册的 **ORM reward 函数**（`planner_*`）在训练侧完成。
- gym env 的多轮 rollout 走 **server 模式**：先起 `swift rollout` server，再跑 `megatron rlhf`。
- **训练后端 = Megatron**：35B MoE 全参 RL 用 Megatron 的 5D 并行（TP/PP/EP/CP/DP）+
  权重/优化器 offload，比 deepspeed zero3 更省显存、吞吐更高（参考
  `../ms-swift/examples/megatron/grpo`）。rollout 侧（`swift rollout`，vLLM）与后端无关，不变。

## 目录

```
train/
├── planner_plugin.py   # 核心：注册 PlannerEnv(gym env) + planner_* reward(ORM)
├── build_dataset.py    # benchmark CSV -> GRPO gym-env jsonl（含 gold 列）
├── run.sh              # 一体化启动脚本：`run.sh rollout` 起 server / `run.sh train` 开训
└── data/               # build_dataset.py 输出 train.jsonl / val.jsonl
```

## 训推一致（关键）

`PlannerEnv` 不再自己拼 prompt，而是**完全委托** `planner/agent.py::PlannerAgent`——
推理侧 `harness.Planner` 用的是**同一个** agent 核心。两条链路的 system / observation /
自修复回灌、以及 route→IR→validate 的状态转移全部来自同一份代码，逐轮、逐 token 一致。

约束来自 ms-swift `GYMScheduler`：`env.reset` 只能注入「1 条 system + 1 条 user observation」，
`env.step` 只能追加「1 条 user」，**无法注入 few-shot 的独立 role 轮次**。因此 few-shot 在
`prompts.py` 里**以文本内嵌**进 system(路由) / observation(IR)，推理侧同样如此——这是达成
token 级训推一致的前提。生成(policy 采样)始终归 rollout 引擎，agent 只吃「模型这轮输出」、
吐「下一条 observation 或终止」，不含任何模型调用。

## 数据流与打分契约

`planner_plugin.py` 通过 ms-swift 的两个注册表接入：

| 注册表 | 键 | 作用 |
|---|---|---|
| `swift.rollout.gym_env.envs` | `planner_env` | 多轮 rollout：route→IR→修复（委托 `PlannerAgent`） |
| `swift.rewards.orms` | `planner_accuracy`（主）/ `planner_route` / `planner_ir_valid` / `planner_equiv` | 端到端/分层打分 |

env 轨迹形态（与推理逐 token 一致）：

```
system : route_system_prompt()          (ROUTE_SYSTEM + 路由 few-shot，文本内嵌)
user   : route_observation(query)        (reset)
asst   : 路由 JSON                        (turn 1)
user   : ir_observation(domain, query)   (IR_SYSTEM + IR few-shot + 字段清单，文本内嵌)
asst   : IR JSON                          (turn 2)
[user  : repair_observation(errs)         (自修复, <=max_repairs)
 asst  : 修正后的 IR]                     (turn 3..)
```

reward 函数拿到的关键 kwargs（来自 ms-swift `compute_rewards_per_func` / `to_reward_row`）：

- `completions`：每条 = 最后一个 assistant turn（最终 IR）
- `messages`：每条 = **完整多轮轨迹**（用于取出 route + 最终 IR）
- 数据集透传列 `gold_tool` / `gold_params` / `gold_domain`（batched list）

`planner_accuracy`（主奖励，与 bench 严格度量对齐）：

- 非检索类工具样本：仅按**路由是否命中**打分（0/1）。
- 检索类样本：`0.3*route + 0.7*param`，且 `param` **仅在路由命中时计入**。
  `param` 用「**召回 + 子集硬门控**」：把 IR 用 `gold_tool` 编译后取 (field,value) 及
  顶层键（sort/fee 等）指纹与 `gold_params` 指纹比——出现任何 gold 没有的指纹
  （多写字段 / 写错值）直接 `0.0`（硬惩罚过度指定），否则按召回 `|pred∩gold|/|gold|`
  给分，全覆盖即精确等价 `=1.0`。此设计消除了旧 Jaccard 部分分的「安全子集」塌缩
  吸引子（模型只输出 title 单字段），逼模型产出完整复合查询。
- 路由命中容忍「慢链路→精确检索」回退等价（routed `*_slow_search` vs gold `*_search`）。
- 训评一致：`bench.py` 的 `params_equivalent` 会把单元素 `and[X]` 拍平为 `X`，与此处
  指纹归一化对齐，避免「内容全对但结构包裹不同」被误判。

## 运行

```bash
cd train

# 1) 构建数据集（gold 来自 benchmark CSV 的 expected_tool / expected_params）
python build_dataset.py -i ../benchmark_vod.csv -o data --val-ratio 0.1

# 2) 终端 A：起 rollout server（承载 env）
bash run.sh rollout          # 默认 GPU 2,3；模型 = $MODEL

# 3) 终端 B：开训（Megatron 后端）
bash run.sh train            # 默认 GPU 4-7；连 127.0.0.1:8000 的 server
```

模型通过环境变量指定（默认占位 `Qwen/Qwen3.5-35B-A`，替换成真实 id / 本地路径）：

```bash
MODEL=/path/to/qwen3.5-35B-A ROLLOUT_GPUS=2,3 bash run.sh rollout
# Megatron 并行度可用环境变量覆盖：TP/PP/EP/CP（world_size = CP*TP*PP*DP）
MODEL=/path/to/qwen3.5-35B-A TRAIN_GPUS=2,3,4,5,6,7 TP=2 PP=1 EP=2 CP=1 bash run.sh train
```

> Megatron 首次加载 HF 权重会经 `mcore_bridge` 自动转 mcore 格式（`--save_safetensors true`
> 保证 checkpoint 存回 safetensors）。如遇转换/显存问题参考
> `../ms-swift/examples/megatron/grpo/moe_colocate_full.sh` 与 `moe_colocate_lora.sh`。

## 备注

- `--max_turns 5` = route(1) + IR(1) + 自修复(≤2) + 余量；rollout 与训练两侧需一致。
- `use_gym_env=true` 时 env 的 `total_reward`（恒 0）会作为一列常数附加进 reward 矩阵，
  组内 std=0 不产生优势，不影响训练；主信号来自 `planner_accuracy`。
- 想做**分项监控/多目标**：把 `--reward_funcs` 换成
  `planner_route planner_ir_valid planner_equiv` 并用 `--reward_weights` 调权重。
- **Megatron 5D 并行**：`TP*PP*CP*DP = world_size`，MoE 另用 `EP` 切专家；显存紧张按
  `run_grpo.sh` 末尾的阶梯：加大 EP/TP/PP → `recompute_granularity full` → 改 `--tuner_type lora`。
- 35B MoE 全参 RL 默认开 `sleep_level 2 + offload_model/optimizer + 精度感知优化器`；
  换 LoRA 时把 `--lr` 调到 `5e-5`。
- 训练时默认**不开约束解码**，让模型学会自己产出合法结构（非法结构由 reward 惩罚）；
  推理/部署再按主仓库 README 打开 `structured_outputs` 收紧。
