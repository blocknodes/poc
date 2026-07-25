#!/usr/bin/env bash
# =============================================================================
#  planner GRPO 一体化启动脚本（Megatron 后端 + vLLM rollout server）
#
#  用法（两个终端，先 rollout 再 train）：
#      bash run.sh rollout          # 终端 A：起 vLLM rollout server（承载 env 多轮交互）
#      bash run.sh train            # 终端 B：起 GRPO 训练（Megatron 后端，连 server）
#
#  子命令：
#      rollout            启动 swift rollout（vLLM）server
#      train | megatron   启动 megatron rlhf GRPO 训练
#
#  两侧共享同一份配置（模型 / plugin / env 契约 / 长度契约），必须一致：
#      * MODEL、PLUGIN            两侧同一模型、同一 plugin
#      * gym_env / scheduler      planner_env + gym_scheduler
#      * max_turns 5              route(1)+IR(1)+自修复(<=2)+余量
#      * 长度契约                 vllm_max_model_len >= max_length + max_completion_length
#                                 = 6144(prompt预算) + 2048 = 8192
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# ---- 共享配置 ---------------------------------------------------------------
# 模型：Qwen3.5-35B-A3B（MoE：256 experts / top-8，40 层，含 vision tower）。
# 替换成你的真实模型 id / 本地路径。rollout 与 train 两侧务必一致。
MODEL="${MODEL:-../../../models/Qwen3.5-35B-A3B/}"
PLUGIN="$(pwd)/planner_plugin.py"

# gym env / 多轮调度 / 长度，两侧必须对齐
GYM_ENV="planner_env"
SCHEDULER="gym_scheduler"
MAX_TURNS="${MAX_TURNS:-5}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8000}"

usage() {
    cat <<'EOF'
用法: bash run.sh <rollout|train|sft>

  rollout            启动 vLLM rollout server（承载 planner gym env 多轮交互）
  train | megatron   启动 Megatron 后端 GRPO 训练（server 模式，连 rollout server）
  sft                启动 Megatron 后端 SFT（监督微调，冷启动/基线，单进程无需 rollout）

GRPO：先在终端 A 起 rollout，再在终端 B 起 train。
SFT ：直接 bash run.sh sft（先 python build_sft_dataset.py 造数据）。
EOF
}

# =============================================================================
#  rollout —— vLLM rollout server
#  承载 planner gym env 的多轮交互 (route -> IR -> 修复)。
#  gym env 的多轮 rollout 必须走 server 模式：训练进程把生成请求发到这个 server，
#  server 侧用 GYMScheduler 驱动 PlannerEnv 的 reset/step。
# =============================================================================
run_rollout() {
    # rollout server 用哪几张卡（与训练卡错开）
    export CUDA_VISIBLE_DEVICES="${ROLLOUT_GPUS:-2,3}"

    swift rollout \
        --model "$MODEL" \
        --external_plugins "$PLUGIN" \
        --use_gym_env true \
        --gym_env "$GYM_ENV" \
        --multi_turn_scheduler "$SCHEDULER" \
        --max_turns "$MAX_TURNS" \
        --enable_thinking false \
        --host "$ROLLOUT_HOST" \
        --port "$ROLLOUT_PORT" \
        --vllm_tensor_parallel_size "${ROLLOUT_TP:-2}" \
        --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
        --vllm_gpu_memory_utilization 0.9 \
        --vllm_enable_prefix_caching true

    # 说明：
    #  --max_turns 5     : route(1) + IR(1) + 自修复(<=2) + 余量。与训练侧保持一致。
    #  --gym_env planner_env / --multi_turn_scheduler gym_scheduler
    #                    : 选中 plugin 里注册的 env，用内置 GYMScheduler 驱动。
    #  env 每步 reward=0（reward 不从 env 出），真正打分在训练侧的 reward_funcs。
}

# =============================================================================
#  train —— GRPO 训练（Megatron 后端）
#  planner 主流程 (gym env) + 自定义 reward。
#
#  === OOM 修复（参考 ../../toolPoc/swift_gym/train_megatron.sh）===============
#  Qwen3.5-35B-A3B 全参 RL 在 4~6×A100-80G 上放不下（bf16 权重 ~70GB +
#  fp32 优化器状态/master 权重/梯度远超单机显存）。故：
#    1) 改用 LoRA（target_modules all-linear）：冻结底座、只训适配器，
#       无 fp32 优化器状态/梯度开销，是能稳定落地的方案。
#    2) MoE 用 EP=4 把 256 个专家切到 4 张卡（256/4=64/卡），TP=1/PP=1/CP=1。
#    3) recompute_granularity full 重算激活；冻结 vision（freeze_vit/aligner）。
#    4) 去掉 full-FT 才需要的 offload_model / optimizer_cpu_offload /
#       precision_aware_optimizer；LoRA 下底座需常驻卡做前向。
#    5) sequence_parallel=false（TP=1 时无意义）。
#    6) 降峰值：num_generations 4、global_batch_size 8、steps_per_generation 2。
#
#  === 长度契约（必须与 rollout server 对齐）===================================
#  rollout server 起时用了 --vllm_max_model_len 8192，训练侧必须满足：
#      vllm_max_model_len >= max_length + max_completion_length
#  这里 max_length 6144 + max_completion_length 2048 = 8192。
#
#  reward：只用 planner_accuracy（稠密）。env 自身 reward 恒 0。
#          分项监控：--reward_funcs planner_route planner_ir_valid planner_equiv
# =============================================================================
run_train() {
    TRAIN_DATA="${TRAIN_DATA:-data/train.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-megatron_output}"

    # === 断点续训（resume）====================================================
    # LoRA + Megatron 的续训要点（读 swift/megatron/trainers/base.py 得到）：
    #   * --mcore_adapter <ckpt> : 从 dist-checkpoint(iter_xxxxxxx/*.distcp) 恢复
    #                              适配器权重 + 优化器状态 + RNG + 已消耗样本数。
    #   * --finetune false       : 关键！默认 finetune=true 会强制 no_load_optim、
    #                              重置 LR 调度并从 step0 开始；设 false 才会读
    #                              latest_checkpointed_iteration.txt 从原 step 续跑。
    #   （--adapters 只加载权重、不含优化器，不是真正的续训，仅用于推理/热启动。）
    #
    # 用法：
    #   RESUME=auto        bash run.sh train   # 自动找 OUTPUT_DIR 下最新的 checkpoint-*
    #   RESUME=<ckpt路径>  bash run.sh train   # 指定某个 checkpoint 目录
    # 约束：续训的 TP/PP/EP/CP 必须与生成该 ckpt 时一致，否则优化器分片对不上。
    RESUME="${RESUME:-}"
    RESUME_ARGS=()
    FINETUNE_ARGS=(--finetune)          # 默认：全新训练（从底座 + 随机初始化适配器）
    if [ -n "$RESUME" ]; then
        if [ "$RESUME" = auto ]; then
            # 在 OUTPUT_DIR/v*/checkpoint-* 里按 step 号取最大的一个
            RESUME="$(find "$OUTPUT_DIR" -maxdepth 2 -type d -name 'checkpoint-[0-9]*' 2>/dev/null \
                | grep -v -- '-merged$' \
                | awk -F'checkpoint-' '{print $2" "$0}' | sort -n | tail -1 | cut -d' ' -f2-)"
            [ -z "$RESUME" ] && { echo "[resume] 在 $OUTPUT_DIR 下没找到 checkpoint-*，无法续训" >&2; exit 1; }
        fi
        if [ ! -f "$RESUME/latest_checkpointed_iteration.txt" ]; then
            echo "[resume] 无效 checkpoint（缺 latest_checkpointed_iteration.txt）: $RESUME" >&2
            exit 1
        fi
        echo "[resume] 从 $RESUME (iter=$(cat "$RESUME/latest_checkpointed_iteration.txt")) 续训" >&2
        RESUME_ARGS=(--mcore_adapter "$RESUME")
        FINETUNE_ARGS=(--finetune false)
    fi

    # 训练卡（与 rollout server 的 2,3 错开）。EP=4 -> 用 4 张：4,5,6,7。
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-4,5,6,7}"
    export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
    # MoE + 大模型建议开，缓解显存碎片
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    # 自动选空闲 MASTER_PORT，避免与 rollout / 其他进程冲突
    if [ -z "${MASTER_PORT:-}" ]; then
        export MASTER_PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
    fi

    # Megatron 并行度：world_size = CP * TP * PP * DP = 4（本例 DP=1）
    #   TP=1, PP=1, CP=1；EP=4 用于 MoE 专家并行（256 experts / 4 = 64 /卡）。
    TP="${TP:-1}"          # tensor_model_parallel_size
    PP="${PP:-1}"          # pipeline_model_parallel_size
    EP="${EP:-4}"          # expert_model_parallel_size（MoE）
    CP="${CP:-1}"          # context_parallel_size

    # === 省显存开关（MEM_SAVE）===============================================
    # 默认关闭，训练动态与之前完全一致。开启后加激活重算（recompute）：用算力换显存，
    # 不改变 GRPO 组大小/超参/数值，是 OOM 时应最先尝试的安全档位。
    #   MEM_SAVE=1 RESUME=auto bash run.sh train
    # 若仍不够，再动会改变训练动态的参数（num_generations / global_batch_size / 加卡调 EP）。
    MEM_SAVE_ARGS=()
    if [ "${MEM_SAVE:-1}" = 1 ]; then
        echo "[mem_save] 开启激活重算：recompute_granularity=full uniform num_layers=1" >&2
        MEM_SAVE_ARGS=(
            --recompute_granularity full
            --recompute_method uniform
            --recompute_num_layers 1
        )
    fi

    megatron rlhf \
        --rlhf_type grpo \
        --model "$MODEL" \
        "${RESUME_ARGS[@]}" \
        "${MEM_SAVE_ARGS[@]}" \
        --save_safetensors true \
        --context_parallel_size "$CP" \
        --tensor_model_parallel_size "$TP" \
        --pipeline_model_parallel_size "$PP" \
        --expert_model_parallel_size "$EP" \
        --sequence_parallel false \
        --moe_permute_fusion true \
        --moe_grouped_gemm true \
        --moe_shared_expert_overlap true \
        --moe_aux_loss_coeff 1e-3 \
        --external_plugins "$PLUGIN" \
        --reward_funcs planner_accuracy \
        --dataset "$TRAIN_DATA" \
        --use_gym_env true \
        --gym_env "$GYM_ENV" \
        --multi_turn_scheduler "$SCHEDULER" \
        --max_turns "$MAX_TURNS" \
        --enable_thinking false \
        --use_vllm true \
        --vllm_mode server \
        --vllm_server_host "$ROLLOUT_HOST" \
        --vllm_server_port "$ROLLOUT_PORT" \
        --vllm_server_pass_dataset true \
        --num_train_epochs 3 \
        --global_batch_size "${GLOBAL_BATCH_SIZE:-24}" \
        --micro_batch_size "${MICRO_BATCH_SIZE:-1}" \
        --steps_per_generation "${STEPS_PER_GENERATION:-4}" \
        --num_generations 8 \
        --max_length 8192 \
        --max_completion_length 2048 \
        --overlong_filter true \
        --tuner_type lora \
        --target_modules all-linear \
        --freeze_vit true \
        --freeze_aligner true \
        "${FINETUNE_ARGS[@]}" \
        --bf16 true \
        --lr 5e-5 \
        --lr_warmup_fraction 0.02 \
        --min_lr 1e-6 \
        --beta 0.001 \
        --importance_sampling_level token \
        --epsilon 0.2 \
        --epsilon_high 0.2 \
        --dynamic_sample false \
        --loss_type grpo \
        --temperature 1.0 \
        --top_p 0.9 \
        --top_k 80 \
        --padding_free true \
        --attention_backend flash \
        --dataloader_num_workers 4 \
        --dataset_num_proc 4 \
        --save_steps 20 \
        --eval_steps 20 \
        --logging_steps 1 \
        --log_completions true \
        --output_dir "$OUTPUT_DIR" \
        --report_to tensorboard

    # 关键参数：
    #  --tuner_type lora --target_modules all-linear : 只训适配器，规避 35B 全参优化器爆显存
    #  --expert_model_parallel_size 4                : MoE 专家并行，256 experts 切到 4 卡
    #  --freeze_vit/--freeze_aligner true            : 冻结视觉塔/对齐层（本模型带 vision）
    #  --max_length 6144 + --max_completion_length 2048 = 8192 : 与 rollout vllm_max_model_len 对齐
    #  --vllm_mode server + host/port                : 连 rollout 子命令起的 vLLM server
    #
    # 若仍 OOM，从小到大加码：
    #   1) 降 --num_generations 4->2、--global_batch_size 8->4
    #   2) 降 --max_completion_length 2048->1024（同时可降 max_length，保证和 <= 8192）
    #   3) 加卡并调 EP：用 8 卡时 EP=8（256/8=32/卡），NPROC/CUDA_VISIBLE_DEVICES 同步改
    #   4) --lora_rank 8（更小适配器）
}

# =============================================================================
#  sft —— Megatron 后端 SFT（监督微调）
#  planner 的 route + IR 多轮 gold 对话（build_sft_dataset.py 造）。
#
#  与 GRPO(train) 的关系：SFT 常用作 **RL 冷启动**——先 SFT 把 route/IR 的格式与
#  基本能力学出来，再 GRPO 精修参数召回。也可单独作为基线对照。
#
#  与 train 的差异：SFT 是**离线单进程**训练，没有 rollout server / reward / vLLM。
#  其余（模型 / LoRA / 5D 并行 / 冻结 vision / 省显存开关）与 train 完全一致，
#  确保 SFT 产出的 adapter 能被同一套 deploy.sh export 出来、也能被 GRPO 续训。
#
#  数据：先造 SFT jsonl（route 全监督 + 可验证的 IR 轮）：
#      python build_sft_dataset.py -i ../benchmark_vod.csv -o data_sft --val-ratio 0.1
#
#  用法：
#      bash run.sh sft
#      SFT_GPUS=0,1,2,3 EP=4 LR=1e-4 EPOCHS=3 bash run.sh sft
#      RESUME=<ckpt> bash run.sh sft          # 从已有 adapter 续训（同 train 的语义）
# =============================================================================
run_sft() {
    SFT_DATA="${SFT_DATA:-data_sft/train.jsonl}"
    SFT_VAL="${SFT_VAL:-data_sft/val.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-megatron_output_sft}"

    if [ ! -f "$SFT_DATA" ]; then
        echo "[sft] 找不到 $SFT_DATA，请先执行：" >&2
        echo "      python build_sft_dataset.py -i ../benchmark_vod.csv -o data_sft --val-ratio 0.1" >&2
        exit 1
    fi

    # 断点续训（与 train 同语义；--finetune false 才会读回 step/优化器/consumed_samples，
    # 且本仓库 planner_plugin 的 monkeypatch 会修正 --mcore_adapter 的 consumed_train_samples）
    RESUME="${RESUME:-}"
    RESUME_ARGS=()
    FINETUNE_ARGS=(--finetune)
    if [ -n "$RESUME" ]; then
        if [ "$RESUME" = auto ]; then
            RESUME="$(find "$OUTPUT_DIR" -maxdepth 2 -type d -name 'checkpoint-[0-9]*' 2>/dev/null \
                | grep -v -- '-merged$' \
                | awk -F'checkpoint-' '{print $2" "$0}' | sort -n | tail -1 | cut -d' ' -f2-)"
            [ -z "$RESUME" ] && { echo "[resume] $OUTPUT_DIR 下没有 checkpoint-*" >&2; exit 1; }
        fi
        [ -f "$RESUME/latest_checkpointed_iteration.txt" ] || {
            echo "[resume] 无效 checkpoint: $RESUME" >&2; exit 1; }
        echo "[resume] 从 $RESUME (iter=$(cat "$RESUME/latest_checkpointed_iteration.txt")) 续训 SFT" >&2
        RESUME_ARGS=(--mcore_adapter "$RESUME")
        FINETUNE_ARGS=(--finetune false)
    fi

    export CUDA_VISIBLE_DEVICES="${SFT_GPUS:-0,1,2,3}"
    export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    if [ -z "${MASTER_PORT:-}" ]; then
        export MASTER_PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
    fi

    # 并行度（与 train 一致）：world_size = CP*TP*PP*DP；MoE 用 EP 切专家
    TP="${TP:-1}"; PP="${PP:-1}"; EP="${EP:-4}"; CP="${CP:-1}"

    # 省显存（激活重算），默认开，与 train 一致
    MEM_SAVE_ARGS=()
    if [ "${MEM_SAVE:-1}" = 1 ]; then
        echo "[mem_save] 开启激活重算：recompute_granularity=full uniform num_layers=1" >&2
        MEM_SAVE_ARGS=(--recompute_granularity full --recompute_method uniform --recompute_num_layers 1)
    fi

    # 可选验证集
    VAL_ARGS=()
    [ -f "$SFT_VAL" ] && VAL_ARGS=(--val_dataset "$SFT_VAL")

    megatron sft \
        --model "$MODEL" \
        "${RESUME_ARGS[@]}" \
        "${MEM_SAVE_ARGS[@]}" \
        --save_safetensors true \
        --context_parallel_size "$CP" \
        --tensor_model_parallel_size "$TP" \
        --pipeline_model_parallel_size "$PP" \
        --expert_model_parallel_size "$EP" \
        --sequence_parallel false \
        --moe_permute_fusion true \
        --moe_grouped_gemm true \
        --moe_shared_expert_overlap true \
        --moe_aux_loss_coeff 1e-3 \
        --dataset "$SFT_DATA" \
        "${VAL_ARGS[@]}" \
        --enable_thinking false \
        --num_train_epochs "${EPOCHS:-3}" \
        --global_batch_size "${GLOBAL_BATCH_SIZE:-24}" \
        --micro_batch_size "${MICRO_BATCH_SIZE:-1}" \
        --max_length 8192 \
        --tuner_type lora \
        --target_modules all-linear \
        --freeze_vit true \
        --freeze_aligner true \
        "${FINETUNE_ARGS[@]}" \
        --bf16 true \
        --lr "${LR:-1e-4}" \
        --lr_warmup_fraction 0.03 \
        --min_lr 1e-6 \
        --padding_free true \
        --attention_backend flash \
        --dataloader_num_workers 4 \
        --dataset_num_proc 4 \
        --save_steps "${SAVE_STEPS:-20}" \
        --eval_steps "${EVAL_STEPS:-20}" \
        --logging_steps 1 \
        --output_dir "$OUTPUT_DIR" \
        --report_to tensorboard

    # 关键点：
    #  * 只训 LoRA（target_modules all-linear）+ EP=4 切 MoE 专家，显存与 train 同档。
    #  * SFT 只在 assistant 轮（route / IR gold）上算 loss；system/user 的长 prompt
    #    （含 few-shot）不计损失，只作条件——与推理 prompt 逐 token 一致。
    #  * LoRA SFT 学习率取 1e-4（比 GRPO 的 5e-5 略大，快速收敛格式与基本能力）。
    #  * 产物在 megatron_output_sft/；用 deploy.sh export <ckpt> 合并成 HF 后部署，
    #    或作为 GRPO 冷启动：run.sh train 时 RESUME 指向该 SFT checkpoint。
    #  若 OOM：MEM_SAVE=1（默认已开）→ 降 GLOBAL_BATCH_SIZE → 加卡调 EP → --lora_rank 8。
}

# ---- 入口分发 ---------------------------------------------------------------
MODE="${1:-}"
case "$MODE" in
    rollout)
        run_rollout
        ;;
    train | megatron | grpo)
        run_train
        ;;
    sft)
        run_sft
        ;;
    ""|-h|--help|help)
        usage
        [ -z "$MODE" ] && exit 1 || exit 0
        ;;
    *)
        echo "未知子命令: $MODE" >&2
        usage
        exit 1
        ;;
esac
