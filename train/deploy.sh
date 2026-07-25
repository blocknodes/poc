#!/usr/bin/env bash
# =============================================================================
#  planner 部署脚本 —— 用 GRPO 训好的 checkpoint 起 OpenAI 兼容推理服务
#
#  训练侧 (run.sh train) 是 Megatron 后端 + LoRA，落盘的是 **mcore 格式的 LoRA
#  adapter**（torch_dist），vLLM 不能直接吃。所以：
#
#      1) export : 本脚本负责——mcore LoRA adapter --合并--> HF safetensors 全量权重
#      2) deploy : 直接用原生 vLLM 起服务（见文末命令，不走 swift deploy）
#
#  为什么 deploy 不用 swift deploy：
#      planner 客户端(vllm_client.py)靠 extra_body.structured_outputs={"json":schema}
#      （旧格式 guided_json）做**约束解码**。原生 vLLM 的 OpenAI server 原生支持它；
#      而 swift deploy 的 server 只暴露 structured_outputs_regex，不吃 JSON schema，
#      planner 的结构化约束会失效。故合并出 HF 后，用原生 vllm serve 部署。
#
#  用法：
#      bash deploy.sh export               # 合并 LoRA -> HF（自动找最新 checkpoint）
#      bash deploy.sh export <checkpoint>  # 指定某个 checkpoint 目录
#      bash deploy.sh serve                # 打印/执行原生 vllm serve 命令（部署导出的 HF）
#      bash deploy.sh serve <hf_dir>       # 部署指定 HF 目录
#
#  长度 / 结构化输出契约（与训练、与 planner 客户端一致）：
#      * max-model-len 8192 = max_length 6144 + max_completion_length 2048
#      * structured-outputs backend = xgrammar（planner VLLMConfig.guided_backend 默认）
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

# ---- 共享配置（与 run.sh 保持一致）-----------------------------------------
# 底座模型：与训练时同一个（合并 LoRA 时需要它做基准）。
MODEL="${MODEL:-../../../models/Qwen3.5-35B-A3B/}"

# 训练产物根目录（run.sh train 里 --output_dir megatron_output）。
OUTPUT_DIR="${OUTPUT_DIR:-megatron_output}"

# 合并导出后的 HF 目录（serve 默认吃它）。
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged-hf}"

# 对外暴露的模型名：填进 planner 的 VLLMConfig.model / VLLM_MODEL。
SERVED_NAME="${SERVED_NAME:-planner-35b-a3b}"

# 长度契约（与 run.sh / planner 客户端对齐）。
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

# 结构化解码后端；需与 planner VLLMConfig.guided_backend 一致。
SO_BACKEND="${SO_BACKEND:-xgrammar}"

SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
SERVE_PORT="${SERVE_PORT:-8000}"

usage() {
    cat <<'EOF'
用法: bash deploy.sh <export|serve> [路径]

  export [checkpoint]  把 Megatron LoRA adapter 合并进底座并转成 HF safetensors
                       不带参数时自动挑 OUTPUT_DIR 下最新的 checkpoint-*
  serve  [hf_dir]      用原生 vllm serve 起 OpenAI 兼容服务（默认部署 MERGED_DIR）

环境变量（均可覆盖）：
  MODEL              底座模型 id/路径（须与训练一致）
  OUTPUT_DIR         训练产物根目录（默认 megatron_output）
  MERGED_DIR         合并导出目录（默认 OUTPUT_DIR/merged-hf）
  SERVED_NAME        对外模型名（填进 planner VLLMConfig.model，默认 planner-35b-a3b）
  VLLM_MAX_MODEL_LEN 最大上下文（默认 8192，与训练长度契约一致）
  SO_BACKEND         结构化解码后端（默认 xgrammar，须与 planner 一致）
  EXPORT_GPUS        export 用的卡（默认 0,1,2,3）
  EXPORT_TP/EP/PP    export 加载并行度（默认 TP=1 EP=4 PP=1，与训练一致）
  SERVE_GPUS         serve 用的卡（默认 0,1,2,3）
  SERVE_TP           serve 张量并行度（默认按 SERVE_GPUS 数量）
  SERVE_HOST/PORT    服务监听地址（默认 0.0.0.0:8000）
EOF
}

# 自动挑最新的 checkpoint-*（按修改时间）。megatron 产物结构：
#   OUTPUT_DIR/vX-<timestamp>/checkpoint-<step>/
find_latest_checkpoint() {
    local latest
    latest=$(find "$OUTPUT_DIR" -maxdepth 3 -type d -name 'checkpoint-*' \
                -print0 2>/dev/null | xargs -0 ls -dt 2>/dev/null | head -n1 || true)
    if [ -z "$latest" ]; then
        echo "错误：在 $OUTPUT_DIR 下找不到 checkpoint-* 目录，请显式传入 checkpoint 路径。" >&2
        exit 1
    fi
    echo "$latest"
}

# =============================================================================
#  export —— mcore LoRA adapter --merge--> HF safetensors
#
#  依据 examples/megatron/export/lora.sh：
#    --mcore_adapter  指向训练落盘的 torch_dist LoRA 目录（checkpoint-*）
#    --merge_lora true --to_hf true  合并进底座并输出 HF safetensors
#  合并到 HF 后，原生 vLLM 当成普通全量模型加载，不需要 LoRA 加载路径，
#  也规避 MoE all-linear LoRA 在 vLLM 上支持不全的问题。
# =============================================================================
run_export() {
    local ckpt="${1:-}"
    if [ -z "$ckpt" ]; then
        ckpt="$(find_latest_checkpoint)"
        echo ">> 自动选中 checkpoint: $ckpt"
    fi
    if [ ! -d "$ckpt" ]; then
        echo "错误：checkpoint 目录不存在: $ckpt" >&2
        exit 1
    fi

    export CUDA_VISIBLE_DEVICES="${EXPORT_GPUS:-0,1,2,3}"
    export NPROC_PER_NODE="${EXPORT_NPROC:-4}"

    echo ">> 合并 LoRA 并导出 HF 到: $MERGED_DIR"
    megatron export \
        --model "$MODEL" \
        --mcore_adapter "$ckpt" \
        --output_dir "$MERGED_DIR" \
        --merge_lora true \
        --to_hf true \
        --tensor_model_parallel_size "${EXPORT_TP:-1}" \
        --expert_model_parallel_size "${EXPORT_EP:-4}" \
        --pipeline_model_parallel_size "${EXPORT_PP:-1}" \
        --test_convert_precision true

    echo ">> 导出完成：$MERGED_DIR"
    echo "   下一步：bash deploy.sh serve"
}

# =============================================================================
#  serve —— 原生 vllm serve 起 OpenAI 兼容服务
#
#  与主仓库 README 的 vllm 启动一致，只是模型换成合并导出的 HF 目录。
#  planner 把 IR/route schema 通过 structured_outputs / guided_json 传入，
#  由原生 vLLM 的 structured outputs 做约束解码。
# =============================================================================
run_serve() {
    local hf_dir="${1:-$MERGED_DIR}"
    if [ ! -d "$hf_dir" ]; then
        echo "错误：要部署的 HF 目录不存在: $hf_dir" >&2
        echo "      请先执行: bash deploy.sh export" >&2
        exit 1
    fi

    export CUDA_VISIBLE_DEVICES="${SERVE_GPUS:-0,1,2,3}"
    local tp="${SERVE_TP:-}"
    if [ -z "$tp" ]; then
        tp=$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
    fi

    echo ">> 原生 vLLM 部署: $hf_dir  (served-model-name=$SERVED_NAME, TP=$tp)"
    vllm serve "$hf_dir" \
        --served-model-name "$SERVED_NAME" \
        --host "$SERVE_HOST" --port "$SERVE_PORT" \
        --structured-outputs-config.backend "$SO_BACKEND" \
        --tensor-parallel-size "$tp" \
        --enable-expert-parallel \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --gpu-memory-utilization "${SERVE_GPU_UTIL:-0.90}" \
        --enable-prefix-caching

    # 说明（与主仓库 README「vLLM 启动 / OOM 排查」一致）：
    #  --structured-outputs-config.backend xgrammar
    #        与 planner VLLMConfig.guided_backend 一致；老版本 vLLM(≤0.9) 改用
    #        --guided-decoding-backend xgrammar；不确定就删掉走默认 auto。
    #  --served-model-name  须与 planner VLLMConfig.model / VLLM_MODEL 一致。
    #  --enable-expert-parallel  MoE 专家并行；单卡可去掉它和 --tensor-parallel-size。
    #  接口即 http://<host>:8000/v1，填入 planner 的 VLLM_BASE_URL。
}

# ---- 入口分发 ---------------------------------------------------------------
MODE="${1:-}"
ARG="${2:-}"
case "$MODE" in
    export)
        run_export "$ARG"
        ;;
    serve|deploy)
        run_serve "$ARG"
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
