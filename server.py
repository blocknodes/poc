#!/usr/bin/env python3
"""API Server —— 慢任务接口_v2。

按新版接口规范，将 Planner 主流程封装为 HTTP POST 服务。
返回带 DAG 依赖关系、policy、retext 的多步骤执行计划。

接口路径：POST /slowAgent/poc_1784284048243
启动方式：
    # 开发模式（自动 reload）
    python server.py --port 8082 --reload

    # 连接真实 vLLM
    VLLM_BASE_URL=http://localhost:8080/v1 VLLM_MODEL=baseline python server.py --host 0.0.0.0 --port 8082

    # 离线 stub 模式（用于调试，不连模型）
    python server.py --stub

    # 生产部署（gunicorn/uvicorn）
    uvicorn server:app --host 0.0.0.0 --port 8082 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# 确保 planner 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("ERROR: 缺少 fastapi 和 uvicorn，请安装：pip install fastapi uvicorn")
    sys.exit(1)

from planner import Planner, VLLMClient, VLLMConfig
from planner.ir import IRError
from planner.compiler import CompileError

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("slow_agent_server")

# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(
    title="慢任务接口_v2",
    description="电视端 AI 慢任务 Planner API —— 用户自然语言 → 结构化工具调用执行计划",
    version="2.0.0",
)

# 全局 Planner 实例（启动时初始化）
_planner: Optional[Planner] = None


def get_planner() -> Planner:
    global _planner
    if _planner is None:
        raise RuntimeError("Planner 未初始化")
    return _planner


# --------------------------------------------------------------------------
# 工具分类（用于依赖推断）
# --------------------------------------------------------------------------
# 检索类工具
SEARCH_TOOLS = {
    "vod_search", "vod_search_all", "educ_search", "vod_relate_search",
    "vod_slow_search_data_search", "audio_search", "audio_chat_qa",
    "vod_personalized_search", "vod_history",
}

# 设备控制类工具
DEVICE_CONTROL_TOOLS = {
    "volume_control", "power_control", "signal_source_control",
    "screen_display_control", "camera_control", "network_control",
    "bluetooth_control", "input_lang_control", "image_quality_control",
    "sound_mode_control", "projection_control", "media_center_control",
    "demo_control", "personalization_control", "scene_mode_control",
    "screen_safety_control", "playback_control", "ambient_light_control",
    "system_settings_control", "ai_picture_sound_control",
}

# 播放类工具
PLAY_TOOLS = {"vod_play", "audio_play"}


# --------------------------------------------------------------------------
# Step 构建
# --------------------------------------------------------------------------
@dataclass
class PlanStep:
    id: str
    toolName: str
    parameters: dict[str, Any]
    dependsOn: list[str]
    retext: str
    dependsReason: Optional[str] = None


def _generate_plan_id() -> str:
    """生成短 planId。"""
    return "plan_" + uuid.uuid4().hex[:4]


def _build_retext(tool_name: str, parameters: dict, query: str) -> str:
    """根据工具名和参数生成 retext（人类可读的步骤描述）。"""
    # 设备控制类
    if tool_name in DEVICE_CONTROL_TOOLS:
        op = parameters.get("operation", "")
        obj = parameters.get("object", "")
        val = parameters.get("value", "")
        if val:
            return f"{op}{obj}到{val}" if "设置" in op or "调" in op else f"{op}{obj} {val}"
        return f"{op}{obj}" if op and obj else query

    # 检索类
    if tool_name in SEARCH_TOOLS:
        return query

    # 播放类
    if tool_name in PLAY_TOOLS:
        return f"播放: {query}"

    return query


# --------------------------------------------------------------------------
# 响应构建
# --------------------------------------------------------------------------
def build_success_response(
    trace_id: str,
    device_id: str,
    steps: list[PlanStep],
    plan_confidence: float,
) -> dict:
    """构建成功响应，包含完整的执行计划。"""
    steps_data = []
    for s in steps:
        step_dict = {
            "id": s.id,
            "toolName": s.toolName,
            "parameters": s.parameters,
            "dependsOn": s.dependsOn,
            "retext": s.retext,
        }
        if s.dependsReason:
            step_dict["dependsReason"] = s.dependsReason
        steps_data.append(step_dict)

    return {
        "traceId": trace_id,
        "deviceId": device_id,
        "code": 200,
        "message": "success",
        "data": {
            "planId": _generate_plan_id(),
            "schemaVersion": "1.0",
            "planType": "execute",
            "steps": steps_data,
            "planConfidence": plan_confidence,
        },
    }


def build_error_response(
    trace_id: str,
    device_id: str,
    code: int,
    message: str,
) -> dict:
    return {
        "traceId": trace_id,
        "deviceId": device_id,
        "code": code,
        "message": message,
        "data": {
            "planId": _generate_plan_id(),
            "schemaVersion": "1.0",
            "planType": "execute",
            "steps": [],
            "planConfidence": 0.0,
        },
    }


# --------------------------------------------------------------------------
# 请求校验
# --------------------------------------------------------------------------
REQUIRED_TOP_FIELDS = ["traceId", "deviceId", "deviceType", "data"]
REQUIRED_DATA_FIELDS = ["query", "tvMode", "toolList", "timestamp"]


def validate_request(body: dict) -> Optional[str]:
    """校验请求参数，返回错误信息或 None。"""
    for field in REQUIRED_TOP_FIELDS:
        if field not in body or body[field] is None:
            return f"缺少必须字段: {field}"

    data = body.get("data")
    if not isinstance(data, dict):
        return "data 字段必须为 object"

    for field in REQUIRED_DATA_FIELDS:
        if field not in data or data[field] is None:
            return f"缺少必须字段: data.{field}"

    if not isinstance(data.get("query"), str) or not data["query"].strip():
        return "data.query 不能为空"

    if not isinstance(data.get("toolList"), list):
        return "data.toolList 必须为数组"

    return None


# --------------------------------------------------------------------------
# 从请求中提取 memory hint
# --------------------------------------------------------------------------
def extract_memory_hint(data: dict) -> str:
    """从请求的 memory 字段提取对话历史摘要，作为 planner 的 memory_hint。"""
    memory = data.get("memory")
    if not memory:
        return ""

    parts = []

    # 短期记忆：最近 N 轮对话
    short_memory = memory.get("shortMemory", [])
    if short_memory:
        for turn in short_memory[-3:]:  # 最多取最近 3 轮
            if isinstance(turn, dict):
                user_msg = turn.get("query", turn.get("user", ""))
                bot_msg = turn.get("answer", turn.get("bot", ""))
                if user_msg:
                    parts.append(f"用户: {user_msg}")
                if bot_msg:
                    parts.append(f"助手: {bot_msg}")

    # 长期记忆：用户画像
    long_memory = memory.get("longMemory")
    if long_memory and isinstance(long_memory, dict):
        profile_parts = []
        for k, v in long_memory.items():
            if v:
                profile_parts.append(f"{k}:{v}")
        if profile_parts:
            parts.append(f"用户画像: {'; '.join(profile_parts)}")

    return "\n".join(parts)


# --------------------------------------------------------------------------
# 依赖关系推断
# --------------------------------------------------------------------------
def _infer_dependencies(plan_results: list, original_query: str = "") -> list[PlanStep]:
    """根据 PlanResult 列表推断步骤间依赖关系，构建 DAG。

    规则：
    1. 同域的检索步骤之间无依赖（可并发）。
    2. 设备控制步骤之间：同类工具有顺序依赖（串行），不同工具可并发。
    3. 播放步骤依赖所有检索步骤和设备控制步骤（等资源就绪再播放）。
    4. 同一工具的多次调用按顺序串行。
    """
    steps: list[PlanStep] = []
    step_id_counter = 0

    # 分类收集 step ids
    search_step_ids: list[str] = []
    device_step_ids_by_tool: dict[str, list[str]] = {}
    all_step_ids: list[str] = []

    for result in plan_results:
        if result.tool_name == "error":
            continue

        step_id_counter += 1
        sid = f"s{step_id_counter}"

        tool_name = result.tool_name
        params = dict(result.parameters) if isinstance(result.parameters, dict) else result.parameters
        action = params.get("action") if isinstance(params, dict) else None

        # 从 parameters 中提取 retext（编译器产出），提取后从 params 中移除避免重复
        params_retext = ""
        if isinstance(params, dict) and "retext" in params:
            params_retext = params.pop("retext", "")

        # 推断 retext 优先级：
        # 1. 编译器产出的 params 中的 retext
        # 2. notes 中的子请求原文
        # 3. 基于工具名 + 参数自动生成
        retext = params_retext

        if not retext:
            for note in (result.notes or []):
                if "子请求" in note and "=" in note:
                    try:
                        retext = note.split("='", 1)[1].rstrip("'")
                    except (IndexError, ValueError):
                        pass
                    if retext:
                        break

        if not retext:
            # 用原始 query（从 agent 获取）而非 params 内的 query 结构
            retext = _build_retext(tool_name, params, original_query)

        # 推断依赖关系
        depends_on: list[str] = []
        depends_reason: Optional[str] = None

        if tool_name in PLAY_TOOLS or action == "play":
            # 播放步骤依赖所有前置的检索和设备步骤
            depends_on = list(all_step_ids)
        elif tool_name in DEVICE_CONTROL_TOOLS:
            # 同类设备工具串行
            if tool_name in device_step_ids_by_tool:
                depends_on = [device_step_ids_by_tool[tool_name][-1]]
                depends_reason = "order_only"
            # 记录
            device_step_ids_by_tool.setdefault(tool_name, []).append(sid)
        elif tool_name in SEARCH_TOOLS:
            # 检索步骤之间无依赖
            search_step_ids.append(sid)

        steps.append(PlanStep(
            id=sid,
            toolName=tool_name,
            parameters=params,
            dependsOn=depends_on,
            retext=retext,
            dependsReason=depends_reason,
        ))

        all_step_ids.append(sid)

    return steps


# --------------------------------------------------------------------------
# 核心接口
# --------------------------------------------------------------------------
@app.post("/slowAgent/poc_1784284048243")
async def slow_agent(request: Request):
    """慢任务接口_v2 —— 主入口。

    完整链路：
      1. 参数校验
      2. 提取 query + memory
      3. 调用 Planner.plan_multi() 走完 route → IR → compile 全流程
      4. 推断步骤间依赖关系，构建执行 DAG
      5. 封装为标准响应返回
    """
    start_time = time.time()

    # 解析请求 body
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=build_error_response("", "", 400, "请求体 JSON 解析失败"),
        )

    trace_id = body.get("traceId", "")
    device_id = body.get("deviceId", "")

    # 参数校验
    err = validate_request(body)
    if err:
        logger.warning(f"[{trace_id}] 参数校验失败: {err}")
        return JSONResponse(
            status_code=400,
            content=build_error_response(trace_id, device_id, 400, err),
        )

    data = body["data"]
    query = data["query"].strip()
    memory_hint = extract_memory_hint(data)

    logger.info(f"[{trace_id}] 收到请求: query='{query}', device={device_id}, tvMode={data.get('tvMode')}")

    # 调用 Planner 主流程
    try:
        planner = get_planner()

        # 多意图并发规划
        results = planner.plan_multi(query, memory_hint=memory_hint)

        # 过滤失败结果
        success_results = [r for r in results if r.tool_name != "error"]
        failed_results = [r for r in results if r.tool_name == "error"]

        for fr in failed_results:
            logger.warning(
                f"[{trace_id}] 子请求失败: {fr.parameters.get('error', 'unknown')}"
            )

        # 如果全部失败，返回错误
        if not success_results:
            elapsed = time.time() - start_time
            logger.error(f"[{trace_id}] 所有子请求均失败 (elapsed={elapsed:.2f}s)")
            return JSONResponse(
                status_code=200,
                content=build_error_response(trace_id, device_id, 500, "规划失败：所有子请求均执行失败"),
            )

        # 推断依赖关系，构建步骤 DAG
        steps = _infer_dependencies(success_results, original_query=query)

        # 计算 planConfidence：取各子结果 route_confidence 的加权平均
        confidences = [
            r.route_confidence for r in success_results
            if r.route_confidence is not None
        ]
        plan_confidence = round(
            sum(confidences) / len(confidences), 2
        ) if confidences else 0.85

        elapsed = time.time() - start_time
        tool_names = [s.toolName for s in steps]
        logger.info(
            f"[{trace_id}] 规划完成: tools={tool_names}, "
            f"steps={len(steps)}, multi={len(results) > 1}, "
            f"confidence={plan_confidence}, elapsed={elapsed:.2f}s"
        )

        return JSONResponse(
            status_code=200,
            content=build_success_response(trace_id, device_id, steps, plan_confidence),
        )

    except IRError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{trace_id}] IR 错误: {e} (elapsed={elapsed:.2f}s)")
        return JSONResponse(
            status_code=200,
            content=build_error_response(trace_id, device_id, 500, f"IR 校验失败: {e}"),
        )

    except CompileError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{trace_id}] 编译错误: {e} (elapsed={elapsed:.2f}s)")
        return JSONResponse(
            status_code=200,
            content=build_error_response(trace_id, device_id, 500, f"编译失败: {e}"),
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{trace_id}] 内部异常: {e} (elapsed={elapsed:.2f}s)\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=200,
            content=build_error_response(trace_id, device_id, 500, f"服务内部异常: {type(e).__name__}"),
        )


# --------------------------------------------------------------------------
# 健康检查
# --------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "planner_ready": _planner is not None}


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------
def init_planner(stub: bool = False, vod_only: bool = False):
    """初始化全局 Planner 实例。"""
    global _planner

    if stub:
        # 离线 stub 模式（复用 demo.py 的 stub responder）
        from demo import make_stub
        client = VLLMClient(responder=make_stub())
        logger.info("[STUB] Planner 使用离线 stub 模式")
    else:
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        model = os.environ.get("VLLM_MODEL", "qwen-30b-moe")
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        timeout = float(os.environ.get("VLLM_TIMEOUT", "60"))

        config = VLLMConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
        client = VLLMClient(config)
        logger.info(f"[LIVE] Planner 连接 vLLM: {base_url} / {model}")

    _planner = Planner(client, max_repairs=2, vod_only=vod_only)
    logger.info("Planner 初始化完成")


@app.on_event("startup")
async def on_startup():
    """FastAPI 启动事件：初始化 Planner（如果尚未初始化）。"""
    if _planner is None:
        stub = os.environ.get("PLANNER_STUB", "").lower() in ("1", "true", "yes")
        vod_only = os.environ.get("PLANNER_VOD_ONLY", "").lower() in ("1", "true", "yes")
        init_planner(stub=stub, vod_only=vod_only)


# --------------------------------------------------------------------------
# CLI 入口
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="慢任务接口_v2 API Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8082, help="监听端口 (默认 8082)")
    parser.add_argument("--stub", action="store_true", help="使用离线 stub（不连 vLLM）")
    parser.add_argument("--vod-only", action="store_true", help="仅启用 vod 域")
    parser.add_argument("--reload", action="store_true", help="开发模式，自动热重载")
    parser.add_argument("--workers", type=int, default=1, help="worker 进程数 (默认 1)")
    args = parser.parse_args()

    # 通过环境变量传递给 startup event
    if args.stub:
        os.environ["PLANNER_STUB"] = "1"
    if args.vod_only:
        os.environ["PLANNER_VOD_ONLY"] = "1"

    logger.info(f"启动慢任务接口_v2 服务: {args.host}:{args.port}")
    logger.info(f"  stub={args.stub}, vod_only={args.vod_only}, workers={args.workers}")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
