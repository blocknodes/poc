#!/usr/bin/env python3
"""enhanced_vllm_server.py —— 增强版 vLLM 中间层。

接口与 vLLM /v1/chat/completions 100% 一致，无任何扩展字段。
上层用 VLLMClient 调本服务和调裸 vLLM 完全一样——只需改 base_url。

中间层做的事情（对调用方完全透明）：
  1. 收到 messages
  2. 根据 messages 内容自动识别当前阶段（route / IR / audio / device / intent_split）
  3. 自动生成对应的 guided_json schema
  4. 带着 schema 调后端 vLLM 做约束解码
  5. 对模型输出做后处理：compile / EB compiler 层 / validate
  6. IR 校验失败时自动重试（自修复）
  7. 返回标准 chat/completions 响应（content 是处理后的 JSON）

上层代码零改动：
    # 只改 base_url 指向中间层
    client = VLLMClient(VLLMConfig(base_url="http://localhost:9000/v1"))
    result = client.complete_json(messages)  # 和以前一模一样

启动：
    python enhanced_vllm_server.py --port 9000 --backend http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    import httpx
except ImportError:
    print("ERROR: pip install fastapi uvicorn httpx")
    sys.exit(1)

from vllm_client import VLLMClient, VLLMConfig
from grammar import (
    build_route_schema,
    build_ir_schema,
    build_audio_schema,
    build_device_schema,
    build_intent_split_schema,
)
from compiler import (
    compile_with_fallback,
    compile_audio,
    compile_device,
    CompileError,
)
from ir import parse_ir, validate_ir, IRError
from prompts import repair_observation

# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("enhanced_vllm")

app = FastAPI(title="Enhanced vLLM", version="1.0.0")


@dataclass
class ServerConfig:
    backend_base_url: str = "http://localhost:8000/v1"
    backend_api_key: str = "EMPTY"
    backend_timeout: float = 60.0
    backend_request_format: str = "structured_outputs"
    host: str = "0.0.0.0"
    port: int = 9000
    max_repairs: int = 2


_cfg: Optional[ServerConfig] = None
_backend: Optional[VLLMClient] = None
_http_client: Optional[httpx.AsyncClient] = None


# ==========================================================================
# 阶段识别：从 messages 内容自动推断
#
# 不依赖任何扩展字段，纯粹看 prompt 内容特征来判断阶段。
# 这样上层发给裸 vLLM 的请求和发给中间层的请求完全一样。
# ==========================================================================

# prompt 中的关键特征词（用于识别阶段）
_ROUTE_MARKERS = ["domain", "intent", "tool", "confidence"]
_IR_MARKERS = ["IR", "布尔表达式", "query节点", "and/or/not"]
_AUDIO_MARKERS = ["play_mode", "screen_mode", "audio_search"]
_DEVICE_MARKERS = ["operation", "object", "value", "solve_picture_sound"]
_INTENT_SPLIT_MARKERS = ["multi", "sub_queries", "意图拆分"]


def _detect_stage(messages: list[dict], guided_json: Optional[dict]) -> str:
    """从 messages + guided_json 推断当前阶段。"""

    # 如果请求里已经带了 guided_json，可以从 schema 结构判断
    if guided_json:
        props = guided_json.get("properties", {})
        # route schema: 有 domain + tool + intent + confidence
        if "domain" in props and "confidence" in props and "tool" in props:
            return "route"
        # intent_split schema: 有 multi + sub_queries
        if "multi" in props and "sub_queries" in props:
            return "intent_split"
        # IR schema: 有 action + query (嵌套结构)
        if "action" in props and "query" in props and "sort" in props:
            return "ir"
        # audio schema: 有 play_mode
        if "play_mode" in props:
            return "audio"
        # device schema: 有 tool(enum 长列表) + operation + object
        if "tool" in props and "operation" in props and "object" in props:
            return "device"

    # fallback: 从最后一条 user message 的内容特征判断
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
            break

    # 简单关键词匹配
    if any(m in last_user for m in _INTENT_SPLIT_MARKERS):
        return "intent_split"

    # 从 system prompt 判断
    sys_content = ""
    for msg in messages:
        if msg.get("role") == "system":
            sys_content = msg.get("content", "")
            break

    all_content = sys_content + " " + last_user

    if "IR" in all_content and ("布尔" in all_content or "query节点" in all_content):
        return "ir"
    if "play_mode" in all_content or "audio_search" in all_content:
        return "audio"
    if "solve_picture_sound" in all_content or ("operation" in all_content and "object" in all_content and "设备" in all_content):
        return "device"

    # 默认当路由
    return "route"


# ==========================================================================
# EB prompt 层注入（对上层透明）
# ==========================================================================

def _inject_eb_prompt(stage: str, messages: list[dict]) -> list[dict]:
    """在调后端之前，对特定阶段注入 EB prompt 层规则到 messages 中。

    只影响发给后端 vLLM 的 messages，上层完全不感知。
    """
    if stage == "ir":
        return _inject_ir_eb_prompt(messages)
    if stage == "device":
        return _inject_device_eb_prompt(messages)
    # 其他阶段不注入
    return messages


def _inject_ir_eb_prompt(messages: list[dict]) -> list[dict]:
    """IR 阶段：用完整的 ir_observation（含 EB 规则）替换最后一条 user message。"""
    from prompts import ir_observation

    query = _extract_query_from_messages(messages)
    domain = "vod"
    for msg in messages:
        if msg.get("role") == "assistant":
            try:
                data = json.loads(msg.get("content", ""))
                if data.get("domain"):
                    domain = data["domain"]
            except (json.JSONDecodeError, TypeError):
                pass

    # 用完整的 ir_observation 生成 prompt（含 EB prompt 层 27 条规则）
    full_obs = ir_observation(query, domain, "", intent="", use_experience_bank=True)

    # 替换最后一条 user message
    enhanced = list(messages)
    for i in range(len(enhanced) - 1, -1, -1):
        if enhanced[i].get("role") == "user":
            enhanced[i] = {"role": "user", "content": full_obs}
            break
    return enhanced


def _inject_device_eb_prompt(messages: list[dict]) -> list[dict]:
    """Device 阶段：用完整的 device_observation（含 few-shot）替换最后一条 user message。"""
    from prompts import device_observation

    query = _extract_query_from_messages(messages)
    full_obs = device_observation(query, "")

    enhanced = list(messages)
    for i in range(len(enhanced) - 1, -1, -1):
        if enhanced[i].get("role") == "user":
            enhanced[i] = {"role": "user", "content": full_obs}
            break
    return enhanced


# ==========================================================================
# 后处理：根据阶段对模型输出做 compile / EB
# ==========================================================================

def _postprocess(stage: str, raw_output: dict, messages: list[dict],
                 guided_json: Optional[dict]) -> dict:
    """对模型原始输出做后处理。

    - route / intent_split：无需后处理，直接返回
    - ir：validate + compile + EB compiler 层
    - audio：compile_audio
    - device：compile_device + EB compiler 层
    """
    if stage in ("route", "intent_split"):
        return raw_output

    if stage == "audio":
        tool_name, params = compile_audio(raw_output)
        return {"tool_name": tool_name, "parameters": params}

    if stage == "device":
        # 需要原始 query 来做 compile_device
        query = _extract_query_from_messages(messages)
        tool_name, params = compile_device(raw_output, query=query)
        return {"tool_name": tool_name, "parameters": params}

    if stage == "ir":
        return _postprocess_ir(raw_output, messages, guided_json)

    return raw_output


def _postprocess_ir(raw_output: dict, messages: list[dict],
                    guided_json: Optional[dict]) -> dict:
    """IR 后处理：validate → 自修复 → compile + EB。"""
    query = _extract_query_from_messages(messages)
    domain = raw_output.get("domain", "vod")
    # 从 messages 历史里提取 routed_tool（路由结果通常在前面的 assistant message 里）
    routed_tool = _extract_routed_tool(messages)
    intent = _extract_intent(messages)

    # validate
    try:
        ir = parse_ir(raw_output, domain_hint=domain)
        errs = validate_ir(ir)
    except (IRError, Exception) as e:
        errs = [str(e)]

    if not errs:
        # 校验通过，直接 compile
        actual_tool, params = compile_with_fallback(ir, routed_tool, retext=query, intent=intent)
        return {"tool_name": actual_tool, "parameters": params}

    # 校验失败：自修复
    repair_messages = list(messages)
    repair_messages.append({"role": "assistant", "content": json.dumps(raw_output, ensure_ascii=False)})

    schema = guided_json or build_ir_schema(domain)

    for attempt in range(_cfg.max_repairs):
        repair_messages.append({"role": "user", "content": repair_observation(errs)})
        repaired = _backend.complete_json(repair_messages, guided_json=schema)
        repair_messages.append({"role": "assistant", "content": json.dumps(repaired, ensure_ascii=False)})

        try:
            ir = parse_ir(repaired, domain_hint=domain)
            errs = validate_ir(ir)
        except (IRError, Exception) as e:
            errs = [str(e)]

        if not errs:
            actual_tool, params = compile_with_fallback(ir, routed_tool, retext=query, intent=intent)
            return {"tool_name": actual_tool, "parameters": params}

        logger.info(f"IR repair #{attempt+1} failed: {errs}")

    # 所有重试失败，仍尝试 compile（best effort）
    try:
        ir = parse_ir(repaired if 'repaired' in dir() else raw_output, domain_hint=domain)
        actual_tool, params = compile_with_fallback(ir, routed_tool, retext=query, intent=intent)
        return {"tool_name": actual_tool, "parameters": params}
    except Exception:
        return raw_output  # 兜底返回原始输出


def _extract_query_from_messages(messages: list[dict]) -> str:
    """从 messages 中提取用户原始 query（第一条 user message 或最后一条）。"""
    # 通常第一条 user message 包含原始 query
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # 提取引号内的 query（常见 prompt 格式：用户说："..."）
            for marker in ["用户说：", "用户请求：", "query：", "用户输入："]:
                if marker in content:
                    rest = content.split(marker, 1)[1].strip()
                    # 取引号内内容
                    if rest.startswith('"') or rest.startswith('"') or rest.startswith('「'):
                        for open_q, close_q in [('"', '"'), ('"', '"'), ('「', '」'), ("'", "'")]:
                            if rest.startswith(open_q):
                                end = rest.find(close_q, 1)
                                if end > 0:
                                    return rest[1:end]
                    # 取到换行
                    line = rest.split("\n")[0].strip().strip('"').strip('"').strip('」').strip("'")
                    if line:
                        return line
            # 没有明确标记，取整段（如果短的话）
            if len(content) < 100:
                return content
    return ""


def _extract_routed_tool(messages: list[dict]) -> str:
    """从 messages 的 assistant 回复中提取路由工具名。"""
    for msg in messages:
        if msg.get("role") == "assistant":
            try:
                data = json.loads(msg.get("content", ""))
                if "tool" in data:
                    return data["tool"]
            except (json.JSONDecodeError, TypeError):
                pass
    return "vod_search"


def _extract_intent(messages: list[dict]) -> str:
    """从 messages 的 assistant 回复中提取 intent。"""
    for msg in messages:
        if msg.get("role") == "assistant":
            try:
                data = json.loads(msg.get("content", ""))
                if "intent" in data:
                    return data["intent"]
            except (json.JSONDecodeError, TypeError):
                pass
    return ""


# ==========================================================================
# API —— 与 vLLM /v1/chat/completions 完全一致
# ==========================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """接口与 vLLM 100% 一致。无任何扩展字段。"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={
            "error": {"message": "Invalid JSON", "type": "invalid_request_error"}
        })

    request_id = uuid.uuid4().hex[:12]
    start_time = time.time()
    model = body.get("model", "")
    messages = body.get("messages", [])

    # 提取上层可能传的 guided_json（VLLMClient 就是这样传的）
    guided_json = body.get("guided_json") or (body.get("structured_outputs") or {}).get("json")

    try:
        # 1. 识别阶段
        stage = _detect_stage(messages, guided_json)

        # 2. 用中间层内部的完整 schema（上层传的仅用于阶段识别）
        full_schema = _auto_schema(stage, messages)
        if full_schema:
            guided_json = full_schema

        # 3. 注入 EB prompt 层规则（对上层透明）
        enhanced_messages = _inject_eb_prompt(stage, messages)

        # 4. 调后端 vLLM（约束解码）
        raw_output = _backend.complete_json(enhanced_messages, guided_json=guided_json)

        # 5. 后处理（compile / EB / 自修复）—— 默认开启
        if os.environ.get("ENHANCED_POSTPROCESS", "1").lower() in ("0", "false", "no"):
            final_output = raw_output
        else:
            final_output = _postprocess(stage, raw_output, enhanced_messages, guided_json)

        # 5. 返回标准 chat/completions 响应
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] stage={stage} ({elapsed:.2f}s)")

        content_str = json.dumps(final_output, ensure_ascii=False)
        return JSONResponse(status_code=200, content={
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content_str},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{request_id}] ERROR ({elapsed:.2f}s): {e}\n{traceback.format_exc()}")
        # 失败时仍返回合法响应
        return JSONResponse(status_code=200, content={
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps({"error": str(e)}, ensure_ascii=False)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


def _auto_schema(stage: str, messages: list[dict]) -> Optional[dict]:
    """当上层没传 guided_json 时，根据阶段自动生成 schema。"""
    if stage == "route":
        return build_route_schema()
    elif stage == "ir":
        # 从 messages 里推断 domain
        domain = "vod"
        for msg in messages:
            if msg.get("role") == "assistant":
                try:
                    data = json.loads(msg.get("content", ""))
                    if data.get("domain"):
                        domain = data["domain"]
                except (json.JSONDecodeError, TypeError):
                    pass
        return build_ir_schema(domain)
    elif stage == "audio":
        return build_audio_schema()
    elif stage == "device":
        return build_device_schema()
    elif stage == "intent_split":
        return build_intent_split_schema()
    return None


# --------------------------------------------------------------------------
# 辅助端点
# --------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    try:
        resp = await _http_client.get(f"{_cfg.backend_base_url}/models",
                                      headers={"Authorization": f"Bearer {_cfg.backend_api_key}"},
                                      timeout=10.0)
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})


@app.get("/health")
async def health():
    backend_ok = False
    try:
        resp = await _http_client.get(f"{_cfg.backend_base_url}/models",
                                      headers={"Authorization": f"Bearer {_cfg.backend_api_key}"},
                                      timeout=5.0)
        backend_ok = resp.status_code == 200
    except Exception:
        pass
    return {"status": "ok", "backend_url": _cfg.backend_base_url, "backend_reachable": backend_ok}


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    global _backend, _http_client, _cfg
    if _cfg is None:
        # 从环境变量读取（uvicorn 重加载模块时 main() 设置的全局变量会丢失）
        _cfg = ServerConfig(
            backend_base_url=os.environ.get("ENHANCED_BACKEND_URL", "http://localhost:8000/v1"),
            backend_api_key=os.environ.get("ENHANCED_BACKEND_KEY", "EMPTY"),
            backend_timeout=float(os.environ.get("ENHANCED_BACKEND_TIMEOUT", "60")),
            backend_request_format=os.environ.get("ENHANCED_REQUEST_FORMAT", "structured_outputs"),
            max_repairs=int(os.environ.get("ENHANCED_MAX_REPAIRS", "2")),
        )
    _http_client = httpx.AsyncClient()
    model = os.environ.get("VLLM_MODEL", "qwen-30b-moe")
    _backend = VLLMClient(VLLMConfig(
        base_url=_cfg.backend_base_url,
        model=model,
        api_key=_cfg.backend_api_key,
        timeout=_cfg.backend_timeout,
        request_format=_cfg.backend_request_format,
    ))
    logger.info(f"Enhanced vLLM ready → backend: {_cfg.backend_base_url}, model: {model}")


@app.on_event("shutdown")
async def on_shutdown():
    if _http_client:
        await _http_client.aclose()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    global _cfg
    parser = argparse.ArgumentParser(description="Enhanced vLLM Middleware")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--backend-key", default="EMPTY")
    parser.add_argument("--backend-timeout", type=float, default=60.0)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--request-format", default="structured_outputs",
                        choices=["structured_outputs", "guided"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    backend_url = args.backend or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    # 通过环境变量传递给 on_startup（uvicorn 重加载模块时全局变量会丢失）
    os.environ["ENHANCED_BACKEND_URL"] = backend_url
    os.environ["ENHANCED_BACKEND_KEY"] = args.backend_key
    os.environ["ENHANCED_BACKEND_TIMEOUT"] = str(args.backend_timeout)
    os.environ["ENHANCED_REQUEST_FORMAT"] = args.request_format
    os.environ["ENHANCED_MAX_REPAIRS"] = str(args.max_repairs)

    logger.info(f"Enhanced vLLM: {args.host}:{args.port} → {backend_url}")
    uvicorn.run("server:app", host=args.host, port=args.port,
                reload=args.reload, workers=args.workers if not args.reload else 1,
                log_level="info")


if __name__ == "__main__":
    main()
