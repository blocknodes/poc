#!/usr/bin/env python3
"""server.py —— 慢任务接口。

启动：
    VLLM_BASE_URL=http://localhost:9000/v1 VLLM_MODEL=baseline python slim/server.py --port 8082
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("ERROR: pip install fastapi uvicorn requests")
    sys.exit(1)

from vllm_client import VLLMClient, VLLMConfig
from planner import Planner, PlanResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("server")

app = FastAPI(title="slow_agent", version="1.0.0")

_planner: Optional[Planner] = None


@app.on_event("startup")
async def on_startup():
    global _planner
    debug = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    cfg = VLLMConfig(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:9000/v1"),
        model=os.environ.get("VLLM_MODEL", "baseline"),
        timeout=float(os.environ.get("VLLM_TIMEOUT", "60")),
    )
    client = VLLMClient(cfg, debug=debug)
    _planner = Planner(client)
    logger.info(f"Planner ready → {cfg.base_url} / {cfg.model} (debug={debug})")


@app.post("/slowAgent/poc_1784284048243")
async def slow_agent(request: Request):
    start = time.time()

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"code": 400, "message": "JSON 解析失败"})

    trace_id = body.get("traceId", "")
    device_id = body.get("deviceId", "")
    data = body.get("data") or {}
    query = (data.get("query") or "").strip()

    if not query:
        return JSONResponse(status_code=400, content={
            "traceId": trace_id, "code": 400, "message": "query 为空"
        })

    # 提取 available_tools
    tool_list = data.get("toolList") or []
    available_tools = None
    if tool_list:
        available_tools = [t["toolName"] for t in tool_list
                          if isinstance(t, dict) and t.get("toolName")]

    # 提取 memory_hint
    memory_hint = _extract_memory(data)

    logger.info(f"[{trace_id}] query='{query}'")

    try:
        results = _planner.plan_multi(query, memory_hint=memory_hint,
                                      available_tools=available_tools)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[{trace_id}] ERROR ({elapsed:.2f}s): {e}")
        return JSONResponse(status_code=200, content={
            "traceId": trace_id, "deviceId": device_id,
            "code": 500, "message": str(e),
            "data": {"steps": []}
        })

    # 构建响应
    steps = []
    for i, r in enumerate(results):
        if r.tool_name == "error":
            continue
        steps.append({
            "id": f"s{i+1}",
            "toolName": r.tool_name,
            "parameters": r.parameters,
            "dependsOn": [],
            "retext": query,
        })

    elapsed = time.time() - start
    confidence = results[0].confidence if results and results[0].confidence else 0.9
    logger.info(f"[{trace_id}] → {[s['toolName'] for s in steps]} ({elapsed:.2f}s)")

    return JSONResponse(status_code=200, content={
        "traceId": trace_id,
        "deviceId": device_id,
        "code": 200,
        "message": "success",
        "data": {
            "planId": f"plan_{uuid.uuid4().hex[:4]}",
            "schemaVersion": "1.0",
            "planType": "execute",
            "steps": steps,
            "planConfidence": confidence,
        },
    })


def _extract_memory(data: dict) -> str:
    memory = data.get("memory") or {}
    short_mem = memory.get("shortMemory") or []
    if not short_mem:
        return ""
    parts = []
    for turn in short_mem[-3:]:
        if isinstance(turn, dict):
            u = turn.get("query", turn.get("user", ""))
            b = turn.get("answer", turn.get("bot", ""))
            if u:
                parts.append(f"用户: {u}")
            if b:
                parts.append(f"助手: {b}")
    return "\n".join(parts)


@app.get("/health")
async def health():
    return {"status": "ok", "ready": _planner is not None}


def main():
    parser = argparse.ArgumentParser(description="慢任务接口")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--debug", action="store_true", help="打印每次 vLLM 调用的输入输出")
    args = parser.parse_args()

    if args.debug:
        os.environ["DEBUG"] = "1"

    logger.info(f"server: {args.host}:{args.port} debug={args.debug}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
