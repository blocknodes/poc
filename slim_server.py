#!/usr/bin/env python3
"""slim_server.py —— 轻量版慢任务接口（自包含，单文件）。

只做一件事：接收请求 → 调 Planner → 返回结果。
无 DAG 推断、无 retext 生成、无复杂依赖。

启动：
    python slim_server.py --debug --log slim.log
    python slim_server.py --stub
    python slim_server.py --port 8083
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
import uuid
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("ERROR: pip install fastapi uvicorn")
    sys.exit(1)

# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("slim_server")

app = FastAPI(title="slim_slow_agent", version="1.0.0")

_planner = None
_debug = os.environ.get("PLANNER_DEBUG", "").lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------
# Init
# --------------------------------------------------------------------------
def _init_planner(stub: bool = False):
    global _planner
    from config import cfg, make_planner, make_vllm_config
    from planner import Planner, VLLMClient

    if stub:
        from demo import make_stub
        client = VLLMClient(responder=make_stub())
        logger.info("[STUB] 离线模式")
    else:
        client = VLLMClient(make_vllm_config())
        logger.info(f"[LIVE] vLLM: {cfg.vllm_base_url} / {cfg.vllm_model}")

    _planner = make_planner(client)
    if cfg.retrieve_enabled:
        logger.info(f"[RETRIEVE] {cfg.retrieve_base_url}")
    logger.info("Planner ready")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
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
    available_tools: Optional[list[str]] = None
    if tool_list:
        available_tools = [t["toolName"] for t in tool_list
                          if isinstance(t, dict) and t.get("toolName")]
        if not available_tools:
            available_tools = None

    # 提取 memory_hint
    memory_hint = ""
    memory = data.get("memory") or {}
    short_mem = memory.get("shortMemory") or []
    if short_mem:
        parts = []
        for turn in short_mem[-3:]:
            if isinstance(turn, dict):
                u = turn.get("query", turn.get("user", ""))
                b = turn.get("answer", turn.get("bot", ""))
                if u: parts.append(f"用户: {u}")
                if b: parts.append(f"助手: {b}")
        memory_hint = "\n".join(parts)

    logger.info(f"[{trace_id}] query='{query}' tools={available_tools}")

    # 调 planner
    try:
        results = _planner.plan_multi(query, memory_hint=memory_hint,
                                      available_tools=available_tools)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"[{trace_id}] ERROR: {e} ({elapsed:.2f}s)\n{traceback.format_exc()}")
        return JSONResponse(status_code=200, content={
            "traceId": trace_id, "deviceId": device_id,
            "code": 500, "message": f"{type(e).__name__}: {e}",
            "data": {"steps": []}
        })

    # debug 打印
    if _debug:
        print(f"\n{'━'*60}")
        print(f"  [{trace_id}] query='{query}'")
        print(f"{'━'*60}")
        print(f"\n  REQUEST BODY:")
        print(json.dumps(body, ensure_ascii=False, indent=2))
        for i, r in enumerate(results):
            print(f"\n  [结果 {i}] tool={r.tool_name} domain={r.domain}")
            print(f"    params={json.dumps(r.parameters, ensure_ascii=False)}")
            for step in (r.trace or []):
                stage = step.get("stage", "?")
                if "guided_json" in step:
                    print(f"    [{stage}] guided_json.tool.enum={step['guided_json'].get('properties',{}).get('tool',{}).get('enum','N/A')}")
                if "output" in step:
                    print(f"    [{stage}] output={json.dumps(step['output'], ensure_ascii=False)}")
        print(f"{'━'*60}\n")

    # 构建响应
    steps = []
    for r in results:
        if r.tool_name == "error":
            continue
        params = dict(r.parameters) if r.parameters else {}
        retext = params.pop("retext", query)
        steps.append({
            "id": f"s{len(steps)+1}",
            "toolName": r.tool_name,
            "parameters": params,
            "dependsOn": [],
            "retext": retext,
        })

    elapsed = time.time() - start
    confidences = [r.route_confidence for r in results if r.route_confidence is not None]
    confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.85

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


@app.get("/health")
async def health():
    return {"status": "ok", "ready": _planner is not None}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    global _debug

    parser = argparse.ArgumentParser(description="轻量版慢任务接口")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--stub", action="store_true", help="离线 stub 模式")
    parser.add_argument("--debug", action="store_true", help="打印完整请求/响应")
    parser.add_argument("--log", type=str, default="", help="日志文件路径")
    args = parser.parse_args()

    _debug = args.debug
    if args.debug:
        os.environ["PLANNER_DEBUG"] = "1"

    # log to file
    if args.log:
        fh = logging.FileHandler(args.log, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(fh)

        class Tee:
            def __init__(self, orig, f):
                self.orig = orig
                self.f = open(f, "a", encoding="utf-8")
            def write(self, s):
                self.orig.write(s); self.f.write(s); self.f.flush()
            def flush(self):
                self.orig.flush(); self.f.flush()
            def isatty(self):
                return hasattr(self.orig, 'isatty') and self.orig.isatty()
            def fileno(self):
                return self.orig.fileno()
            def __getattr__(self, n):
                return getattr(self.orig, n)

        sys.stdout = Tee(sys.stdout, args.log)
        sys.stderr = Tee(sys.stderr, args.log)

    # init planner before uvicorn (single worker)
    _init_planner(stub=args.stub)

    logger.info(f"启动 slim_server: {args.host}:{args.port} debug={args.debug}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
