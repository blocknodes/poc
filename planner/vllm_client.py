"""vLLM 客户端封装。

使用 vLLM 的 OpenAI 兼容接口 (/v1/chat/completions) 做约束解码（structured outputs）。
- vLLM ≥ 0.11：请求走 extra_body.structured_outputs = {"json": <schema>}（新格式，默认）；
  服务端后端用 --structured-outputs-config.backend 指定，默认 auto。
- vLLM ≤ 0.10：设 VLLMConfig.request_format="guided" 切回旧的 guided_json 格式。

离线/无服务环境下可注入一个 stub responder 用于测试，无需真实网络。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - requests 可能未安装
    requests = None


# 一个 responder 接收 (messages, guided_json) 返回模型文本输出，便于测试注入
Responder = Callable[[list[dict], Optional[dict]], str]


@dataclass
class VLLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "qwen-30b-moe"       # 你的 30B MoE
    api_key: str = "EMPTY"
    temperature: float = 0.0          # 结构化生成，低温更稳
    max_tokens: int = 1024
    timeout: float = 60.0
    guided_backend: str = "xgrammar"  # xgrammar | guidance | outlines
    # 请求侧结构化输出格式：
    #   "structured_outputs" —— vLLM ≥ 0.11 的新格式（默认，推荐）
    #   "guided"             —— 旧格式 guided_json，兼容 v0.9 及更早
    request_format: str = "structured_outputs"


class VLLMClient:
    def __init__(self, config: Optional[VLLMConfig] = None, responder: Optional[Responder] = None):
        self.config = config or VLLMConfig()
        # responder 用于测试/离线：若提供则不走网络
        self._responder = responder

    def complete_json(
        self,
        messages: list[dict],
        guided_json: Optional[dict] = None,
        *,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """调用模型并把输出解析为 JSON dict。"""
        text = self._raw_complete(messages, guided_json, max_tokens)
        return _loads_lenient(text)

    def _raw_complete(
        self,
        messages: list[dict],
        guided_json: Optional[dict],
        max_tokens: Optional[int],
    ) -> str:
        if self._responder is not None:
            return self._responder(messages, guided_json)

        if requests is None:
            raise RuntimeError("requests 未安装，且未注入 responder")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        # vLLM 结构化输出（约束解码）
        extra_body: dict[str, Any] = {}
        if guided_json is not None:
            if self.config.request_format == "guided":
                # 旧格式（vLLM ≤ 0.10）：guided_json + guided_decoding_backend
                extra_body["guided_json"] = guided_json
                extra_body["guided_decoding_backend"] = self.config.guided_backend
            else:
                # 新格式（vLLM ≥ 0.11）：统一的 structured_outputs.json
                # backend 在服务端用 --structured-outputs-config.backend 指定，
                # 不在请求里选后端（新版会因请求内后端选择而报错）。
                extra_body["structured_outputs"] = {"json": guided_json}
        payload.update(extra_body)

        resp = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _loads_lenient(text: str) -> dict:
    """容错解析：约束解码通常已是纯 JSON；仍兜底剥离 ```json fenced 代码块。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    # 取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)
