"""vLLM 客户端 —— OpenAI 兼容 chat/completions 接口。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

try:
    import requests
except ImportError:
    requests = None


@dataclass
class VLLMConfig:
    base_url: str = "http://localhost:9000/v1"
    model: str = "baseline"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 60.0


class VLLMClient:
    """OpenAI 兼容的 chat/completions 客户端。"""

    def __init__(self, config: Optional[VLLMConfig] = None, debug: bool = False):
        self.config = config or VLLMConfig()
        self.debug = debug

    def complete(self, messages: list[dict], *,
                 guided_json: Optional[dict] = None,
                 max_tokens: Optional[int] = None) -> dict:
        """发送 messages，返回解析后的 JSON dict。"""
        if requests is None:
            raise RuntimeError("requests 未安装")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if guided_json is not None:
            payload["structured_outputs"] = {"json": guided_json}

        if self.debug:
            print(f"\n{'─'*60}")
            print(f"  [REQUEST] POST {self.config.base_url}/chat/completions")
            print(f"  messages ({len(messages)} 轮):")
            for i, m in enumerate(messages):
                content = m.get("content", "")
                preview = content[:200].replace("\n", "\\n") + ("..." if len(content) > 200 else "")
                print(f"    [{i}] {m['role']}: {preview}")
            if guided_json:
                print(f"  guided_json: {json.dumps(guided_json, ensure_ascii=False)[:300]}")

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
        content = data["choices"][0]["message"]["content"]

        if self.debug:
            print(f"  [RESPONSE] {content}")
            print(f"{'─'*60}")

        return _parse_json(content)

    def complete_raw(self, messages: list[dict], *,
                     guided_json: Optional[dict] = None,
                     max_tokens: Optional[int] = None) -> str:
        """发送 messages，返回原始 content 字符串。"""
        if requests is None:
            raise RuntimeError("requests 未安装")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if guided_json is not None:
            payload["structured_outputs"] = {"json": guided_json}

        if self.debug:
            print(f"\n{'─'*60}")
            print(f"  [REQUEST] POST {self.config.base_url}/chat/completions")
            for i, m in enumerate(messages):
                content = m.get("content", "")
                preview = content[:200].replace("\n", "\\n") + ("..." if len(content) > 200 else "")
                print(f"    [{i}] {m['role']}: {preview}")

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
        content = data["choices"][0]["message"]["content"]

        if self.debug:
            print(f"  [RESPONSE] {content}")
            print(f"{'─'*60}")

        return content


def _parse_json(text: str) -> dict:
    """容错 JSON 解析。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)
