"""检索层客户端 —— 调用 /retrieve 接口获取工具/参数/值候选。

该模块为可选组件，通过 Planner(use_retrieve=True) 开关控制。
关闭时整条管线与原有逻辑完全一致，不产生任何额外开销或副作用。

接口来源：lizheng21 搭建的慢任务规划-检索层 API
    POST /retrieve
    Body: {query, domain?, tool_k, parameter_k, value_k}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


@dataclass
class RetrieveConfig:
    """检索层接口配置。"""
    base_url: str = "http://10.18.231.63:31935"
    tool_k: int = 3
    parameter_k: int = 6
    value_k: int = 6
    timeout: float = 5.0  # 检索不应阻塞主流程太久


@dataclass
class RetrieveResult:
    """检索层返回结果的结构化封装。"""
    tools: list[dict] = field(default_factory=list)
    parameters: list[dict] = field(default_factory=list)
    values: list[dict] = field(default_factory=list)
    suggested: Optional[dict] = None
    raw: Optional[dict] = None  # 原始响应，供 trace 保存

    @property
    def top_tool(self) -> Optional[str]:
        """排名第一的工具名。"""
        if self.tools:
            return self.tools[0].get("tool_name")
        return None

    @property
    def top_tool_domain(self) -> Optional[str]:
        """排名第一的工具所属域。"""
        if self.tools:
            return self.tools[0].get("domain")
        return None

    def format_tool_hint(self, max_tools: int = 3) -> str:
        """格式化工具候选提示，供注入 prompt。"""
        if not self.tools:
            return ""
        lines = []
        for t in self.tools[:max_tools]:
            name = t.get("tool_name", "?")
            domain = t.get("domain", "?")
            score = t.get("score", 0)
            lines.append(f"  {domain}::{name} (score={score:.4f})")
        return "检索系统建议工具:\n" + "\n".join(lines)

    def format_parameter_hint(self, tool_name: Optional[str] = None,
                              max_params: int = 6) -> str:
        """格式化参数候选提示，供注入 IR/slot-fill prompt。
        
        Args:
            tool_name: 若指定，只保留属于该工具的参数。
            max_params: 最多展示几个参数。
        """
        params = self.parameters
        if tool_name:
            params = [p for p in params if p.get("tool_name") == tool_name]
        if not params:
            return ""
        # 按 parameter_id 去重，保留最高分的
        seen: dict[str, dict] = {}
        for p in params:
            pid = p.get("parameter_id", "?")
            if pid not in seen or p.get("score", 0) > seen[pid].get("score", 0):
                seen[pid] = p
        deduped = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
        lines = []
        for p in deduped[:max_params]:
            pid = p.get("parameter_id", "?")
            score = p.get("score", 0)
            tool = p.get("tool_name", "")
            lines.append(f"  {pid} (score={score:.4f}, tool={tool})")
        return "检索建议相关参数:\n" + "\n".join(lines)

    def format_value_hint(self, max_values: int = 6) -> str:
        """格式化取值候选提示。"""
        if not self.values:
            return ""
        lines = []
        for v in self.values[:max_values]:
            pid = v.get("parameter_id", "?")
            val = v.get("value", "?")
            score = v.get("score", 0)
            lines.append(f"  {pid}={val} (score={score:.4f})")
        return "检索建议取值:\n" + "\n".join(lines)


class RetrieveClient:
    """检索层 HTTP 客户端。

    调用失败时静默降级（返回空 RetrieveResult），不阻塞主流程。
    """

    def __init__(self, config: Optional[RetrieveConfig] = None):
        self.config = config or RetrieveConfig()

    def retrieve(self, query: str, domain: Optional[str] = None,
                 tool_k: Optional[int] = None,
                 parameter_k: Optional[int] = None,
                 value_k: Optional[int] = None) -> RetrieveResult:
        """调用检索层获取候选。

        Args:
            query: 用户原始请求
            domain: 可选，指定域（不给则全域搜索）
            tool_k/parameter_k/value_k: 可覆盖默认 top-k

        Returns:
            RetrieveResult，调用失败时返回空结果。
        """
        if requests is None:
            return RetrieveResult()

        payload: dict[str, Any] = {
            "query": query,
            "tool_k": tool_k or self.config.tool_k,
            "parameter_k": parameter_k or self.config.parameter_k,
            "value_k": value_k or self.config.value_k,
        }
        if domain:
            payload["domain"] = domain

        try:
            resp = requests.post(
                f"{self.config.base_url}/retrieve",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload, ensure_ascii=False),
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # 检索层不可用时静默降级，不影响主流程
            return RetrieveResult()

        return self._parse_response(data)

    @staticmethod
    def _parse_response(data: dict) -> RetrieveResult:
        """解析检索层返回，兼容两种格式：

        格式 A（嵌套）：parameters/values 嵌套在每个 tool 内部
            tools: [{tool_name, parameters: [{parameter_id, values: [{value}]}]}]

        格式 B（扁平，旧版）：顶层独立的 tools/parameters/values 数组
            tools: [...], parameters: [...], values: [...]
        """
        raw_tools = data.get("tools", [])

        # 检测格式：如果 tools[0] 有 "parameters" 字段，则为嵌套格式
        is_nested = (raw_tools and isinstance(raw_tools[0], dict)
                     and "parameters" in raw_tools[0])

        if is_nested:
            # 嵌套格式：从 tools 中展开 parameters 和 values
            tools: list[dict] = []
            parameters: list[dict] = []
            values: list[dict] = []

            for t in raw_tools:
                tool_name = t.get("tool_name", "")
                domain = t.get("domain", "")
                tools.append({
                    "rank": t.get("rank"),
                    "score": t.get("score", 0),
                    "tool_id": t.get("tool_id", ""),
                    "tool_name": tool_name,
                    "domain": domain,
                    "title": t.get("title", ""),
                })
                for p in t.get("parameters", []):
                    param_entry = {
                        "parameter_id": p.get("parameter_id", ""),
                        "tool_name": tool_name,
                        "domain": domain,
                        "score": p.get("score", 0),
                        "required": p.get("required", False),
                    }
                    parameters.append(param_entry)
                    for v in p.get("values", []):
                        values.append({
                            "parameter_id": p.get("parameter_id", ""),
                            "tool_name": tool_name,
                            "value": v.get("value", ""),
                            "score": v.get("score", 0),
                        })

            return RetrieveResult(
                tools=tools,
                parameters=parameters,
                values=values,
                suggested=data.get("suggested"),
                raw=data,
            )
        else:
            # 扁平格式（旧版 / 或未来变更后的格式）
            return RetrieveResult(
                tools=raw_tools,
                parameters=data.get("parameters", []),
                values=data.get("values", []),
                suggested=data.get("suggested"),
                raw=data,
            )
