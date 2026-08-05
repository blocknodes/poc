"""Few-shot 配置加载器 —— 从 YAML 文件加载，支持热加载 + 后续动态检索扩展。

设计目标：
  1. few-shot 数据与 prompt 逻辑解耦，方便独立维护/审核。
  2. 文件级缓存，首次加载后常驻内存，可通过 reload_all() 热刷新。
  3. 预留 retrieve_fewshots() 接口，后续接入向量检索按 query 相似度选 shot。
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

_FEWSHOT_DIR = Path(__file__).parent / "fewshots"

# 内存缓存：key = "{stage}_{domain}" -> [(query, response_dict), ...]
_cache: dict[str, list[tuple[str, dict]]] = {}

# 文件名映射表
_FILE_MAP: dict[tuple[str, str | None], str] = {
    ("intent_split", None): "intent_split.yaml",
    ("route", "vod"): "route_vod.yaml",
    ("route", "educ"): "route_educ.yaml",
    ("route", "full"): "route_full.yaml",
    ("route", None): "route_full.yaml",
    ("ir", "vod"): "ir_vod.yaml",
    ("ir", "educ"): "ir_educ.yaml",
    ("slot_fill", "audio"): "audio.yaml",
    ("slot_fill", "device"): "device.yaml",
}


def load_fewshots(stage: str, domain: str | None = None,
                  reload: bool = False) -> list[tuple[str, dict]]:
    """加载指定阶段+域的 few-shot 列表。

    Args:
        stage: route | ir | slot_fill | intent_split
        domain: vod | educ | audio | device | full | None
        reload: True 时强制重新从文件读取（热加载）

    Returns:
        [(query, response_dict), ...]  直接用于 _render_fewshot
    """
    key = f"{stage}_{domain or 'default'}"
    if not reload and key in _cache:
        return _cache[key]

    filename = _FILE_MAP.get((stage, domain))
    if filename is None:
        # 尝试自动拼接
        filename = f"{stage}_{domain}.yaml" if domain else f"{stage}.yaml"

    path = _FEWSHOT_DIR / filename
    if not path.exists():
        _cache[key] = []
        return []

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    shots = [(s["query"], s["response"]) for s in data.get("shots", [])]
    _cache[key] = shots
    return shots


def reload_all() -> None:
    """清空所有缓存，下次 load 时重新从文件读取。支持运行时热更新 YAML。"""
    _cache.clear()


def retrieve_fewshots(stage: str, domain: str | None, query: str,
                      top_k: int = 5) -> list[tuple[str, dict]]:
    """基于 query 相似度动态选取 few-shot（后续实现）。

    当前实现：直接返回全量 shots（等价于静态模式）。
    后续：接入向量索引（如 faiss/annoy），按 embedding 余弦相似度取 top_k。

    Args:
        stage: 阶段名
        domain: 域名
        query: 用户原始 query，用于相似度匹配
        top_k: 返回条数上限

    Returns:
        [(query, response_dict), ...] 最多 top_k 条
    """
    all_shots = load_fewshots(stage, domain)
    # TODO: 接入向量检索，按 query embedding 相似度排序取 top_k
    return all_shots[:top_k]


def list_available() -> dict[str, int]:
    """列出所有可用的 few-shot 文件及其条数（调试/管理用）。"""
    result = {}
    if not _FEWSHOT_DIR.exists():
        return result
    for f in sorted(_FEWSHOT_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            count = len(data.get("shots", []))
            result[f.stem] = count
        except Exception:
            result[f.stem] = -1
    return result
