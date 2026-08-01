"""按需 EB 检索层 —— 从用户 query 中确定性提取结构化候选条件。

设计思路（三层检索）：
  1. 枚举映射表（dict_enum）：is_fee / gender / content_type 等硬编码字段
  2. 标签映射表（dict_genre）：children_third_genre / training_objectives 同义词
  3. 标题别名表（dict_title）：IP缩写→标准名

输出：candidates list[dict]，每个 dict 是一个确定性叶子节点。
这些 candidates 会：
  - 注入到 LLM prompt 中作为硬约束（prompt 层）
  - 在 LLM 输出后强制合并（compiler 层兜底）

仅 educ 域使用，不影响 vod 管线。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ===========================================================================
# 枚举映射表：状态字段 / 精确枚举
# ===========================================================================
_ENUM_RULES: list[dict] = [
    # is_fee
    {"field": "is_fee", "value": 0,
     "keywords": ["免费", "不要钱", "白嫖", "不收费", "免会员", "不要VIP",
                  "不要vip", "不用会员", "不用VIP", "不用vip", "不花钱", "不要会员"]},
    # gender
    {"field": "gender", "value": "男",
     "keywords": ["男孩", "男生", "男宝", "小男生", "男孩子"]},
    {"field": "gender", "value": "女",
     "keywords": ["女孩", "女生", "女宝", "小女生", "女孩子"]},
    # content_type（仅当明确提到时触发）
    {"field": "content_type", "value": "动画",
     "keywords": ["动画片", "动画片儿"]},
    {"field": "content_type", "value": "卡通",
     "keywords": ["卡通片", "卡通动画"]},
]

# ===========================================================================
# 标签/题材映射表：genre3 + training_objectives
# ===========================================================================
_GENRE_RULES: list[dict] = [
    # children_third_genre
    {"field": "children_third_genre", "value": "恐龙",
     "keywords": ["恐龙", "霸王龙", "三角龙", "翼龙"]},
    {"field": "children_third_genre", "value": "古诗词",
     "keywords": ["古诗", "古诗词", "唐诗", "宋词", "背古诗"]},
    {"field": "children_third_genre", "value": "汉字",
     "keywords": ["汉字", "写字", "识字"]},
    {"field": "children_third_genre", "value": "舞蹈",
     "keywords": ["跳舞", "舞蹈", "跳个舞"]},
    {"field": "children_third_genre", "value": "搞笑",
     "keywords": ["搞笑", "搞笑视频"]},
    {"field": "children_third_genre", "value": "科幻",
     "keywords": ["科幻"]},
    {"field": "children_third_genre", "value": "冒险",
     "keywords": ["冒险"]},
    {"field": "children_third_genre", "value": "热血",
     "keywords": ["热血"]},
    {"field": "children_third_genre", "value": "机甲",
     "keywords": ["机甲", "机器人"]},
    # children_second_genre
    {"field": "children_second_genre", "value": "动画电影",
     "keywords": ["动画电影", "卡通电影"]},
    {"field": "children_second_genre", "value": "绘本",
     "keywords": ["绘本"]},
    {"field": "children_second_genre", "value": "儿歌",
     "keywords": ["儿歌"]},
    {"field": "children_second_genre", "value": "故事",
     "keywords": ["的故事", "故事吧", "讲故事"]},
    {"field": "children_second_genre", "value": "少儿动漫",
     "keywords": ["动漫"]},
    {"field": "children_second_genre", "value": "早教",
     "keywords": ["早教"]},
    {"field": "children_second_genre", "value": "玩具",
     "keywords": ["玩具"]},
    # training_objectives
    {"field": "training_objectives", "value": "情绪管理",
     "keywords": ["情绪管理", "管理情绪"]},
    {"field": "training_objectives", "value": "英语启蒙",
     "keywords": ["英语启蒙", "英文启蒙"]},
]

# ===========================================================================
# 国产 / 语言 / 地区映射
# ===========================================================================
_COUNTRY_RULES: list[dict] = [
    {"field": "country", "values": ["内地", "中国澳门", "中国香港", "中国台湾"], "operator": "or",
     "keywords": ["国产", "中国版", "中国的"]},
    {"field": "country", "value": "日本",
     "keywords": ["日本的", "日本动漫", "日本动画"]},
    {"field": "country", "value": "美国",
     "keywords": ["美国的", "美国动画"]},
]

_LANGUAGE_RULES: list[dict] = [
    {"field": "language", "value": "英文",
     "keywords": ["英文版", "英语版", "英文的", "英语动画"]},
    {"field": "language", "value": "普通话",
     "keywords": ["中文版", "国语版", "普通话版", "中文的"]},
    {"field": "language", "value": "日语",
     "keywords": ["日文版", "日语版", "日文的"]},
]


# ===========================================================================
# 检索主函数
# ===========================================================================
@dataclass
class RetrievalResult:
    """检索结果：确定性候选条件列表。"""
    candidates: list[dict] = field(default_factory=list)
    prompt_hint: str = ""  # 注入 LLM prompt 的文本


def retrieve_candidates(query: str, domain: str = "educ") -> RetrievalResult:
    """从 query 中提取确定性候选条件。

    Args:
        query: 用户原始请求文本
        domain: 当前域（仅 educ 生效）

    Returns:
        RetrievalResult 含 candidates 列表和 prompt 提示文本
    """
    if domain != "educ":
        return RetrievalResult()

    candidates: list[dict] = []
    q = query.strip()

    # 1. 枚举映射
    for rule in _ENUM_RULES:
        for kw in rule["keywords"]:
            if kw in q:
                cand = {"field": rule["field"], "value": rule["value"]}
                if cand not in candidates:
                    candidates.append(cand)
                break

    # 2. 标签/题材映射
    for rule in _GENRE_RULES:
        for kw in rule["keywords"]:
            if kw in q:
                cand = {"field": rule["field"], "value": rule["value"]}
                if cand not in candidates:
                    candidates.append(cand)
                break

    # 3. 国家/地区
    for rule in _COUNTRY_RULES:
        for kw in rule["keywords"]:
            if kw in q:
                if "values" in rule:
                    cand = {"field": rule["field"], "values": rule["values"],
                            "operator": rule.get("operator", "or")}
                else:
                    cand = {"field": rule["field"], "value": rule["value"]}
                if cand not in candidates:
                    candidates.append(cand)
                break

    # 4. 语言
    for rule in _LANGUAGE_RULES:
        for kw in rule["keywords"]:
            if kw in q:
                cand = {"field": rule["field"], "value": rule["value"]}
                if cand not in candidates:
                    candidates.append(cand)
                break

    # 构造 prompt hint
    prompt_hint = ""
    if candidates:
        import json
        lines = []
        for c in candidates:
            if "values" in c:
                lines.append(f'  {c["field"]}: {c["values"]} (operator={c.get("operator","or")})')
            else:
                lines.append(f'  {c["field"]}: {c["value"]}')
        prompt_hint = "系统已从用户请求中提取以下确定性条件（严禁遗漏）：\n" + "\n".join(lines)

    return RetrievalResult(candidates=candidates, prompt_hint=prompt_hint)


# ===========================================================================
# 后置强制合并
# ===========================================================================
def force_merge_candidates(params: dict[str, Any], candidates: list[dict]) -> None:
    """强制将检索到的候选条件合并到编译后的 params 中。

    逻辑：
    1. 遍历 candidates，检查 params["query"] 中是否已包含该条件。
    2. 如果缺失，强制注入到根节点的 AND 中。

    Args:
        params: 编译后的参数（含 query 树）
        candidates: 检索提取的确定性条件列表
    """
    query = params.get("query")
    if not query:
        return

    # 提取已有叶子
    existing = set()
    _collect_leaves(query, existing)

    # 筛选缺失的
    missing: list[dict] = []
    for cand in candidates:
        if "value" in cand:
            key = f'{cand["field"]}::{cand["value"]}'
            if key not in existing:
                missing.append({"field": cand["field"], "value": cand["value"]})
        elif "values" in cand:
            # 检查是否已包含任一值
            has_any = any(f'{cand["field"]}::{v}' in existing for v in cand["values"])
            if not has_any:
                missing.append({
                    "field": cand["field"],
                    "values": cand["values"],
                    "operator": cand.get("operator", "or"),
                })

    if not missing:
        return

    # 强制注入到根 AND
    if isinstance(query, dict) and "and" in query:
        query["and"].extend(missing)
    elif isinstance(query, dict) and "field" in query:
        params["query"] = {"and": [query] + missing}
    else:
        params["query"] = {"and": [query] + missing}


def _collect_leaves(node: Any, out: set) -> None:
    """递归收集所有叶子节点的 field::value 键。"""
    if not isinstance(node, dict):
        return
    if "field" in node:
        if "value" in node:
            out.add(f'{node["field"]}::{node["value"]}')
        if "values" in node:
            for v in node["values"]:
                out.add(f'{node["field"]}::{v}')
    for k in ("and", "or"):
        if k in node and isinstance(node[k], list):
            for item in node[k]:
                _collect_leaves(item, out)
    if "not" in node:
        _collect_leaves(node["not"], out)
