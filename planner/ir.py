"""IR (Intermediate Representation) —— 域无关的布尔查询中间表示。

模型只需产出这一种结构，无需关心影视/少儿、也无需关心最终工具是精确检索
(*_search, 嵌套 JSON) 还是慢链路 (*_slow_search, 扁平字符串)。序列化差异由
compiler 消化。

IR JSON 形态::

    {
      "domain": "vod" | "educ",
      "action": "search" | "play",          # 可选，默认 search
      "query": <Node>,                        # 布尔查询树
      "sort":  [ {"key": "rate", "order": "desc"} ],   # 可选
      "playback": {"series": 2, "videoIndex": 3}       # 可选，仅 vod
    }

Node 四种形态::

    {"and": [Node, ...]}
    {"or":  [Node, ...]}
    {"not": Node}
    {"field": "actor", "value": "刘德华"}                    # 精确单值
    {"field": "actor", "values": ["刘德华","吴京"], "op": "and"}  # 精确多值
    {"field": "fee",   "value": 0}                          # 状态
    {"field": "release_year", "range": {"from": "20200101", "to": "*"}}  # 范围
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .registry import (
    Kind,
    PLAYBACK_FIELDS,
    SORT_REGISTRY,
    get_field,
)


class IRError(ValueError):
    """IR 结构或语义非法。"""


# ---------------------------------------------------------------------------
# 节点数据类
# ---------------------------------------------------------------------------
@dataclass
class Leaf:
    field: str
    value: Optional[Union[str, int, float]] = None
    values: Optional[list[str]] = None
    op: str = "or"                      # 仅 values 时生效
    range: Optional[dict[str, Any]] = None  # {"from":..., "to":...}


@dataclass
class And:
    items: list["Node"]


@dataclass
class Or:
    items: list["Node"]


@dataclass
class Not:
    item: "Node"


Node = Union[Leaf, And, Or, Not]


@dataclass
class SortItem:
    key: str
    order: str  # asc | desc


@dataclass
class IR:
    domain: str
    query: Node
    action: str = "search"
    sort: list[SortItem] = field(default_factory=list)
    playback: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 解析：dict -> IR（模型输出的 JSON 反序列化）
# ---------------------------------------------------------------------------
def parse_node(obj: Any) -> Node:
    if not isinstance(obj, dict):
        raise IRError(f"节点必须是对象，收到: {type(obj).__name__}")
    keys = set(obj.keys())
    if "and" in keys:
        _require_list(obj["and"], "and")
        return And([parse_node(x) for x in obj["and"]])
    if "or" in keys:
        _require_list(obj["or"], "or")
        return Or([parse_node(x) for x in obj["or"]])
    if "not" in keys:
        return Not(parse_node(obj["not"]))
    if "field" in keys:
        return Leaf(
            field=obj["field"],
            value=obj.get("value"),
            values=obj.get("values"),
            op=obj.get("op", "or"),
            range=obj.get("range"),
        )
    raise IRError(f"无法识别的节点，缺少 and/or/not/field: {obj}")


def _require_list(v: Any, name: str) -> None:
    if not isinstance(v, list) or not v:
        raise IRError(f"{name} 必须是非空数组")


def parse_ir(obj: dict) -> IR:
    if not isinstance(obj, dict):
        raise IRError("IR 根必须是对象")
    domain = obj.get("domain")
    if domain not in ("vod", "educ"):
        raise IRError(f"domain 必须是 vod 或 educ，收到: {domain!r}")
    if "query" not in obj:
        raise IRError("IR 缺少 query")
    action = obj.get("action", "search")
    if action not in ("search", "play"):
        raise IRError(f"action 必须是 search 或 play，收到: {action!r}")
    sort = [SortItem(s["key"], s["order"]) for s in obj.get("sort", [])]
    playback = dict(obj.get("playback", {}))
    return IR(
        domain=domain,
        query=parse_node(obj["query"]),
        action=action,
        sort=sort,
        playback=playback,
    )


# ---------------------------------------------------------------------------
# IR 层语义校验（结构由 grammar 保证，这里管 registry 相关的语义）
# ---------------------------------------------------------------------------
def validate_ir(ir: IR) -> list[str]:
    """返回错误信息列表；空列表表示通过。这些信息可回灌给模型做自修复。"""
    errs: list[str] = []
    _validate_node(ir.query, ir.domain, errs)

    for s in ir.sort:
        spec = SORT_REGISTRY.get(s.key)
        if spec is None:
            errs.append(f"未知排序键 '{s.key}'")
        elif ir.domain not in spec.domains:
            errs.append(f"排序键 '{s.key}' 在 domain '{ir.domain}' 不可用")
        if s.order not in ("asc", "desc"):
            errs.append(f"排序 order 必须是 asc/desc，收到 '{s.order}'")

    if ir.playback:
        if ir.domain != "vod":
            errs.append("playback 仅影视(vod)支持")
        for k, v in ir.playback.items():
            if k not in PLAYBACK_FIELDS:
                errs.append(f"未知 playback 字段 '{k}'")
            elif not isinstance(v, int):
                errs.append(f"playback.{k} 必须是整数")
    return errs


def _validate_node(node: Node, domain: str, errs: list[str]) -> None:
    if isinstance(node, And):
        for x in node.items:
            _validate_node(x, domain, errs)
    elif isinstance(node, Or):
        for x in node.items:
            _validate_node(x, domain, errs)
    elif isinstance(node, Not):
        _validate_node(node.item, domain, errs)
    elif isinstance(node, Leaf):
        _validate_leaf(node, domain, errs)


def _validate_leaf(leaf: Leaf, domain: str, errs: list[str]) -> None:
    spec = get_field(leaf.field)
    if spec is None:
        errs.append(f"未知字段 '{leaf.field}'")
        return
    if domain not in spec.domains:
        errs.append(f"字段 '{leaf.field}' 在 domain '{domain}' 不存在")
        return

    if spec.kind is Kind.EXACT:
        if leaf.range is not None:
            errs.append(f"精确字段 '{leaf.field}' 不能用 range")
        has_v = leaf.value is not None
        has_vs = leaf.values is not None
        if has_v == has_vs:
            errs.append(f"精确字段 '{leaf.field}' 必须且只能给 value 或 values 之一")
        if has_vs:
            if not isinstance(leaf.values, list) or not leaf.values:
                errs.append(f"'{leaf.field}'.values 必须是非空数组")
            if leaf.op not in ("and", "or"):
                errs.append(f"'{leaf.field}'.op 必须是 and/or")
    elif spec.kind is Kind.STATUS:
        if leaf.value not in (0, 1):
            errs.append(f"状态字段 '{leaf.field}'.value 必须是 0 或 1")
        if leaf.values is not None or leaf.range is not None:
            errs.append(f"状态字段 '{leaf.field}' 只能用 value")
    elif spec.kind is Kind.RANGE:
        if not isinstance(leaf.range, dict) or "from" not in leaf.range or "to" not in leaf.range:
            errs.append(f"范围字段 '{leaf.field}' 必须给 range:{{from,to}}")
        else:
            _validate_range_values(spec, leaf.range, errs)
        if leaf.value is not None or leaf.values is not None:
            errs.append(f"范围字段 '{leaf.field}' 只能用 range")


def _validate_range_values(spec, rng: dict, errs: list[str]) -> None:
    fmt = spec.nested.value_fmt if spec.nested else "string"
    for bound in ("from", "to"):
        v = rng[bound]
        if v == "*":
            continue
        if fmt == "yyyyMMdd":
            if not (isinstance(v, str) and len(v) == 8 and v.isdigit()):
                errs.append(f"'{spec.canonical}'.{bound} 必须是 yyyyMMdd 或 '*'，收到 {v!r}")
        elif fmt == "int":
            if not isinstance(v, int):
                errs.append(f"'{spec.canonical}'.{bound} 必须是整数或 '*'，收到 {v!r}")
        elif fmt == "number":
            if not isinstance(v, (int, float)):
                errs.append(f"'{spec.canonical}'.{bound} 必须是数字或 '*'，收到 {v!r}")


# ---------------------------------------------------------------------------
# 便捷构造器（测试与手写用）
# ---------------------------------------------------------------------------
def leaf(field_: str, value=None, values=None, op="or", range=None) -> Leaf:
    return Leaf(field=field_, value=value, values=values, op=op, range=range)
