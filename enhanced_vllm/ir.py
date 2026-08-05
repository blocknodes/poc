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
      "playback": {"series": 2, "video_index": 3}     # 可选，仅 vod
    }

Node 四种形态::

    {"and": [Node, ...]}
    {"or":  [Node, ...]}
    {"not": Node}
    {"field": "actor", "value": "刘德华"}                    # 精确单值
    {"field": "actor", "values": ["刘德华","吴京"], "op": "and"}  # 精确多值
    {"field": "fee",   "value": 0}                          # 状态
    {"field": "release_year", "range": {"from": "20200101", "to": "*"}}  # 范围

非 IR 域（audio / device）不经过本模块，走独立 slot-fill schema。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from registry import (
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


def parse_ir(obj: dict, domain_hint: str | None = None) -> IR:
    """解析 IR JSON 为内部 IR 对象。

    支持两种格式：
      * vod 格式：含 domain/action/playback/sort(数组)
      * educ 格式（0729-v1 tool_schema 对齐）：无 domain/action/playback，
        sort 为对象格式，字段用 tool_schema 名（is_fee/age_range/release_year）

    Args:
        domain_hint: 当 IR 不含 domain 字段时使用（educ 新格式不输出 domain）。
    """
    if not isinstance(obj, dict):
        raise IRError("IR 根必须是对象")

    domain = obj.get("domain")
    if domain is None:
        # educ 新格式不包含 domain；用 hint 补全
        domain = domain_hint or "educ"
    if domain not in ("vod", "educ"):
        raise IRError(f"domain 必须是 vod 或 educ，收到: {domain!r}")

    if "query" not in obj:
        raise IRError("IR 缺少 query")

    action = obj.get("action", "search")
    if action not in ("search", "play"):
        raise IRError(f"action 必须是 search 或 play，收到: {action!r}")

    # ---- sort 解析：兼容数组格式（vod）和对象格式（educ） ----
    sort_raw = obj.get("sort")
    sort: list[SortItem] = []
    if sort_raw:
        if isinstance(sort_raw, list):
            # vod 格式：[{"key":"rate","order":"desc"}]
            for s in sort_raw:
                if not isinstance(s, dict) or "key" not in s or "order" not in s:
                    raise IRError(f"sort 项必须是含 key/order 的对象，收到: {s!r}")
                sort.append(SortItem(s["key"], s["order"]))
        elif isinstance(sort_raw, dict):
            # educ 对象格式：{"rate":{"order":"desc"}, "hot":{"order":"asc"}}
            for key, val in sort_raw.items():
                if isinstance(val, dict) and "order" in val:
                    sort.append(SortItem(key, val["order"]))
                else:
                    raise IRError(f"sort.{key} 必须含 order 字段，收到: {val!r}")
        else:
            raise IRError(f"sort 必须是数组或对象，收到: {type(sort_raw).__name__}")

    # ---- playback（仅 vod）----
    playback_raw = obj.get("playback") or {}
    if not isinstance(playback_raw, dict):
        raise IRError(f"playback 必须是对象，收到: {type(playback_raw).__name__}")
    playback = dict(playback_raw)

    # ---- 解析 query 树，educ 格式需做字段名归一 ----
    query_obj = obj["query"]
    if domain == "educ":
        query_obj = _normalize_educ_query_fields(query_obj)

    return IR(
        domain=domain,
        query=parse_node(query_obj),
        action=action,
        sort=sort,
        playback=playback,
    )


# ---------------------------------------------------------------------------
# educ 字段名归一：tool_schema 名 → canonical 名（供内部 IR 对象使用）
# ---------------------------------------------------------------------------
_EDUC_FIELD_TO_CANONICAL = {
    "is_fee": "fee",
    "age_range": "age",
    # release_year 在 educ canonical 和 tool_schema 名相同，无需映射
}


def _normalize_educ_query_fields(node: Any) -> Any:
    """将 educ tool_schema 字段名转回 canonical，并将扁平 range 格式转为 nested range。

    educ tool_schema 范围格式：{"field":"age_range","from":"3","to":"6"}
    canonical range 格式：    {"field":"age","range":{"from":"3","to":"6"}}
    """
    if not isinstance(node, dict):
        return node
    # 递归处理 and/or/not
    if "and" in node:
        return {"and": [_normalize_educ_query_fields(x) for x in node["and"]]}
    if "or" in node:
        return {"or": [_normalize_educ_query_fields(x) for x in node["or"]]}
    if "not" in node:
        return {"not": _normalize_educ_query_fields(node["not"])}
    # 叶子节点
    if "field" in node:
        field_name = node["field"]
        canonical = _EDUC_FIELD_TO_CANONICAL.get(field_name, field_name)
        result = dict(node)
        result["field"] = canonical

        # educ 范围字段扁平格式 {"field":"age_range","from":"3","to":"6"}
        # → canonical {"field":"age","range":{"from":3,"to":6}}
        if "from" in result or "to" in result:
            range_obj = {}
            if "from" in result:
                range_obj["from"] = result.pop("from")
            if "to" in result:
                range_obj["to"] = result.pop("to")
            # age 字段：值应为整数（educ tool_schema 传字符串，需转换）
            if canonical == "age":
                for k in ("from", "to"):
                    if k in range_obj and range_obj[k] != "*":
                        try:
                            range_obj[k] = int(range_obj[k])
                        except (ValueError, TypeError):
                            pass  # 留给 validate_ir 报错
            result["range"] = range_obj
        elif "range" in result and isinstance(result["range"], dict):
            # 模型已输出 canonical range 格式 {"field":"age","range":{"from":"0","to":"3"}}
            # 但 from/to 可能是字符串，需转为 int
            if canonical == "age":
                for k in ("from", "to"):
                    if k in result["range"] and result["range"][k] != "*":
                        try:
                            result["range"][k] = int(result["range"][k])
                        except (ValueError, TypeError):
                            pass
        return result
    return node


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
