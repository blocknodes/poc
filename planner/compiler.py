"""Compiler —— 把一份 domain 无关的 IR 编译成具体工具的 parameters。

两个后端：
  * nested backend  -> *_search   (vod_search / educ_search)，嵌套 QueryNode/BoolNode/LeafNode JSON
  * flat   backend  -> *_slow_search (vod_slow_search_data_search / educ_slow_search_data_search)，
                        扁平的 field->字符串 mini-DSL

flat 后端表达力弱于 nested（字段间只能隐式 AND，无跨字段 OR/嵌套 NOT），
因此提供 can_compile_flat() 做“可编译性检查”。harness 在目标是慢链路却不可无损
编译时，应回退到 *_search，而不是静默丢逻辑。
"""
from __future__ import annotations

from typing import Any, Optional

from .ir import IR, And, Leaf, Node, Not, Or
from .registry import (
    FlatMode,
    Kind,
    SORT_REGISTRY,
    get_field,
)


class CompileError(ValueError):
    """IR 无法编译到目标后端（通常是 flat 表达力不足）。"""


# ===========================================================================
# nested backend  ->  *_search
# ===========================================================================
def compile_nested(ir: IR) -> dict[str, Any]:
    params: dict[str, Any] = {"query": _nested_node(ir.query, ir.domain)}

    if ir.action == "play":
        # action 作为一个叶子并入 query（schema 的 PlaybackControlFieldCondition）
        action_leaf = {"field": "action", "value": "play"}
        params["query"] = {"and": [action_leaf, params["query"]]}

    if ir.sort:
        sort_obj: dict[str, Any] = {}
        for s in ir.sort:
            spec = SORT_REGISTRY[s.key]
            sort_obj[spec.nested_key] = {"order": s.order}
        params["sort"] = sort_obj

    # 播放控制字段合并进 query 的顶层 AND
    if ir.playback:
        pb_leaves = [{"field": k, "value": v} for k, v in ir.playback.items()]
        params["query"] = _merge_and(params["query"], pb_leaves)

    return params


def _nested_node(node: Node, domain: str) -> dict[str, Any]:
    if isinstance(node, And):
        return {"and": [_nested_node(x, domain) for x in node.items]}
    if isinstance(node, Or):
        return {"or": [_nested_node(x, domain) for x in node.items]}
    if isinstance(node, Not):
        return {"not": _nested_node(node.item, domain)}
    if isinstance(node, Leaf):
        return _nested_leaf(node, domain)
    raise CompileError(f"未知节点类型 {type(node).__name__}")


def _nested_leaf(leaf: Leaf, domain: str) -> dict[str, Any]:
    spec = get_field(leaf.field)
    if spec is None or spec.nested_name(domain) is None:
        raise CompileError(f"字段 '{leaf.field}' 无法编译到 {domain} 的 *_search")
    name = spec.nested_name(domain)

    if spec.kind is Kind.EXACT:
        if leaf.values is not None:
            out = {"field": name, "values": list(leaf.values)}
            if leaf.op and leaf.op != "or":
                out["operator"] = leaf.op
            return out
        return {"field": name, "value": leaf.value}

    if spec.kind is Kind.STATUS:
        return {"field": name, "value": int(leaf.value)}

    if spec.kind is Kind.RANGE:
        rng = leaf.range or {}
        return {"field": name, "from": rng.get("from"), "to": rng.get("to")}

    raise CompileError(f"未知字段类型 {spec.kind}")


def _merge_and(query: dict[str, Any], extra_leaves: list[dict]) -> dict[str, Any]:
    if "and" in query:
        return {"and": query["and"] + extra_leaves}
    return {"and": [query] + extra_leaves}


# ===========================================================================
# flat backend  ->  *_slow_search
# ===========================================================================
def can_compile_flat(ir: IR) -> tuple[bool, Optional[str]]:
    """检查 IR 是否可无损编译到 flat（慢链路）后端。返回 (ok, reason)。"""
    try:
        _flatten(ir.query, ir.domain, dry_run=True)
    except CompileError as e:
        return False, str(e)
    # sort / playback 可用性
    for s in ir.sort:
        spec = SORT_REGISTRY.get(s.key)
        if not spec or spec.flat_field_by_domain.get(ir.domain) is None:
            return False, f"排序键 '{s.key}' 在 {ir.domain} 慢链路无对应字段"
    if ir.playback:
        return False, "慢链路不支持播放控制(playback)字段"
    return True, None


def compile_flat(ir: IR) -> dict[str, Any]:
    ok, reason = can_compile_flat(ir)
    if not ok:
        raise CompileError(reason or "无法编译到 flat 后端")
    params: dict[str, str] = {}
    _flatten(ir.query, ir.domain, out=params, dry_run=False)

    # sort 编码进字段值
    for s in ir.sort:
        spec = SORT_REGISTRY[s.key]
        fname = spec.flat_field_by_domain[ir.domain]
        # 若同字段已被范围占用（如 date 既做区间又做排序），排序让位/冲突
        if fname in params:
            raise CompileError(f"排序键 '{s.key}' 与已有条件 '{fname}' 冲突")
        params[fname] = s.order
    return params


def compile_flat_best_effort(ir: IR) -> dict[str, Any]:
    """Best-effort 编译到 flat 慢链路：能表达的字段尽量表达，其余交还语义层。

    与严格版 compile_flat 的区别——绝不抛异常、绝不回退：
      * 无 flat 落地名的字段（title/role/company/writer 等）→ 跳过（原始 query 已含该信息）；
      * 跨字段 OR / 嵌套结构 → 尽量摊平其中的单字段叶子，无法无损表达的布尔关系交还语义层；
      * 同一 flat 字段被多个条件占用 → 保留第一个。
    这样路由判定的慢链路工具得以保持，不会因表达力不足被改写成精确检索。
    """
    params: dict[str, str] = {}
    _collect_flat_leaves(ir.query, ir.domain, params, negate=False)

    for s in ir.sort:
        spec = SORT_REGISTRY.get(s.key)
        if not spec:
            continue
        fname = spec.flat_field_by_domain.get(ir.domain)
        if fname and fname not in params:
            params[fname] = s.order
    return params


def _collect_flat_leaves(node: Node, domain: str, out: dict, negate: bool) -> None:
    """遍历 IR 树，尽力把可表达的叶子写入 flat 参数；不可表达的静默跳过（交还语义层）。"""
    if isinstance(node, (And, Or)):
        # best-effort：无论 AND / OR，都摊平其中的单字段叶子（布尔关系交还语义层）
        for child in node.items:
            _collect_flat_leaves(child, domain, out, negate)
        return
    if isinstance(node, Not):
        _collect_flat_leaves(node.item, domain, out, not negate)
        return
    if isinstance(node, Leaf):
        try:
            _emit_flat_leaf(node, domain, out, dry_run=False, negate=negate)
        except CompileError:
            # 无 flat 落地名 / 多值枚举 / 范围取反 等无法无损表达 → 跳过
            pass
        return


def _flatten(node: Node, domain: str, out: Optional[dict] = None, *, dry_run: bool) -> None:
    """把 query 树摊平成 field->string。只接受 单叶子 或 顶层 AND(叶子...)。"""
    if isinstance(node, And):
        for child in node.items:
            _flatten_leaf_or_not(child, domain, out, dry_run)
    else:
        _flatten_leaf_or_not(node, domain, out, dry_run)


def _flatten_leaf_or_not(node: Node, domain: str, out: Optional[dict], dry_run: bool) -> None:
    if isinstance(node, Or):
        raise CompileError("慢链路(flat)不支持跨字段 OR，请回退到 *_search")
    if isinstance(node, And):
        raise CompileError("慢链路(flat)不支持嵌套 AND，请回退到 *_search")
    if isinstance(node, Not):
        if not isinstance(node.item, Leaf):
            raise CompileError("慢链路(flat)的 NOT 只能作用于单个字段条件")
        _emit_flat_leaf(node.item, domain, out, dry_run, negate=True)
        return
    if isinstance(node, Leaf):
        _emit_flat_leaf(node, domain, out, dry_run, negate=False)
        return
    raise CompileError(f"未知节点类型 {type(node).__name__}")


def _emit_flat_leaf(leaf: Leaf, domain: str, out: Optional[dict], dry_run: bool, negate: bool) -> None:
    spec = get_field(leaf.field)
    if spec is None or spec.flat is None or spec.flat_name(domain) is None:
        raise CompileError(f"字段 '{leaf.field}' 在 {domain} 慢链路无对应参数")
    fname = spec.flat_name(domain)
    mode = spec.flat.mode

    # 构造字符串值
    if spec.kind is Kind.RANGE:
        if negate:
            raise CompileError(f"范围字段 '{leaf.field}' 不支持 NOT")
        rng = leaf.range or {}
        val = f"{rng.get('from')} TO {rng.get('to')}"
    elif spec.kind is Kind.STATUS:
        if negate:
            raise CompileError(f"状态字段 '{leaf.field}' 不支持 NOT")
        val = str(int(leaf.value))
    else:  # EXACT
        if mode is FlatMode.ENUM:
            if leaf.values and len(leaf.values) > 1:
                raise CompileError(f"枚举型慢链路字段 '{fname}' 不支持多值")
            base = leaf.values[0] if leaf.values else leaf.value
            val = f"NOT {base}" if negate else str(base)
        else:  # LOGIC
            if leaf.values:
                joiner = " AND " if leaf.op == "and" else " OR "
                base = joiner.join(leaf.values)
            else:
                base = str(leaf.value)
            val = f"NOT {base}" if negate else base

    if dry_run:
        return
    assert out is not None
    if fname in out:
        raise CompileError(f"慢链路字段 '{fname}' 被多个条件占用，无法合并")
    out[fname] = val


# ===========================================================================
# 统一入口：按目标工具名选择后端（含可编译性回退提示）
# ===========================================================================
_NESTED_TOOLS = {"vod_search", "educ_search"}
_FLAT_TOOLS = {"vod_slow_search_data_search", "educ_slow_search_data_search"}

# 当 flat 不可编译时的回退目标
_FLAT_TO_NESTED_FALLBACK = {
    "vod_slow_search_data_search": "vod_search",
    "educ_slow_search_data_search": "educ_search",
}


def compile_ir(ir: IR, tool_name: str) -> dict[str, Any]:
    if tool_name in _NESTED_TOOLS:
        return compile_nested(ir)
    if tool_name in _FLAT_TOOLS:
        return compile_flat(ir)
    raise CompileError(f"工具 '{tool_name}' 不由 IR 编译器负责（应走独立 schema）")


def compile_with_fallback(ir: IR, tool_name: str) -> tuple[str, dict[str, Any]]:
    """返回 (实际使用的工具名, params)。

    慢链路(*_slow_search)是「语义检索」后端：真实接口除了可选的结构化过滤字段外，
    还会收到用户原始 query 文本并自行做语义理解（见 api.md / 工具定义 required:[]）。
    因此路由判定为慢链路时，应当保持慢链路工具，用 best-effort 方式把「能表达的」
    过滤字段编译出来，把无法用扁平 mini-DSL 无损表达的部分（无 flat 落地名的字段、
    跨字段 OR / 嵌套结构）交还给原始 query 语义层——而不是回退到精确检索 *_search
    （对模糊 query 用精确检索反而结果更差）。
    """
    if tool_name in _FLAT_TOOLS:
        return tool_name, compile_flat_best_effort(ir)
    return tool_name, compile_ir(ir, tool_name)
