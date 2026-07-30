"""Compiler —— 把 IR / slot-fill 结果编译成具体工具的 parameters。

三大类编译路径：
  1. 影视/少儿检索类工具：IR → nested/flat backend
  2. 有声域：slot-fill dict → audio_search / audio_chat_qa 参数
  3. 设备域：slot-fill dict → *_control 参数

影视/少儿的 nested 后端：
  * vod_search_all  -> 24 精确字段全量版
  * vod_search      -> 10 精确字段精简版（IR 字段全在 VOD_SEARCH_FIELDS 内时选它）
  * vod_relate_search -> 4 精确字段（title/actor/director/category），布尔 DSL
  * educ_search     -> 少儿全量

flat 后端：
  * vod_slow_search -> 新版只传 query（用户原文），无需编译器产出结构化 flat 参数
                       但仍保留 compile_flat_best_effort 供训练/评测对比

flat 后端表达力弱于 nested（字段间只能隐式 AND，无跨字段 OR/嵌套 NOT），
因此提供 can_compile_flat() 做"可编译性检查"。

vod_search vs vod_search_all 判定逻辑：
  生成 IR 后，提取所有精确字段。若全部 ⊆ VOD_SEARCH_FIELDS（10 个），则用 vod_search；
  否则用 vod_search_all。二者结构完全一致，仅字段枚举宽度不同。
"""
from __future__ import annotations

from typing import Any, Optional

from .ir import IR, And, Leaf, Node, Not, Or
from .registry import (
    AUDIO_PLAY_MODES,
    AUDIO_SCREEN_MODES,
    AUDIO_TOOLS,
    DEVICE_TOOLS,
    FlatMode,
    Kind,
    SORT_REGISTRY,
    VOD_RELATE_FIELDS,
    VOD_SEARCH_FIELDS,
    get_field,
)


# ===========================================================================
# Experience Bank 配置
# ===========================================================================
from dataclasses import dataclass


@dataclass
class ExperienceBankConfig:
    """Experience Bank 各子模块开关。

    按需开启/关闭 compiler 层的各类后处理规则。
    prompt 层规则的开关在 prompts.py 中通过 use_experience_bank 参数控制。
    """
    # 总开关
    enabled: bool = True

    # --- Compiler 层细粒度开关 ---
    # action 动词覆盖（强/弱动词判定 play/search）
    action_override: bool = True
    # 字段名修正（好莱坞→company, 湖南卫视→channel, TVB→company）
    field_remap: bool = True
    # 值归一化（人名间隔符、vender大小写、tag大小写、hostess大小写）
    value_normalize: bool = True
    # 结构简化（单元素and解包、not is_fee→is_fee:0）
    structure_simplify: bool = True
    # query-text 感知补全（影片→category、韩剧拆分、节目→综艺）
    query_text_complete: bool = True
    # sort 精细控制（好看的不触发、有分数不加sort、最近→sort:new）
    sort_control: bool = True
    # title 处理（去后缀、拆数字+series）
    title_process: bool = True
    # 时间转换（去年/今年/前年→绝对年份）
    time_convert: bool = True
    # 名称映射（奖项归一、tag去后缀、company归一）
    name_normalize: bool = True
    # operator 始终输出
    operator_always: bool = True
    # rate 范围归一（*→10.0）
    rate_normalize: bool = True


# 全局默认配置（模块级单例，可通过 set_experience_bank_config 替换）
_EB_CONFIG = ExperienceBankConfig()


def get_experience_bank_config() -> ExperienceBankConfig:
    """获取当前 Experience Bank 配置。"""
    return _EB_CONFIG


def set_experience_bank_config(config: ExperienceBankConfig) -> None:
    """替换全局 Experience Bank 配置。"""
    global _EB_CONFIG
    _EB_CONFIG = config


class CompileError(ValueError):
    """IR 无法编译到目标后端（通常是 flat 表达力不足）。"""


# ===========================================================================
# 工具选择：vod_search vs vod_search_all
# ===========================================================================
def _collect_exact_fields(node: Node) -> set[str]:
    """递归收集 IR query 树中所有精确匹配字段名。"""
    fields: set[str] = set()
    if isinstance(node, And):
        for child in node.items:
            fields |= _collect_exact_fields(child)
    elif isinstance(node, Or):
        for child in node.items:
            fields |= _collect_exact_fields(child)
    elif isinstance(node, Not):
        fields |= _collect_exact_fields(node.item)
    elif isinstance(node, Leaf):
        spec = get_field(node.field)
        if spec and spec.kind is Kind.EXACT:
            fields.add(node.field)
    return fields


def select_vod_search_tool(ir: IR) -> str:
    """根据 IR 中使用的精确字段集，选择 vod_search 或 vod_search_all。

    逻辑：若 IR 中出现的所有精确字段都在 VOD_SEARCH_FIELDS 内，用精简版 vod_search；
    否则用全量版 vod_search_all。
    """
    exact_fields = _collect_exact_fields(ir.query)
    if exact_fields <= VOD_SEARCH_FIELDS:
        return "vod_search"
    return "vod_search_all"


def check_relate_fields(ir: IR) -> bool:
    """检查 IR 是否适合 vod_relate_search（仅支持 title/actor/director/category，无范围/状态/sort/playback）。"""
    exact_fields = _collect_exact_fields(ir.query)
    # relate 只支持精确字段，且必须都在 VOD_RELATE_FIELDS 内
    if not exact_fields <= VOD_RELATE_FIELDS:
        return False
    # 不能有状态/范围字段
    if _has_non_exact_leaf(ir.query):
        return False
    # 不能有 sort / playback
    if ir.sort or ir.playback:
        return False
    return True


def _has_non_exact_leaf(node: Node) -> bool:
    """检查是否存在非精确匹配的叶子节点。"""
    if isinstance(node, And):
        return any(_has_non_exact_leaf(c) for c in node.items)
    if isinstance(node, Or):
        return any(_has_non_exact_leaf(c) for c in node.items)
    if isinstance(node, Not):
        return _has_non_exact_leaf(node.item)
    if isinstance(node, Leaf):
        spec = get_field(node.field)
        return spec is not None and spec.kind is not Kind.EXACT
    return False


# ===========================================================================
# nested backend  ->  vod_search / vod_search_all / vod_relate_search
# ===========================================================================
def compile_nested(ir: IR) -> dict[str, Any]:
    """编译成 vod_search / vod_search_all 格式（含 action/retext/query/sort）。"""
    params: dict[str, Any] = {
        "action": ir.action,  # 顶层 required 字段，search 或 play
        "query": _nested_node(ir.query, ir.domain),
    }

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


def compile_relate(ir: IR) -> dict[str, Any]:
    """编译成 vod_relate_search 格式（仅 query，无 action/sort/playback）。"""
    return {
        "query": _nested_node(ir.query, ir.domain),
    }


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
        raise CompileError(f"字段 '{leaf.field}' 无法编译到 {domain} 的 nested 后端")
    name = spec.nested_name(domain)

    if spec.kind is Kind.EXACT:
        if leaf.values is not None:
            out: dict[str, Any] = {"field": name, "values": list(leaf.values)}
            # 始终输出 operator（标注格式要求显式声明 and/or）
            out["operator"] = leaf.op if leaf.op else "or"
            return out
        return {"field": name, "value": leaf.value}

    if spec.kind is Kind.STATUS:
        return {"field": name, "value": int(leaf.value)}

    if spec.kind is Kind.RANGE:
        rng = leaf.range or {}
        from_val = rng.get("from")
        to_val = rng.get("to")
        # rate 范围规范化：开区间 "*" → 10.0（满分）; from 确保是 number
        if leaf.field == "rate":
            if from_val is not None and from_val != "*":
                from_val = float(from_val)
            else:
                from_val = 0.0
            if to_val is None or to_val == "*":
                to_val = 10.0
            else:
                to_val = float(to_val)
        return {"field": name, "from": from_val, "to": to_val}

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
    """遍历 IR 树，尽力把可表达的叶子写入 flat 参数；不可表达的静默跳过。"""
    if isinstance(node, (And, Or)):
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
# Audio 编译（slot-fill → 最终参数）
# ===========================================================================
def compile_audio(slot_fill: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """把有声域的 slot-fill 结果编译成工具名 + 参数。

    输入: {"tool": "audio_search", "query": "...", "play_mode": "...", "screen_mode": "..."}
    输出: (tool_name, parameters)
    """
    tool = slot_fill.get("tool", "audio_search")
    if tool not in AUDIO_TOOLS:
        tool = "audio_search"

    params: dict[str, Any] = {}
    query = slot_fill.get("query", "")
    if query:
        params["query"] = query

    play_mode = slot_fill.get("play_mode", "search")
    if play_mode in AUDIO_PLAY_MODES:
        params["play_mode"] = play_mode
    else:
        params["play_mode"] = "search"

    screen_mode = slot_fill.get("screen_mode")
    if screen_mode in AUDIO_SCREEN_MODES:
        params["screen_mode"] = screen_mode

    return tool, params


# ===========================================================================
# Device 编译（slot-fill → 最终参数）
# ===========================================================================
def compile_device(slot_fill: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """把设备域的 slot-fill 结果编译成工具名 + 参数。

    输入: {"tool": "volume_control", "operation": "提高", "object": "音量", "value": "10"}
    输出: (tool_name, parameters)
    """
    tool = slot_fill.get("tool", "system_settings_control")
    if tool not in DEVICE_TOOLS:
        tool = "system_settings_control"

    params: dict[str, Any] = {}
    operation = slot_fill.get("operation")
    if operation:
        params["operation"] = operation
    obj = slot_fill.get("object")
    if obj:
        params["object"] = obj
    value = slot_fill.get("value")
    if value:
        params["value"] = value

    return tool, params


# ===========================================================================
# 统一入口：按目标工具名选择后端（含可编译性回退提示）
# ===========================================================================
# vod_search 和 vod_search_all 是同一意图，IR 生成后再选择
_NESTED_TOOLS = {"vod_search", "vod_search_all", "educ_search",
                 "vod_relate_search", "educ_relate_recommend"}
_FLAT_TOOLS = {"vod_slow_search_data_search", "educ_slow_search_data_search"}

# 当 flat 不可编译时的回退目标（但新版 slow_search 只传 query，一般不回退了）
_FLAT_TO_NESTED_FALLBACK = {
    "vod_slow_search_data_search": "vod_search",
    "educ_slow_search_data_search": "educ_search",
}


def compile_ir(ir: IR, tool_name: str) -> dict[str, Any]:
    if tool_name == "vod_relate_search":
        return compile_relate(ir)
    if tool_name in _NESTED_TOOLS:
        return compile_nested(ir)
    if tool_name in _FLAT_TOOLS:
        return compile_flat(ir)
    raise CompileError(f"工具 '{tool_name}' 不由 IR 编译器负责（应走独立 schema）")


def compile_with_fallback(ir: IR, tool_name: str, *, retext: str = "",
                          intent: str = "") -> tuple[str, dict[str, Any]]:
    """返回 (实际使用的工具名, params)。

    核心逻辑：
      1. 若路由为 vod_search / vod_search_all，生成 nested params，
         再根据字段子集选择实际工具名；
      2. 若路由为 vod_relate_search，编译 relate 格式；
      3. 若路由为慢链路（vod_slow_search_data_search），新版只传 query 原文；
      4. retext 为用户原始文本，vod_search/vod_search_all 需要填入。
      5. 最终参数经过领域归一化后处理（tag/category 同义词映射）。
      6. intent 为路由阶段判定的意图，用于 action 覆盖。

    Experience Bank compiler层规则受全局 ExperienceBankConfig 控制。
    """
    cfg = _EB_CONFIG

    # 慢链路：新版 vod_slow_search 只需传 query（用户原文），服务端自行解析
    if tool_name in _FLAT_TOOLS:
        if retext:
            return tool_name, {"query": retext}
        # 无 retext 时 best-effort 编译（向后兼容训练/评测）
        return tool_name, compile_flat_best_effort(ir)

    # relate 工具
    if tool_name == "vod_relate_search":
        params = compile_relate(ir)
        if cfg.enabled:
            _domain_normalize_params(params, cfg)
            if retext:
                _post_process_with_query_text(params, retext, cfg)
            if cfg.structure_simplify and "query" in params:
                params["query"] = _simplify_query(params["query"])
        return tool_name, params

    # nested 工具（vod_search / vod_search_all 统一入口）
    if tool_name in ("vod_search", "vod_search_all"):
        params = compile_nested(ir)
        # 根据字段子集选择实际工具
        actual_tool = select_vod_search_tool(ir)
        # 填入 retext
        if retext:
            params["retext"] = retext
        if cfg.enabled:
            _domain_normalize_params(params, cfg)
            _post_process_with_query_text(params, retext, cfg)
            if cfg.action_override:
                _override_action_from_intent(params, intent, query_text=retext)
            if cfg.structure_simplify and "query" in params:
                params["query"] = _simplify_query(params["query"])
        return actual_tool, params

    if tool_name in _NESTED_TOOLS:
        params = compile_nested(ir)
        if retext:
            params["retext"] = retext
        if cfg.enabled:
            _domain_normalize_params(params, cfg)
            _post_process_with_query_text(params, retext, cfg)
            if cfg.action_override:
                _override_action_from_intent(params, intent, query_text=retext)
            if cfg.structure_simplify and "query" in params:
                params["query"] = _simplify_query(params["query"])
        if "query" in params:
            params["query"] = _simplify_query(params["query"])
        return tool_name, params

    raise CompileError(f"工具 '{tool_name}' 不由 IR 编译器负责")


# ===========================================================================
# query-text-aware 后处理 —— 需要原始用户文本才能做的补全/修正
# ===========================================================================
def _post_process_with_query_text(params: dict[str, Any], query_text: str,
                                   cfg: Optional[ExperienceBankConfig] = None) -> None:
    """根据用户原文补全编译结果中缺失的信息。"""
    if cfg is None:
        cfg = _EB_CONFIG
    if not query_text or "query" not in params:
        return

    import re as _re
    from datetime import datetime

    query_node = params["query"]

    # 收集当前已有的字段
    existing_fields = set()
    _collect_existing_fields(query_node, existing_fields)

    # --- 规则1: "影片/大片" → 补 category:电影（如果没有 category）---
    if cfg.query_text_complete and "category" not in existing_fields:
        if _re.search(r'影片|大片', query_text):
            _inject_field(params, "category", "电影")
            existing_fields.add("category")

    # --- 规则1b: "节目/脱口秀节目" → 补 category:综艺 ---
    if cfg.query_text_complete and "category" not in existing_fields:
        if _re.search(r'节目', query_text) and not _re.search(r'纪录片|电影|电视剧', query_text):
            _inject_field(params, "category", "综艺")
            existing_fields.add("category")

    # --- 规则2: "韩剧/美剧/日剧/泰剧/港剧" 拆分 → area + category ---
    if cfg.query_text_complete:
        _DRAMA_AREA = {
            "韩剧": "韩国", "美剧": "美国", "日剧": "日本",
            "泰剧": "泰国", "港剧": "香港", "台剧": "台湾",
        }
        for drama_word, area_val in _DRAMA_AREA.items():
            if drama_word in query_text:
                _replace_field_value(query_node, "tag", drama_word, "area", area_val)
                if "category" not in existing_fields:
                    _inject_field(params, "category", "电视剧")
                    existing_fields.add("category")
                break

    # --- 规则3: sort 精细控制 ---
    if cfg.sort_control:
        # 3a: 有明确分数(N分以上/N到M分) → 保留 rate range，不加 sort
        has_explicit_score = bool(_re.search(r'(\d+\.?\d*)分', query_text))
        # 3b: "高分/评分高" 无明确数字 → sort:rate:desc，删 rate range
        if _re.search(r'高分|评分高|豆瓣评分高', query_text) and not has_explicit_score:
            _remove_field(query_node, "rate")
            if "sort" not in params:
                params["sort"] = {"rate": {"order": "desc"}}
        # 3c: 有明确分数时不需要 sort:rate（已经有 range 约束了）
        if has_explicit_score and "sort" in params:
            sort_obj = params["sort"]
            if isinstance(sort_obj, dict) and "rate" in sort_obj:
                del sort_obj["rate"]
                if not sort_obj:
                    del params["sort"]
        # 3d: "好看的" 单独出现不加 sort（主观修饰不是排序意图）
        if (_re.search(r'好看的', query_text) and not _re.search(r'最好看|好评|口碑', query_text)
                and "sort" in params):
            sort_obj = params.get("sort", {})
            if isinstance(sort_obj, dict) and list(sort_obj.keys()) == ["rate"]:
                del params["sort"]

    # --- 规则4: 多人 op 默认 (顿号/和字连接 → and) ---
    if cfg.value_normalize:
        _fix_multi_value_op(query_node, query_text)

    # --- 规则5: title 去通用后缀 (视频/节目/全集) ---
    if cfg.title_process:
        _strip_title_suffix(query_node)

    # --- 规则6: tag 去通用后缀 (赛/片/类) ---
    if cfg.name_normalize:
        _normalize_tag_suffix(query_node)

    # --- 规则7: company 字段大小写统一(tvb) ---
    if cfg.name_normalize:
        _normalize_company_case(query_node)

    # --- 规则8: 时间相对词 → 绝对年份 ---
    if cfg.time_convert:
        _fix_relative_time(query_node, query_text)

    # --- 规则9: 奖项名归一 ---
    if cfg.name_normalize:
        _normalize_prize_names(query_node)

    # --- 规则10: "最近/新上的" → sort:new:desc，删除 release_time range ---
    if cfg.sort_control:
        if _re.search(r'最近|新上|新出', query_text) and not _re.search(r'\d{4}年', query_text):
            if "release_time" in existing_fields:
                _remove_field(query_node, "release_time")
                if "sort" not in params:
                    params["sort"] = {"new": {"order": "desc"}}
                elif isinstance(params.get("sort"), dict) and "new" not in params["sort"]:
                    params["sort"]["new"] = {"order": "desc"}


# ---------------------------------------------------------------------------
# 规则5: title 去通用后缀
# ---------------------------------------------------------------------------
_TITLE_STRIP_SUFFIXES = ("视频", "全集")  # "节目"不去（可能是title一部分）


def _strip_title_suffix(node: dict[str, Any]) -> None:
    """去除 title 值末尾的通用后缀词。"""
    if not isinstance(node, dict):
        return
    if node.get("field") == "title" and "value" in node and isinstance(node["value"], str):
        v = node["value"]
        for suffix in _TITLE_STRIP_SUFFIXES:
            if v.endswith(suffix) and len(v) > len(suffix):
                node["value"] = v[:-len(suffix)]
                return
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _strip_title_suffix(item)
    if "not" in node and isinstance(node["not"], dict):
        _strip_title_suffix(node["not"])


# ---------------------------------------------------------------------------
# 规则6: tag 去后缀归一
# ---------------------------------------------------------------------------
_TAG_SUFFIX_MAP: dict[str, str] = {
    "辩论赛": "辩论",
    "综艺赛": "综艺",
}


def _normalize_tag_suffix(node: dict[str, Any]) -> None:
    """tag 值映射（去后缀/归一化）。"""
    if not isinstance(node, dict):
        return
    if node.get("field") == "tag" and "value" in node and isinstance(node["value"], str):
        v = node["value"]
        if v in _TAG_SUFFIX_MAP:
            node["value"] = _TAG_SUFFIX_MAP[v]
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _normalize_tag_suffix(item)
    if "not" in node and isinstance(node["not"], dict):
        _normalize_tag_suffix(node["not"])


# ---------------------------------------------------------------------------
# 规则7: company 大小写归一
# ---------------------------------------------------------------------------
_COMPANY_CASE_MAP: dict[str, str] = {
    "TVB": "tvb", "Tvb": "tvb",
    "BBC": "BBC", "HBO": "HBO",
}


def _normalize_company_case(node: dict[str, Any]) -> None:
    """company 字段值大小写归一化。"""
    if not isinstance(node, dict):
        return
    if node.get("field") == "company" and "value" in node and isinstance(node["value"], str):
        v = node["value"]
        if v in _COMPANY_CASE_MAP:
            node["value"] = _COMPANY_CASE_MAP[v]
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _normalize_company_case(item)
    if "not" in node and isinstance(node["not"], dict):
        _normalize_company_case(node["not"])


# ---------------------------------------------------------------------------
# 规则8: 时间相对词→绝对年份
# ---------------------------------------------------------------------------
def _fix_relative_time(node: dict[str, Any], query_text: str) -> None:
    """修正 release_time 中的相对时间引用（去年/今年/前年）。"""
    import re as _re
    from datetime import datetime

    current_year = datetime.now().year  # 2026

    # 确定用户意图的年份
    target_year = None
    if "去年" in query_text:
        target_year = current_year - 1
    elif "今年" in query_text:
        target_year = current_year
    elif "前年" in query_text:
        target_year = current_year - 2

    if target_year is None:
        return

    # 找到 release_time 节点并修正
    _fix_release_time_node(node, target_year)


def _fix_release_time_node(node: dict[str, Any], year: int) -> None:
    """递归找到 release_time 节点并设置正确年份。"""
    if not isinstance(node, dict):
        return
    if node.get("field") in ("release_time", "release_year"):
        expected_from = f"{year}0101"
        expected_to = f"{year}1231"
        if "from" in node:
            node["from"] = expected_from
        if "to" in node:
            node["to"] = expected_to
        return
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _fix_release_time_node(item, year)
    if "not" in node and isinstance(node["not"], dict):
        _fix_release_time_node(node["not"], year)


# ---------------------------------------------------------------------------
# 规则9: 奖项名归一
# ---------------------------------------------------------------------------
_PRIZE_NORMALIZE: dict[str, str] = {
    "香港金像奖": "香港电影金像奖",
    "金像奖": "香港电影金像奖",
    "金鸡奖": "中国电影金鸡奖",
    "金马奖": "台湾电影金马奖",
    "百花奖": "大众电影百花奖",
}


def _normalize_prize_names(node: dict[str, Any]) -> None:
    """奖项名称归一化（简称→全称）。"""
    if not isinstance(node, dict):
        return
    if node.get("field") == "prize" and "value" in node and isinstance(node["value"], str):
        v = node["value"]
        if v in _PRIZE_NORMALIZE:
            node["value"] = _PRIZE_NORMALIZE[v]
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _normalize_prize_names(item)
    if "not" in node and isinstance(node["not"], dict):
        _normalize_prize_names(node["not"])


def _collect_existing_fields(node: dict[str, Any], fields: set) -> None:
    """收集 query 树中所有字段名。"""
    if not isinstance(node, dict):
        return
    if "field" in node:
        fields.add(node["field"])
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _collect_existing_fields(item, fields)
    if "not" in node and isinstance(node["not"], dict):
        _collect_existing_fields(node["not"], fields)


def _inject_field(params: dict[str, Any], field_name: str, value: Any) -> None:
    """向 query 树中注入一个新字段条件（加入顶层 and）。"""
    query = params["query"]
    new_leaf = {"field": field_name, "value": value}
    if "and" in query:
        query["and"].append(new_leaf)
    elif "field" in query:
        # 裸叶子 → 包装成 and
        params["query"] = {"and": [query, new_leaf]}
    else:
        # or / not 节点 → 包装
        params["query"] = {"and": [query, new_leaf]}


def _replace_field_value(node: dict[str, Any], src_field: str, src_value: str,
                         dst_field: str, dst_value: str) -> bool:
    """在 query 树中把 (src_field, src_value) 替换为 (dst_field, dst_value)。"""
    if not isinstance(node, dict):
        return False
    if node.get("field") == src_field and node.get("value") == src_value:
        node["field"] = dst_field
        node["value"] = dst_value
        return True
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                if _replace_field_value(item, src_field, src_value, dst_field, dst_value):
                    return True
    if "not" in node and isinstance(node["not"], dict):
        return _replace_field_value(node["not"], src_field, src_value, dst_field, dst_value)
    return False


def _remove_field(node: dict[str, Any], field_name: str) -> None:
    """从 query 树中删除指定字段的叶子节点。"""
    if not isinstance(node, dict):
        return
    if "and" in node and isinstance(node["and"], list):
        node["and"] = [item for item in node["and"]
                       if not (isinstance(item, dict) and item.get("field") == field_name)]
    elif "or" in node and isinstance(node["or"], list):
        node["or"] = [item for item in node["or"]
                      if not (isinstance(item, dict) and item.get("field") == field_name)]


def _fix_multi_value_op(node: dict[str, Any], query_text: str) -> None:
    """修正多值 operator：中文顿号/和字连接的多人默认 and。"""
    import re as _re
    if not isinstance(node, dict):
        return
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _fix_multi_value_op(item, query_text)
            return
    if "not" in node and isinstance(node["not"], dict):
        _fix_multi_value_op(node["not"], query_text)
        return
    # 叶子：如果有 values 且 operator=or，检查原文是否用了"和/、"连接
    if "values" in node and isinstance(node["values"], list) and len(node["values"]) > 1:
        field = node.get("field", "")
        values = node["values"]
        # 检查原文中是否有 "A和B" 或 "A、B" 的形式
        if field in ("actor", "director", "hostess", "entertainer"):
            # 人名用顿号/和字连接 → and
            for v in values:
                if isinstance(v, str) and (f"{v}和" in query_text or f"、{v}" in query_text
                                           or f"{v}、" in query_text or f"和{v}" in query_text):
                    node["operator"] = "and"
                    return


# ===========================================================================
# action 覆盖 —— 路由 intent 与 IR action 不一致时以路由为准
# ===========================================================================
def _override_action_from_intent(params: dict[str, Any], intent: str,
                                  query_text: str = "") -> None:
    """根据用户原文动词，确定性判定 action=play/search。

    规则（按优先级）：
      1. 强 play 动词（播放/播/放/请播/打开/起播/继续播放/继续看）→ play
      2. 弱 play 动词（看/想看/要看/收看）+ 有 title → play
      3. 弱 play 动词 + 无 title → search
      4. 无 play 动词 → search
    """
    if "action" not in params:
        return

    import re as _re

    # 强 play 动词检测
    # "播放/打开/起播/继续播放/继续看" 直接匹配
    _STRONG_PATTERNS = [
        r'播放', r'起播', r'继续播放', r'继续看', r'打开',
        r'(?:^|帮我|给我|请|来)放',    # 放/帮我放/给我放/请放/来放
        r'(?:^|请)播(?!出)',            # 播/请播（但"播出"不算）
    ]
    has_strong_verb = any(_re.search(p, query_text) for p in _STRONG_PATTERNS)

    if has_strong_verb:
        params["action"] = "play"
        return

    # 弱 play 动词检测
    _WEAK_PATTERNS = [
        r'想看', r'要看', r'收看',
        r'(?:^|我|帮我|来)看',  # 看/我看/帮我看/来看
    ]
    has_weak_verb = any(_re.search(p, query_text) for p in _WEAK_PATTERNS)

    query_node = params.get("query", {})
    if has_weak_verb and _has_title_field(query_node):
        params["action"] = "play"
        return

    # 弱动词无 title / 无动词 → search
    if not has_strong_verb:
        params["action"] = "search"


def _has_title_field(node: dict[str, Any]) -> bool:
    """检查 query 树中是否包含 title 字段。"""
    if not isinstance(node, dict):
        return False
    if node.get("field") == "title":
        return True
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                if isinstance(item, dict) and _has_title_field(item):
                    return True
    if "not" in node and isinstance(node["not"], dict):
        return _has_title_field(node["not"])
    return False


# ===========================================================================
# 结构简化 —— 单元素 and/or 解包为裸节点
# ===========================================================================
def _simplify_query(query: dict[str, Any]) -> dict[str, Any]:
    """递归解包单元素 and/or 容器，使输出与标注格式对齐。

    例：{"and": [{"field":"title","value":"X"}]} → {"field":"title","value":"X"}
    """
    if "and" in query:
        items = query["and"]
        if isinstance(items, list):
            items = [_simplify_query(item) if isinstance(item, dict) else item
                     for item in items]
            if len(items) == 1:
                return items[0]
            return {"and": items}
    if "or" in query:
        items = query["or"]
        if isinstance(items, list):
            items = [_simplify_query(item) if isinstance(item, dict) else item
                     for item in items]
            if len(items) == 1:
                return items[0]
            return {"or": items}
    if "not" in query and isinstance(query["not"], dict):
        query["not"] = _simplify_query(query["not"])
    return query


# ===========================================================================
# title + series 自动拆分 —— "战狼2" → title:"战狼" + series:2
# ===========================================================================
import re as _re

# 中文数字映射
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

_TITLE_SERIES_RE = _re.compile(
    r'^(.+?)'           # 标题主体（非贪婪）
    r'(?:第([一二三四五六七八九十\d]+)[部季]'  # "第N部/季"
    r'|(\d+))$'         # 或末尾纯数字
)


def _split_title_series(params: dict[str, Any]) -> None:
    """检测 query 树中 title 末尾带数字/序号，拆分为 title + playback.series。

    仅在 playback 中无 series 时执行。
    """
    if "playback" in params and "series" in params.get("playback", {}):
        return  # 已有 series，不覆盖
    query = params.get("query")
    if not query:
        return
    _try_split_title_in_node(query, params)


def _try_split_title_in_node(node: dict[str, Any], params: dict[str, Any]) -> bool:
    """在 query 树中找到 title 叶子并尝试拆分。返回是否拆分成功。"""
    if "field" in node and node.get("field") == "title" and "value" in node:
        title_val = str(node["value"])
        m = _TITLE_SERIES_RE.match(title_val)
        if m:
            base_title = m.group(1)
            cn_or_digit = m.group(2) or m.group(3)
            if cn_or_digit:
                # 转为数字
                if cn_or_digit in _CN_NUM:
                    series_num = _CN_NUM[cn_or_digit]
                elif cn_or_digit.isdigit():
                    series_num = int(cn_or_digit)
                else:
                    return False
                node["value"] = base_title
                pb = params.setdefault("playback", {})
                pb.setdefault("series", series_num)
                return True
        return False

    # 递归遍历 and/or/not
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                if isinstance(item, dict) and _try_split_title_in_node(item, params):
                    return True
    if "not" in node and isinstance(node["not"], dict):
        return _try_split_title_in_node(node["not"], params)
    return False


# ===========================================================================
# 领域归一化后处理 —— 编译器输出最终参数前对 tag/category 值做标准化
# 后续可通过 RAG 动态更新映射表
# ===========================================================================

# tag 同义词映射（用户口语 → 系统标签）
# 注意：只映射 **非标准词** → 标准词。标准词表中已有的词（如 抗战/言情/破案/抗日/恋爱体验）
# 不做映射，保持原样输出。
TAG_NORMALIZE: dict[str, str] = {
    "鬼片": "恐怖",
    "恐怖片": "恐怖",
    "打仗": "战争",
    # "抗战" 本身是标准 tag，不映射
    "抗战片": "抗战",       # 去掉"片"后缀
    "军事": "军旅",
    "偶像剧": "偶像",
    # "言情" 本身是标准 tag，不映射
    # "破案" 本身是标准 tag，不映射
    # "恋爱体验" 本身是标准 tag，不映射
    # "抗日" 本身是标准 tag，不映射
    "推理": "悬疑",
    "破案片": "破案",       # 去掉"片"后缀
    "搞笑": "喜剧",
    "幽默": "喜剧",
    "功夫": "武侠",
    "功夫片": "武侠",
    "武打": "武侠",
    "科幻片": "科幻",
    "谍战": "谍战",
    "宫斗": "宫斗",
    "穿越": "穿越",
    "玄幻": "玄幻",
}

# category 同义词映射（用户口语 → 系统分类）
CATEGORY_NORMALIZE: dict[str, str] = {
    "影片": "电影",
    "大片": "电影",
    "片子": "电影",
    "连续剧": "电视剧",
    "剧集": "电视剧",
    "综艺节目": "综艺",
    "动画": "动漫",
    "动画片": "动漫",
}

# 值大小写归一化（公司/供应商名）
VALUE_CASE_NORMALIZE: dict[str, str] = {
    "bbc": "BBC",
    "tvb": "TVB",
    "cctv": "CCTV",
    "hbo": "HBO",
    "netflix": "Netflix",
    "芒果tv": "芒果TV",
    "newtv": "NewTV",
}

# vender_name 特殊归一化（标注可能用不同大小写）
VENDER_NORMALIZE: dict[str, str] = {
    "芒果TV": "芒果tv",    # 标注统一用小写 tv
    "芒果Tv": "芒果tv",
    "NewTV": "newtv",
    "NEWTV": "newtv",
}

# tag 大小写统一（标注用小写）
TAG_CASE_NORMALIZE: dict[str, str] = {
    "DVD版": "dvd版",
    "TV版": "tv版",
    "3D": "3d",
}


def _domain_normalize_params(params: dict[str, Any],
                             cfg: Optional[ExperienceBankConfig] = None) -> None:
    """就地归一化编译后的参数中的 tag/category/value 值，并做 title+series 拆分。"""
    if cfg is None:
        cfg = _EB_CONFIG
    if "query" in params:
        if cfg.value_normalize:
            _normalize_node_values(params["query"])
        if cfg.field_remap:
            _remap_fields(params["query"])
        if cfg.value_normalize:
            _normalize_actor_names(params["query"])
        if cfg.structure_simplify:
            _remove_spurious_fields(params["query"])
            _normalize_fee_not(params["query"])
    if cfg.title_process:
        _split_title_series(params)


# ===========================================================================
# 字段名修正规则 —— 特定(field, value)组合映射到正确字段
# ===========================================================================
_FIELD_REMAP_RULES: list[tuple[str, str, str, str]] = [
    # (原field, 值包含, 目标field, 目标value)   —— 目标value为None表示保持原值
    # 好莱坞不是地区，是制片公司
    ("area", "好莱坞", "company", "好莱坞"),
    # 湖南卫视是频道，不是出品公司
    ("company", "湖南卫视", "channel", "湖南卫视"),
    ("company", "山东卫视", "channel", "山东卫视"),
    ("company", "中央一台", "channel", "中央一台"),
    ("company", "CCTV", "channel", "CCTV"),
    # TVB 是出品公司 (company)，不是 vender_name
    ("vender_name", "TVB", "company", "tvb"),
    ("vender_name", "tvb", "company", "tvb"),
]


def _remap_fields(node: dict[str, Any]) -> None:
    """递归修正 query 树中的字段名映射。"""
    if not isinstance(node, dict):
        return
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _remap_fields(item)
            return
    if "not" in node and isinstance(node["not"], dict):
        _remap_fields(node["not"])
        return
    # 叶子节点：检查是否需要 remap
    field = node.get("field")
    value = node.get("value", "")
    if field and isinstance(value, str):
        for src_f, val_match, dst_f, dst_v in _FIELD_REMAP_RULES:
            if field == src_f and val_match in value:
                node["field"] = dst_f
                if dst_v is not None:
                    node["value"] = dst_v
                return


# ===========================================================================
# 人名间隔符归一 —— 外国人名中间加·
# ===========================================================================
# 常见需要归一的外国演员名（无·→有·）
_ACTOR_NAME_NORMALIZE: dict[str, str] = {
    "汤姆克鲁斯": "汤姆·克鲁斯",
    "马里奥毛瑞尔": "马里奥·毛瑞尔",
    "莱昂纳多迪卡普里奥": "莱昂纳多·迪卡普里奥",
    "布拉德皮特": "布拉德·皮特",
    "安吉丽娜朱莉": "安吉丽娜·朱莉",
    "约翰尼德普": "约翰尼·德普",
    "基努里维斯": "基努·里维斯",
    "杰森斯坦森": "杰森·斯坦森",
    "克里斯蒂安贝尔": "克里斯蒂安·贝尔",
    "罗伯特唐尼": "罗伯特·唐尼",
}


def _normalize_actor_names(node: dict[str, Any]) -> None:
    """递归归一化人名间隔符。"""
    if not isinstance(node, dict):
        return
    for key in ("and", "or"):
        if key in node and isinstance(node[key], list):
            for item in node[key]:
                _normalize_actor_names(item)
            return
    if "not" in node and isinstance(node["not"], dict):
        _normalize_actor_names(node["not"])
        return
    field = node.get("field")
    if field in ("actor", "director", "hostess", "entertainer"):
        if "value" in node and isinstance(node["value"], str):
            v = node["value"]
            if v in _ACTOR_NAME_NORMALIZE:
                node["value"] = _ACTOR_NAME_NORMALIZE[v]
        if "values" in node and isinstance(node["values"], list):
            node["values"] = [_ACTOR_NAME_NORMALIZE.get(v, v) if isinstance(v, str) else v
                              for v in node["values"]]


# ===========================================================================
# 多余字段清理规则
# ===========================================================================
def _remove_spurious_fields(node: dict[str, Any]) -> None:
    """删除编译器认为不该出现的字段。

    规则：
      - is_over 在大部分场景下不应出现（"全集"≠完结、"新剧"≠未完结）
        仅当用户明确说"完结了/已完结/没完结/连载中"时才合理，
        但这个判断由 validate_ir 处理，此处不做清理（风险太大）。
    """
    # 目前不做激进清理，保留为扩展点
    pass


# ===========================================================================
# "不是VIP" 归一化 —— not{is_fee:1} → is_fee:0
# ===========================================================================
def _normalize_fee_not(node: dict[str, Any]) -> None:
    """把 not{is_fee:1} 统一为 is_fee:0（二者语义等价，标注统一用后者）。"""
    if not isinstance(node, dict):
        return
    if "and" in node and isinstance(node["and"], list):
        new_items = []
        for item in node["and"]:
            if (isinstance(item, dict) and "not" in item and
                    isinstance(item["not"], dict) and
                    item["not"].get("field") == "is_fee" and item["not"].get("value") == 1):
                # not{is_fee:1} → is_fee:0
                new_items.append({"field": "is_fee", "value": 0})
            else:
                _normalize_fee_not(item)
                new_items.append(item)
        node["and"] = new_items
    elif "or" in node and isinstance(node["or"], list):
        for item in node["or"]:
            _normalize_fee_not(item)
    elif "not" in node and isinstance(node["not"], dict):
        _normalize_fee_not(node["not"])


def _normalize_node_values(node: Any) -> None:
    """递归归一化 query 树中的 tag/category/value 值。"""
    if not isinstance(node, dict):
        return

    for key in ("and", "or"):
        if key in node:
            for item in node[key]:
                _normalize_node_values(item)
            return

    if "not" in node:
        _normalize_node_values(node["not"])
        return

    # 叶子节点：按 field 类型归一化 value
    field = node.get("field")
    if field == "tag":
        if "value" in node and isinstance(node["value"], str):
            v = node["value"]
            # 先做 tag case 归一化
            v = TAG_CASE_NORMALIZE.get(v, v)
            # 再做 tag 同义词映射
            v = TAG_NORMALIZE.get(v, v)
            node["value"] = v
        if "values" in node and isinstance(node["values"], list):
            node["values"] = [
                TAG_NORMALIZE.get(TAG_CASE_NORMALIZE.get(v, v), TAG_CASE_NORMALIZE.get(v, v))
                if isinstance(v, str) else v
                for v in node["values"]
            ]
    elif field == "category":
        if "value" in node and isinstance(node["value"], str):
            node["value"] = CATEGORY_NORMALIZE.get(node["value"], node["value"])
    elif field == "vender_name":
        # vender_name 归一化
        if "value" in node and isinstance(node["value"], str):
            v = node["value"]
            v = VENDER_NORMALIZE.get(v, v)
            node["value"] = v
    elif field in ("company", "channel"):
        # 大小写归一
        if "value" in node and isinstance(node["value"], str):
            lower = node["value"].lower()
            if lower in VALUE_CASE_NORMALIZE:
                node["value"] = VALUE_CASE_NORMALIZE[lower]
    elif field in ("hostess", "actor", "director", "entertainer"):
        # 人名大小写统一（如 小s → 小S）
        if "value" in node and isinstance(node["value"], str):
            v = node["value"]
            # 小s → 小S (保持标注一致)
            if v == "小s":
                node["value"] = "小S"
        if "values" in node and isinstance(node["values"], list):
            node["values"] = [
                "小S" if v == "小s" else v
                for v in node["values"]
            ]
