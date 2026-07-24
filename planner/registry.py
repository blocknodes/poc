"""Field Registry —— 整个 planner 的单一事实源 (single source of truth)。

它同时驱动三件事：
  1. IR 校验（哪些字段在某个 domain 合法、支持哪些操作）；
  2. 约束解码 grammar 的生成（每个 domain 允许的 field 枚举）；
  3. 编译器双后端（nested-json / flat-string）如何把一个 canonical 字段落地到目标工具。

设计原则：模型只认识 canonical 字段名与统一的 IR 结构；影视/少儿的命名差异
（如 fee vs is_fee、age vs age_range）与序列化差异全部在这里声明、由编译器消化。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Kind(str, Enum):
    EXACT = "exact"    # 精确匹配维度：value / values(+op) / not
    STATUS = "status"  # 状态维度：value ∈ {0,1}
    RANGE = "range"    # 范围维度：{from,to}


class FlatMode(str, Enum):
    """flat backend（慢链路）里某字段字符串值的表达模式。"""
    LOGIC = "logic"    # 支持 "a AND b" / "a OR b" / "NOT a" / 单值
    ENUM = "enum"      # 单值枚举，如 is_fee/gender/category
    RANGE = "range"    # "a TO b" / "* TO b" / "a TO *"


@dataclass(frozen=True)
class NestedTarget:
    """canonical 字段 -> *_search（嵌套 JSON）里的落地名，按 domain 可不同。"""
    # domain -> 目标字段名；None 表示该 domain 不支持
    name_by_domain: dict[str, Optional[str]]
    # release_year 需要 yyyyMMdd；age 是 int；rate 是 0-10 number
    value_fmt: str = "string"  # string | int | number | yyyyMMdd | enum01


@dataclass(frozen=True)
class FlatTarget:
    """canonical 字段 -> *_slow_search（扁平字符串）里的落地名，按 domain 可不同。"""
    name_by_domain: dict[str, Optional[str]]
    mode: FlatMode = FlatMode.LOGIC


@dataclass(frozen=True)
class FieldSpec:
    canonical: str
    kind: Kind
    domains: frozenset[str]                 # 该字段在哪些 domain 存在
    nested: Optional[NestedTarget] = None   # 可编译到 *_search 的信息
    flat: Optional[FlatTarget] = None       # 可编译到 *_slow_search 的信息
    desc: str = ""

    def nested_name(self, domain: str) -> Optional[str]:
        if not self.nested:
            return None
        return self.nested.name_by_domain.get(domain)

    def flat_name(self, domain: str) -> Optional[str]:
        if not self.flat:
            return None
        return self.flat.name_by_domain.get(domain)


VOD = "vod"
EDUC = "educ"
BOTH = frozenset({VOD, EDUC})


def _exact(canonical, domains, *, nested_vod=None, nested_educ=None,
           flat_vod=None, flat_educ=None, flat_mode=FlatMode.LOGIC, desc=""):
    """构造一个精确匹配字段。未显式给出的 nested 名默认等于 canonical（若该 domain 拥有它）。"""
    nv = nested_vod if nested_vod is not None else (canonical if VOD in domains else None)
    ne = nested_educ if nested_educ is not None else (canonical if EDUC in domains else None)
    nested = NestedTarget({VOD: nv, EDUC: ne})
    flat = None
    if flat_vod is not None or flat_educ is not None:
        flat = FlatTarget({VOD: flat_vod, EDUC: flat_educ}, flat_mode)
    return FieldSpec(canonical, Kind.EXACT, frozenset(domains),
                     nested=nested, flat=flat, desc=desc)


# ---------------------------------------------------------------------------
# 字段登记表
# ---------------------------------------------------------------------------
_FIELDS: list[FieldSpec] = [
    # ---- 两域共有的精确维度 ----
    _exact("title", BOTH, desc="媒资作品标题（开放型）。作品本身的名字，如西游记；与 role 区分。"),
    _exact("role", BOTH, desc="角色/人物，如孙悟空、路飞；与 title 区分。"),
    _exact("prize", BOTH, desc="奖项整体，如金鸡奖。"),
    _exact("sub_prize", BOTH, desc="子奖项，如最佳男配角，与 prize 配合。"),
    _exact("language", BOTH, flat_vod=None, flat_educ="language", desc="语种/版本词，如中文、英语、韩版。"),
    _exact("gender", BOTH, flat_educ="gender", flat_mode=FlatMode.ENUM, desc="受众性别。"),
    _exact("company", BOTH, desc="出品公司。"),

    # ---- 影视专有精确维度 ----
    _exact("actor", {VOD}, flat_vod="actor", desc="演员。"),
    _exact("director", {VOD}, flat_vod="director", desc="导演。"),
    _exact("dubbing", {VOD}, desc="配音演员。"),
    _exact("entertainer", {VOD}, flat_vod="entertainer", desc="参与人员（歌手/嘉宾等）。"),
    _exact("hostess", {VOD}, desc="主持人。"),
    _exact("creation_source", {VOD}, desc="剧本来源，如小说改编、原创。"),
    _exact("sound", {VOD}, desc="音效，如杜比。"),
    _exact("tag", {VOD}, flat_vod="tag", desc="标签（影视）。"),
    _exact("target", {VOD}, desc="受众人群，如儿童、少儿。"),
    _exact("vender_name", {VOD}, desc="供应商，如芒果TV。"),
    _exact("writer", {VOD}, desc="作者/作家。"),
    _exact("definition", {VOD}, desc="清晰度：is_3d / is_4k。"),
    _exact("area", {VOD}, desc="地区（影视 nested 专用）。"),
    _exact("category", {VOD}, flat_vod="category", flat_mode=FlatMode.ENUM, desc="影视一级分类，如电影、电视剧、综艺。"),
    _exact("channel", {VOD}, desc="频道。"),
    _exact("comedy_brand", {VOD}, desc="喜剧厂牌。"),
    _exact("technology", {VOD}, desc="拍摄技术，如全景。"),

    # country：educ nested 有；两域 flat 都有
    _exact("country", {EDUC}, flat_vod="country", flat_educ="country", desc="国家/地区（地理），与 language 区分。"),

    # ---- 少儿专有精确维度 ----
    _exact("content_type", {EDUC}, desc="媒资表现形式（固定候选）：动画、动画片。"),
    _exact("children_second_genre", {EDUC}, flat_educ="tag", flat_mode=FlatMode.ENUM,
           desc="蛛网二级分类（固定候选），如儿歌/故事/科普。慢链路对应 tag。"),
    _exact("children_third_genre", {EDUC}, desc="蛛网三级分类（开放型），如益智动画。"),
    _exact("training_objectives", {EDUC}, desc="少儿培养目标（开放型）。"),
    _exact("multiple_intelligences", {EDUC}, desc="少儿八大智能（开放型）。"),
    _exact("festival", {EDUC}, desc="节日/学期，如除夕、开学季。"),
    _exact("features", {EDUC}, desc="音效（固定候选）：杜比。"),
    _exact("grade", {EDUC}, desc="年级/学段，如小班、一年级。"),

    # ---- 状态维度 ----
    FieldSpec("fee", Kind.STATUS, BOTH,
              nested=NestedTarget({VOD: "fee", EDUC: "is_fee"}, "enum01"),
              flat=FlatTarget({VOD: "is_fee", EDUC: "is_fee"}, FlatMode.ENUM),
              desc="免付费标识：0=免费，1=付费。"),
    FieldSpec("is_over", Kind.STATUS, frozenset({VOD}),
              nested=NestedTarget({VOD: "is_over", EDUC: None}, "enum01"),
              flat=None,  # 慢链路无对应
              desc="连载状态：0=连载中，1=已完结（仅影视）。"),

    # ---- 范围维度 ----
    FieldSpec("age", Kind.RANGE, BOTH,
              nested=NestedTarget({VOD: "age", EDUC: "age_range"}, "int"),
              flat=FlatTarget({VOD: "age", EDUC: "age_range"}, FlatMode.RANGE),
              desc="年龄范围（整数）。"),
    FieldSpec("release_year", Kind.RANGE, BOTH,
              nested=NestedTarget({VOD: "release_year", EDUC: "release_year"}, "yyyyMMdd"),
              flat=FlatTarget({VOD: "date", EDUC: None}, FlatMode.RANGE),  # educ 慢链路无日期
              desc="发布时间范围，yyyyMMdd。"),
    FieldSpec("rate", Kind.RANGE, frozenset({VOD}),
              nested=NestedTarget({VOD: "rate", EDUC: None}, "number"),
              flat=FlatTarget({VOD: "rate", EDUC: None}, FlatMode.RANGE),
              desc="评分范围 0-10（仅影视 nested）。"),
]

FIELD_REGISTRY: dict[str, FieldSpec] = {f.canonical: f for f in _FIELDS}


# ---------------------------------------------------------------------------
# 排序键：canonical -> 各后端落地
# ---------------------------------------------------------------------------
# nested: sort 是独立对象 {key:{order}}；flat: 编码进字段值（rate/play/hot="desc"，new->date）
@dataclass(frozen=True)
class SortSpec:
    key: str
    domains: frozenset[str]
    nested_key: str                       # nested sort 对象里的键
    flat_field_by_domain: dict[str, Optional[str]]  # flat 里承载该排序的字段名


SORT_REGISTRY: dict[str, SortSpec] = {
    "rate": SortSpec("rate", BOTH, "rate", {VOD: "rate", EDUC: None}),
    "hot":  SortSpec("hot", BOTH, "hot", {VOD: "hot", EDUC: None}),
    "new":  SortSpec("new", BOTH, "new", {VOD: "date", EDUC: None}),
    "play": SortSpec("play", frozenset({VOD}), "play", {VOD: "play", EDUC: None}),
}

# 影视播放控制字段（IR 里放在 playback 段，仅 vod）
PLAYBACK_FIELDS: dict[str, str] = {
    "series": "int",       # 第几部/季
    "videoIndex": "int",   # 第几集/期
    "voiceStartPos": "int" # 起播秒数
}


# ---------------------------------------------------------------------------
# 便捷查询函数（供 grammar / compiler / validator 使用）
# ---------------------------------------------------------------------------
def fields_for_domain(domain: str, kind: Optional[Kind] = None) -> list[FieldSpec]:
    out = [f for f in _FIELDS if domain in f.domains]
    if kind is not None:
        out = [f for f in out if f.kind is kind]
    return out


def field_names(domain: str, kind: Optional[Kind] = None) -> list[str]:
    return [f.canonical for f in fields_for_domain(domain, kind)]


def get_field(canonical: str) -> Optional[FieldSpec]:
    return FIELD_REGISTRY.get(canonical)


def sort_keys_for_domain(domain: str) -> list[str]:
    return [k for k, s in SORT_REGISTRY.items() if domain in s.domains]
