"""Field Registry —— 整个 planner 的单一事实源 (single source of truth)。

它同时驱动三件事：
  1. IR 校验（哪些字段在某个 domain 合法、支持哪些操作）；
  2. 约束解码 grammar 的生成（每个 domain 允许的 field 枚举）；
  3. 编译器双后端（nested-json / flat-string）如何把一个 canonical 字段落地到目标工具。

设计原则：模型只认识 canonical 字段名与统一的 IR 结构；影视/少儿的命名差异
（如 is_fee vs fee、age_range vs age）与序列化差异全部在这里声明、由编译器消化。

字段命名对齐 0725-v1 tool schema：
  * nested vod: age_range / release_time / is_fee / video_index
  * playback: series / video_index / voiceStartPos

扩展：新增 audio（有声）和 device（设备控制）两个域。
  * audio: 不走布尔 IR，仅路由后 slot-fill（query + play_mode + screen_mode）。
  * device: 不走布尔 IR，路由到 20 个设备工具之一，然后 slot-fill（operation + object + value）。
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
    # release_time 需要 yyyyMMdd；age_range 是 int；rate 是 0-10 number
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
AUDIO = "audio"
DEVICE = "device"
BOTH = frozenset({VOD, EDUC})

# 所有支持的 domain
ALL_DOMAINS = frozenset({VOD, EDUC, AUDIO, DEVICE})

# 走 IR 编译器的 domain（audio / device 不走 IR）
IR_DOMAINS = frozenset({VOD, EDUC})


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
# 字段登记表（影视 + 少儿）
# 对齐 0725-v1 tool schema
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

    # ---- 影视专有精确维度（对齐 vod_search_all 24 字段）----
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

    # country：vod nested 里没有（vod_search_all 无 country），flat 有；educ nested 有
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
    # 0725-v1: nested vod 用 is_fee（之前是 fee），flat 也是 is_fee
    FieldSpec("fee", Kind.STATUS, BOTH,
              nested=NestedTarget({VOD: "is_fee", EDUC: "is_fee"}, "enum01"),
              flat=FlatTarget({VOD: "is_fee", EDUC: "is_fee"}, FlatMode.ENUM),
              desc="免付费标识：0=免费，1=付费。"),
    FieldSpec("is_over", Kind.STATUS, frozenset({VOD}),
              nested=NestedTarget({VOD: "is_over", EDUC: None}, "enum01"),
              flat=None,  # 慢链路无对应
              desc="连载状态：0=连载中，1=已完结（仅影视）。"),

    # ---- 范围维度 ----
    # 0725-v1: nested vod 用 age_range（之前是 age），格式不变
    FieldSpec("age", Kind.RANGE, BOTH,
              nested=NestedTarget({VOD: "age_range", EDUC: "age_range"}, "int"),
              flat=FlatTarget({VOD: "age", EDUC: "age_range"}, FlatMode.RANGE),
              desc="年龄范围（整数）。"),
    # 0725-v1: nested vod 用 release_time（之前是 release_year）
    FieldSpec("release_year", Kind.RANGE, BOTH,
              nested=NestedTarget({VOD: "release_time", EDUC: "release_year"}, "yyyyMMdd"),
              flat=FlatTarget({VOD: "date", EDUC: None}, FlatMode.RANGE),  # educ 慢链路无日期
              desc="发布时间范围，yyyyMMdd。"),
    FieldSpec("rate", Kind.RANGE, frozenset({VOD}),
              nested=NestedTarget({VOD: "rate", EDUC: None}, "number"),
              flat=FlatTarget({VOD: "rate", EDUC: None}, FlatMode.RANGE),
              desc="评分范围 0-10（仅影视 nested）。"),
]

FIELD_REGISTRY: dict[str, FieldSpec] = {f.canonical: f for f in _FIELDS}


# ---------------------------------------------------------------------------
# vod_search 精简版字段集（10 个精确字段）—— 用于编译后判断走 vod_search 还是 vod_search_all
# ---------------------------------------------------------------------------
VOD_SEARCH_FIELDS: frozenset[str] = frozenset([
    "title", "actor", "director", "entertainer", "prize",
    "role", "tag", "target", "definition", "category",
])

# vod_relate_search 支持的字段集（4 个精确字段）
VOD_RELATE_FIELDS: frozenset[str] = frozenset([
    "title", "actor", "director", "category",
])


# ---------------------------------------------------------------------------
# 排序键：canonical -> 各后端落地
# ---------------------------------------------------------------------------
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
# 0725-v1: 用 video_index（下划线，之前是 videoIndex）
PLAYBACK_FIELDS: dict[str, str] = {
    "series": "int",        # 第几部/季
    "video_index": "int",   # 第几集/期（0725-v1 命名）
    "voiceStartPos": "int"  # 起播秒数
}


# ---------------------------------------------------------------------------
# 有声（audio）域定义 —— 不走 IR，仅 slot-fill
# ---------------------------------------------------------------------------
AUDIO_PLAY_MODES = ["search", "play", "screen_off_play"]
AUDIO_SCREEN_MODES = ["normal", "screen_standby"]

# 有声域的工具列表
AUDIO_TOOLS = ["audio_search", "audio_chat_qa"]


# ---------------------------------------------------------------------------
# 设备控制（device）域定义 —— 不走 IR，路由后直接 slot-fill
# ---------------------------------------------------------------------------
# 对齐 设备工具0731 schema（17 工具）。其中 solve_picture_sound_problem_control
# 走独立的 intent 参数范式，其余 16 工具走 operation/object/value/date_time。
DEVICE_TOOLS = [
    "numeric_adjust",           # 幅值控制：音量/亮度/清晰度/对比度/色度/分辨率/倍速/刷新率
    "power_control",            # 开关机：开机/关机/重启
    "timer_control",            # 定时控制：定时开机/关机/熄屏、查询剩余关机时长
    "source_switch",            # 信号源切换：HDMI/typec/usb/机顶盒/数字电视等
    "playback_control",         # 播放控制：播放/停止/快进/快退/跳转/上下/循环等
    "video_picture_save_share", # 视频图片存储分享：保存/删除/分享(微信/抖音)
    "mode_control",             # 模式控制：图像/声音/音效/混响/护眼/整机模式切换
    "screen_layout",            # 画面布局：分屏/全屏/小屏/画面缩放
    "screen_lift_rotation",     # 屏幕升降旋转：横竖屏/旋转/支架升降
    "demo_control",             # 演示控制：demo/演示/展示/体验
    "display_control",          # 画质背光开关：画质/背光/畸变校正 功能开关
    "audio_control",            # 音质音效开关：音质/音效 功能开关
    "smart_camera",             # 摄像头/AI视觉/传感：摄像头/雷达/AI视觉
    "network_control",          # 网络设置：无线/有线网络/热点/测速
    "screensaver_control",      # 屏保控制：屏保/壁纸/熄屏/亮屏/待机/睡眠
    "common_control",           # 兜底：主页/设置/蓝牙/语言/输入法/主题/系统更新/媒体中心等
    "solve_picture_sound_problem_control",  # 解决音画问题（intent 范式）
]

# 设备工具统一的 slot 描述
DEVICE_OPERATIONS = ["提高", "降低", "打开", "关闭", "设置", "查询", "切换", "开启", "关", "开"]

# 工具名别名：评测集/上游可能使用的旧名 → 当前 schema 规范名。
# 0731 taxonomy 下无需别名（保留空表以兼容调用方）。
DEVICE_TOOL_ALIASES: dict[str, str] = {}


def canonical_device_tool(tool: str) -> str:
    """把设备工具别名归一到规范名。0731 taxonomy 下为恒等映射。"""
    return DEVICE_TOOL_ALIASES.get((tool or "").strip(), (tool or "").strip())


# solve_picture_sound_problem_control 使用独立的 intent 枚举（画质/音效异常诊断意图），
# 不走 operation/object/value 那套。此处为单一事实源，供 grammar / compiler / prompt 共用。
AI_PICTURE_SOUND_INTENTS = [
    "colors_oversaturated",   # 整体色彩偏浓
    "colors_washed_out",      # 整体色彩偏淡
    "cold_color_cast",        # 色彩偏冷
    "red_color_cast",         # 色彩偏红
    "warm_color_cast",        # 色彩偏暖
    "dark_areas_crushed",     # 暗场发黑，细节丢失
    "highlights_overexposed", # 亮部过曝、细节丢失
    "lack_of_clarity",        # 画面不够清晰
    "image_noise_obvious",    # 画面噪点明显
    "motion_not_smooth",      # 运动画面不够流畅
    "motion_not_sharp",       # 运动画面不够清晰
    "poor_eye_comfort",       # 屏幕不够护眼
    "image_mode_movie",       # 电影模式切换
    "flickering_brightness",  # 亮度忽明忽暗
    "image_mode_auto",        # 自动切换图像模式
    "voice_not_clear",        # 人声听不清楚
    "lack_stereo_surround",   # 没有立体、环绕感
    "bass_effect_poor",       # 低音效果不好
]

# 使用 intent 而非 operation/object/value 的设备工具
INTENT_DEVICE_TOOLS = frozenset({"solve_picture_sound_problem_control"})


# ---------------------------------------------------------------------------
# 设备 object → tool 词表（从 0731 schema 的 object 候选描述解析，schema 为事实源）
# 用于 compiler 层高精度工具纠偏：当模型填出的 object 恰是某工具的登记对象时，
# 强制归到该工具。仅做**精确匹配**，避免子串误伤（如 氛围灯 vs 氛围灯亮度）。
# ---------------------------------------------------------------------------
import csv as _csv
import json as _json
import re as _re
from pathlib import Path as _Path

_DEVICE_OBJECT_TOOL_MAP: Optional[dict[str, str]] = None
# 过于通用、易误伤的 object 词不纳入精确表
_OBJECT_VOCAB_STOPWORDS = frozenset({
    "设置", "屏幕", "画面", "等", "控制对象", "无对应值时为空",
    "当前工具通常为空", "date_time", "信号源",
})


def _parse_device_object_vocab() -> dict[str, str]:
    """从 tools_schema/设备工具0731 解析每个工具的 object 候选，构建 object→tool 精确表。

    schema 缺失或解析失败时返回空表（生产安全：不影响主流程）。
    """
    mapping: dict[str, str] = {}
    # power/timer 靠 date_time 区分（object 都可能是 开机/关机），不纳入 object→tool 表
    _exclude_tools = {"power_control", "timer_control"}
    schema_path = _Path(__file__).resolve().parent.parent / "tools_schema" / "设备工具0731 - Sheet1.csv"
    if not schema_path.exists():
        return mapping
    try:
        with open(schema_path, encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                name = (row.get("name") or "").strip()
                if name not in DEVICE_TOOLS or name in _exclude_tools:
                    continue
                try:
                    pj = _json.loads(row.get("parameters") or "")
                except Exception:
                    continue
                desc = (((pj.get("properties") or {}).get("object") or {}).get("description") or "")
                desc = _re.sub(r"^控制对象[，,]?", "", desc)
                desc = _re.sub(r"^(如|可选范围|例如)[:：]?", "", desc)
                for part in _re.split(r"[、；/，,\s]+", desc):
                    p = part.strip().lower().strip("[]（）()。")
                    if len(p) < 2 or p in _OBJECT_VOCAB_STOPWORDS:
                        continue
                    # 过滤解析残留的非纯净词（含括号等）
                    if any(c in p for c in "（）()[]"):
                        continue
                    # 首个登记者为准（schema 顺序），不覆盖
                    mapping.setdefault(p, name)
    except Exception:
        return {}
    return mapping


def device_object_tool_map() -> dict[str, str]:
    """惰性缓存的 object→tool 精确词表。"""
    global _DEVICE_OBJECT_TOOL_MAP
    if _DEVICE_OBJECT_TOOL_MAP is None:
        _DEVICE_OBJECT_TOOL_MAP = _parse_device_object_vocab()
    return _DEVICE_OBJECT_TOOL_MAP


# 按长度降序缓存的词表键（供 query 子串检索用最长匹配）
_DEVICE_VOCAB_BY_LEN: Optional[list[str]] = None


def device_tool_by_query(query: str, min_len: int = 4) -> Optional[tuple[str, str]]:
    """schema 检索式路由（②）：在 query 中查找**最长**的已登记功能名（object 词表），
    命中则返回 (matched_term, tool)。用于纠正模型工具路由（不改 object 本身）。

    仅用较长(≥min_len)的功能名做子串匹配，避免短词误伤；不含 power/timer（已排除）。
    """
    global _DEVICE_VOCAB_BY_LEN
    q = (query or "").strip().lower()
    if not q:
        return None
    vocab = device_object_tool_map()
    if _DEVICE_VOCAB_BY_LEN is None:
        _DEVICE_VOCAB_BY_LEN = sorted((k for k in vocab if len(k) >= min_len),
                                      key=len, reverse=True)
    for term in _DEVICE_VOCAB_BY_LEN:
        if term in q:
            return term, vocab[term]
    return None


def device_tool_by_object_fuzzy(obj: str, cutoff: float = 0.82) -> Optional[str]:
    """模糊匹配（①）：模型填的 object 不在精确词表时，取最近邻词表项的 tool。"""
    o = (obj or "").strip().lower()
    if len(o) < 3:
        return None
    import difflib
    vocab = device_object_tool_map()
    cand = difflib.get_close_matches(o, list(vocab), n=1, cutoff=cutoff)
    return vocab[cand[0]] if cand else None


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
