"""Prompt 模板：路由阶段 + IR 生成阶段。

设计要点：
  * 路由：显式判定树（域 -> 意图大类 -> 明确检索 vs 模糊描述），输出结构化并受 schema 约束。
  * IR 生成：只注入选中 domain 的字段清单（精简描述），并给覆盖易错模式的 few-shot。
"""
from __future__ import annotations

import json

from .registry import Kind, fields_for_domain, sort_keys_for_domain

ROUTE_SYSTEM = """你是电视端 AI 助手的工具路由器。根据用户请求判断应调用哪个工具。

判定步骤：
1. 判断领域 domain：少儿内容(动画片/儿歌/早教/绘本…)=educ；其余影视(电影/电视剧/综艺/动漫…)=vod。
2. 判断意图 intent：
   - search：用户给出【明确】的名称/演员/分类等可直接检索的条件。如“我想看刘德华的电影”“播放甄嬛传”“免费的英语儿歌”。
   - slow_search：【模糊/描述/问答式】需要语义理解。如“骑着蓝色大鸟的人的电影”“最近有什么好看的”“适合3到6岁男孩看的启蒙动画”。
   - relate：找相似，“类似XX的/像XX一样的”。
   - personalized：个性化，“推荐我喜欢的/给我推荐点好看的”。
   - history：看历史记录，“我看过的/最近看了什么”。
   - clip：影视片段/台词/名场面切片（仅 vod）。
3. 输出对应 tool 名与 confidence(0-1)。低置信时倾向 slow_search（模糊场景安全兜底）。

只输出 JSON。"""

IR_SYSTEM = """你是电视端媒资检索的查询解析器。把用户请求解析成统一的布尔查询 IR（域无关的中间表示）。
你【不需要】关心最终调用哪个工具、也不需要关心字段在不同领域的命名差异——只产出 IR，后续由编译器落地。

IR 结构：
{{
  "domain": "{domain}",
  "action": "search",                       // 起播用 "play"
  "query": <节点>,
  "sort": [{{"key":"rate|hot|new|play","order":"asc|desc"}}],   // 可选
  "playback": {{"series":2,"videoIndex":3}}                     // 可选，仅 vod 起播
}}

节点四种形态（可任意嵌套）：
  {{"and":[节点,...]}}                        // 都要满足
  {{"or":[节点,...]}}                         // 满足其一
  {{"not":节点}}                              // 取反
  {{"field":"F","value":"X"}}                 // 精确单值
  {{"field":"F","values":["X","Y"],"op":"and|or"}}  // 同字段多值，op 默认 or
  {{"field":"F","value":0}}                   // 状态字段(fee/is_over)，0 或 1
  {{"field":"F","range":{{"from":"...","to":"..."}}}}  // 范围，开区间用 "*"

规则：
  - 同一维度多个取值 -> 用 values + op（如“刘德华和吴京都演的” -> op:"and"）。
  - 不同维度组合 -> 用 and/or 包裹。
  - 排除条件（不要/除了）-> not。
  - 日期用 release_year，值为 yyyyMMdd；年龄用 age 整数区间；评分用 rate 0-10（仅 vod）。
  - 只使用下面清单里的字段，不要编造字段名。

{domain} 可用字段：
{field_list}

{sort_hint}

只输出 JSON IR。"""


def build_route_messages(query: str, memory_hint: str = "") -> list[dict]:
    user = f"用户请求：{query}"
    if memory_hint:
        user += f"\n对话上下文：{memory_hint}"
    return [
        {"role": "system", "content": ROUTE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _field_catalog(domain: str) -> str:
    lines = []
    for f in fields_for_domain(domain):
        tag = {Kind.EXACT: "精确", Kind.STATUS: "状态", Kind.RANGE: "范围"}[f.kind]
        lines.append(f"  - {f.canonical} [{tag}]: {f.desc}")
    return "\n".join(lines)


def build_ir_messages(query: str, domain: str, memory_hint: str = "",
                      few_shots: list[tuple[str, dict]] | None = None) -> list[dict]:
    sort_keys = sort_keys_for_domain(domain)
    sort_hint = f"可用排序键：{', '.join(sort_keys)}" if sort_keys else ""
    system = IR_SYSTEM.format(
        domain=domain,
        field_list=_field_catalog(domain),
        sort_hint=sort_hint,
    )
    messages = [{"role": "system", "content": system}]

    for q, ir in (few_shots or _DEFAULT_FEWSHOTS.get(domain, [])):
        messages.append({"role": "user", "content": f"用户请求：{q}"})
        messages.append({"role": "assistant", "content": json.dumps(ir, ensure_ascii=False)})

    user = f"用户请求：{query}"
    if memory_hint:
        user += f"\n对话上下文：{memory_hint}"
    messages.append({"role": "user", "content": user})
    return messages


# 覆盖易错模式的 few-shot（单值/多值 and/or/not/range/status/sort/嵌套）
_DEFAULT_FEWSHOTS: dict[str, list[tuple[str, dict]]] = {
    "vod": [
        ("我想看刘德华的免费电影",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "actor", "value": "刘德华"},
              {"field": "category", "value": "电影"},
              {"field": "fee", "value": 0}]}}),
        ("刘德华和吴京都参演、但不是恐怖片的高分电影，按评分从高到低",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "actor", "values": ["刘德华", "吴京"], "op": "and"},
              {"field": "category", "value": "电影"},
              {"not": {"field": "tag", "value": "恐怖"}},
              {"field": "rate", "range": {"from": 8, "to": "*"}}]},
          "sort": [{"key": "rate", "order": "desc"}]}),
        ("播放甄嬛传第二季第三集",
         {"domain": "vod", "action": "play",
          "query": {"field": "title", "value": "甄嬛传"},
          "playback": {"series": 2, "videoIndex": 3}}),
    ],
    "educ": [
        ("给孩子找免费的英语儿歌",
         {"domain": "educ", "action": "search",
          "query": {"and": [
              {"field": "children_second_genre", "value": "儿歌"},
              {"field": "language", "value": "英语"},
              {"field": "fee", "value": 0}]}}),
        ("适合3到6岁女孩看的科普或数学动画",
         {"domain": "educ", "action": "search",
          "query": {"and": [
              {"field": "gender", "value": "女"},
              {"field": "age", "range": {"from": 3, "to": 6}},
              {"or": [
                  {"field": "children_second_genre", "value": "科普"},
                  {"field": "children_second_genre", "value": "数学"}]}]}}),
    ],
}
