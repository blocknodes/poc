"""Prompt 模板：路由阶段 + IR 生成阶段。

设计要点：
  * 路由：显式判定树（域 -> 意图大类 -> 明确检索 vs 模糊描述），输出结构化并受 schema 约束。
  * IR 生成：只注入选中 domain 的字段清单（精简描述），并给覆盖易错模式的 few-shot。
"""
from __future__ import annotations

import json

from .registry import Kind, fields_for_domain, sort_keys_for_domain

ROUTE_SYSTEM = """你是电视端 AI 助手的工具路由器。根据用户请求判断应调用哪个工具。只输出 JSON。

第一步 判断领域 domain：
  - educ：**学龄前早教/启蒙**类少儿内容——儿歌、绘本、点读、拼音/数字/英语启蒙、早教课程等。
  - vod：其余全部影视。**动画片/动漫/少儿动画/儿童动画/少儿动漫 一律算 vod**（它们是影视内容，
    不要因为出现“少儿/儿童”就选 educ）；电影/电视剧/综艺/纪录片等也都是 vod。拿不准默认 vod。

第二步 判断意图 intent（按下面顺序逐条命中，命中即停）：

【clip】影视「片段/台词/名场面/具体剧情」相关，仅 vod。命中任一即 clip：
  - 引用一句台词问出处：“XXX 是哪部剧/哪部电影的台词”“XXX 出自哪部片子”。
  - 用一段剧情描述问剧名：“那个警察退休后追查旧案的剧叫什么”“讲核潜艇研发的电视剧叫什么”。
  - 问某个具体场景/名场面在第几集：“苏明玉逆袭那段是都挺好第几集”“XX 是哪一集”。
  - 找某类打斗/名场面/反转「场面/片段」：“古装剧打斗场面”“反转剧情盘点”“XX 那段视频”。
  注意：以下不是 clip，属于 search——
  预告片/花絮/片花/短片/短视频/宣传片；以及“第 X 分钟/X 分 Y 秒”这类按时间点起播的请求。

【relate】找相似：“类似XX的/像XX一样的/和XX差不多的”。

【personalized】纯个性化推荐、无任何具体条件：“给我推荐点好看的”“推荐我可能喜欢的”。

【history】看观看历史：“我看过的/最近看了什么/继续看上次的”。

【slow_search】需要语义理解、无法直接落成「名称/演员/分类/年份/免费/热度」结构化条件的模糊检索。命中任一即 slow_search：
  - 场景/情绪/人群/节日/季节/天气 推荐：“适合心情低落时看的电影”“适合全家/女生/孕期看的”“适合劳动节/跨年看的”“适合冬天追的韩剧”。
  - 特殊内容形态：评书/快板/小品/相声/音乐剧/歌剧/话剧/脱口秀/春晚/演唱会/颁奖晚会/辩论赛。
  - 出品方/频道/奖项 限定：“正午阳光出品的剧”“cctv8 的剧”“XX 影业出品”“获得金像奖最佳男主角的影片”“白玉兰奖晚会”。
  - 版本/画质等模糊描述：“小兵张嘎高清版”“超高清电视节目”“全景拍摄的电影”。
  - 剧集问答（问集数/更新状态）：“河西走廊一共多少集”“XX 更新完了吗”。
  - 待播/时效：“即将上线的待播剧”“本周新播出的国产剧”。
  - 复杂多重排除或依赖上下文的模糊描述。
  - 问答式查询：问票房/集数/播出时间/演员年龄等（“XX 票房如何”“更新到第几集”“今年票房最高的电影”）。
  - 多轮追加条件：请求是一个条件列表（如 [\"高分爱情电影\", \"这里面要最近的\"]），依赖上下文细化。

【search】其余「可直接检索」的明确条件（这是最常见的默认分支）：
  - 明确名称/演员/分类/地区/年份/免费：“我想看刘德华的电影”“播放甄嬛传”“免费的国产剧”“4K 的外国电影”。
  - 热度/口碑/流行度 排序（这类归 search，用排序实现，不要归 slow_search）：
    “最近很火的电影”“口碑好的悬疑剧”“播放量高的电视剧”“大家都在看的”“最新上线的口碑新片”“好看的内地综艺”。

第三步 输出 domain、intent、对应 tool（domain 前缀 + intent；vod 的 slow_search 用 vod_slow_search_data_search）、confidence(0-1)。
区分不了 search 与 slow_search 时：能落成结构化条件就选 search，否则选 slow_search。"""

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


def _render_fewshot(query: str, obj: dict) -> str:
    return f"用户请求：{query}\n输出：{json.dumps(obj, ensure_ascii=False)}"


def route_system_prompt() -> str:
    """路由阶段 system prompt = ROUTE_SYSTEM + few-shot（以**文本**内嵌）。

    few-shot 内嵌成文本而非独立 role 轮次，是为了和 RL rollout 时 ms-swift
    GYMScheduler 只能注入「单 system + 单 user observation」的形状严格对齐——
    这样训练(rollout)与推理(harness)走的是逐 token 一致的 prompt。
    """
    shots = "\n\n".join(_render_fewshot(q, r) for q, r in _ROUTE_FEWSHOTS)
    return ROUTE_SYSTEM + "\n\n# 参考示例（只示范判定边界，不要照抄内容）：\n" + shots


def route_observation(query: str, memory_hint: str = "") -> str:
    """路由阶段注入的首个 user observation。"""
    obs = f"用户请求：{query}"
    if memory_hint:
        obs += f"\n对话上下文：{memory_hint}"
    obs += "\n\n请做路由判定：给出 domain / intent / tool / confidence，只输出路由 JSON。"
    return obs


# 覆盖 search / slow_search / clip / relate / personalized / history 各易错分支的路由 few-shot。
# 目的：把「明确检索 vs 模糊检索 vs 片段」的边界示范给模型，尤其纠正模型把
# 模糊/特殊形态/出处询问一律当成 vod_search 的倾向。
def _r(domain: str, intent: str, tool: str, conf: float = 0.9) -> dict:
    return {"domain": domain, "intent": intent, "tool": tool, "confidence": conf}


_ROUTE_FEWSHOTS: list[tuple[str, dict]] = [
    # --- search：明确条件 / 热度口碑排序 ---
    ("我想看刘德华的免费电影", _r("vod", "search", "vod_search")),
    ("最近很火的悬疑剧有哪些", _r("vod", "search", "vod_search")),
    ("口碑好、播放量高的国产剧", _r("vod", "search", "vod_search")),
    ("4K 的外国动作电影", _r("vod", "search", "vod_search")),
    ("最好看的少儿动画是什么", _r("vod", "search", "vod_search")),
    ("豆瓣评分8分以上的动画片", _r("vod", "search", "vod_search")),
    ("拼音启蒙儿歌", _r("educ", "search", "educ_search")),
    # --- slow_search：模糊场景 / 特殊形态 / 出品方频道奖项 / 剧集问答 / 待播 ---
    ("推荐一些适合心情低落时观看的电影吧", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("适合全家人一起看的电视剧", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("评书岳飞传刘兰芳", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("音乐剧悲惨世界", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("正午阳光出品的电视剧", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("获得香港金像奖最佳男主角的影片", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("超高清电视节目", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("河西走廊一共多少集，更新完了吗", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("即将上线的待播剧", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("今年票房最高的电影是哪部", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ('["高分爱情电影", "这里面要最近的"]', _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("适合3到6岁孩子看的启蒙拼音课程", _r("educ", "slow_search", "educ_slow_search_data_search")),
    # --- clip：台词出处 / 剧情找剧名 / 场景第几集 / 名场面片段 ---
    ("“臣妾做不到啊”出自哪部电视剧", _r("vod", "clip", "vod_clip_search")),
    ("那个警察退休后追查旧案的剧叫什么", _r("vod", "clip", "vod_clip_search")),
    ("苏明玉逆袭那段是都挺好第几集", _r("vod", "clip", "vod_clip_search")),
    ("2025 年古装剧的打斗场面盘点", _r("vod", "clip", "vod_clip_search")),
    # --- relate / personalized / history ---
    ("有没有类似甄嬛传的宫斗剧", _r("vod", "relate", "vod_relate_recommend")),
    ("给我推荐点好看的", _r("vod", "personalized", "vod_personalized_recommend")),
    ("我最近看过什么", _r("vod", "history", "vod_history")),
]


def _field_catalog(domain: str) -> str:
    lines = []
    for f in fields_for_domain(domain):
        tag = {Kind.EXACT: "精确", Kind.STATUS: "状态", Kind.RANGE: "范围"}[f.kind]
        lines.append(f"  - {f.canonical} [{tag}]: {f.desc}")
    return "\n".join(lines)


def ir_observation(query: str, domain: str, memory_hint: str = "",
                   few_shots: list[tuple[str, dict]] | None = None) -> str:
    """IR 阶段注入的 user observation（IR_SYSTEM + few-shot + 请求，**全部为文本**）。

    注意：IR 指令这里放进 user turn 而非 system——RL rollout 时 GYMScheduler 的
    后续 observation 只能作为 user 追加，无法再插入新的 system。推理侧同样走此形状，
    确保训练/推理一致。
    """
    sort_keys = sort_keys_for_domain(domain)
    sort_hint = f"可用排序键：{', '.join(sort_keys)}" if sort_keys else ""
    system = IR_SYSTEM.format(
        domain=domain,
        field_list=_field_catalog(domain),
        sort_hint=sort_hint,
    )
    obs = system
    shots = "\n\n".join(_render_fewshot(q, ir)
                        for q, ir in (few_shots or _DEFAULT_FEWSHOTS.get(domain, [])))
    if shots:
        obs += "\n\n# 参考示例：\n" + shots
    obs += f"\n\n用户请求：{query}"
    if memory_hint:
        obs += f"\n对话上下文：{memory_hint}"
    obs += "\n只输出 IR JSON。"
    return obs


def repair_observation(errs: list[str]) -> str:
    """IR 校验失败时回灌给模型的 observation。"""
    return "上面的 IR 有以下问题，请修正后重新只输出 JSON：\n- " + "\n- ".join(errs)


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
