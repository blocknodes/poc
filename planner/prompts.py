"""Prompt 模板：路由阶段 + IR 生成阶段 + 有声/设备 slot-fill 阶段。

设计要点：
  * 路由：显式判定树（域 -> 意图大类），输出结构化并受 schema 约束。
    - 支持全域路由（4 域）和影视单域路由（只输出 vod 工具）。
  * IR 生成：只注入选中 domain 的字段清单，few-shot 覆盖易错模式。
  * 有声 slot-fill：填 query + play_mode。
  * 设备 slot-fill：选工具 + 填 operation/object/value。
"""
from __future__ import annotations

import json

from .registry import Kind, fields_for_domain, sort_keys_for_domain, DEVICE_TOOLS

# ===========================================================================
# 路由 —— 全域版（vod/educ/audio/device）
# ===========================================================================
ROUTE_SYSTEM = """你是电视端 AI 助手的工具路由器。根据用户请求判断应调用哪个工具。只输出 JSON。

第一步 判断领域 domain：
  - device：设备控制——音量、电源、信号源、屏幕、摄像头、网络、蓝牙、画质、声音模式、投影、播放控制等。
  - audio：有声内容——听书、评书、相声、广播剧、有声故事等音频专辑（不含歌曲/音乐）。
  - educ：学龄前早教/启蒙——儿歌、绘本、点读、拼音/英语启蒙等。
  - vod：其余全部影视。动画片/动漫/少儿动画一律算 vod。拿不准默认 vod。

第二步 按 domain 判断 intent + tool：

domain=device → intent=device_control，tool 从设备工具表选。
domain=audio → intent∈{audio_search, audio_play, audio_screen_off_play, audio_chat_qa}，tool=audio_search|audio_chat_qa。
domain=educ → intent∈{search, slow_search, relate, history}，tool=educ_search|educ_slow_search_data_search|educ_relate_recommend|educ_history。
domain=vod → 见下方判定规则。

=== vod intent 判定规则 ===
【play】起播/观看意图动词（播放/放/起播/继续播放/看/想看/要看/打开/收看）+ **具体片名** → tool=vod_search
  - "播放琅琊榜" → play；"我想看铡美案" → play；"打开白日提灯" → play
  - 注意：动词+纯条件无具体片名 → search（如"播放言情剧""我要看免费电影"）
【search】条件筛选/浏览/无具体片名的请求（默认）→ tool=vod_search：
  - 明确条件（演员/分类/地区/年份/免费/出品方/频道/奖项）
  - "想看/要看" + 只有条件没有具体片名
  - 热度/口碑排序
【relate】"类似XX/像XX一样/和XX差不多" → tool=vod_relate_search
【personalized】纯个性化推荐、无具体条件（"推荐我喜欢的/根据我兴趣"）→ tool=vod_personalized_search
【history】查观看历史（"我看过的/最近看了什么/继续看上次的"）→ tool=vod_history
【slow_search】需语义理解、无法直接结构化的模糊检索 → tool=vod_slow_search_data_search：
  - 场景/情绪/人群/节日推荐
  - 版本词/画质词（如"八三版""修复版""高清版"——标签不支持的）
  - 剧情/台词/名场面出处查询
  - 剧集问答（集数/更新状态/票房）、待播/时效
  - 复杂模糊描述
【search】其余可直接检索的明确条件（默认） → tool=vod_search：
  - 明确名称/演员/分类/地区/年份/免费/出品方/频道/奖项
  - 热度/口碑排序

第三步 输出 {domain, intent, tool, confidence}。

设备 tool 表：volume_control|power_control|signal_source_control|screen_display_control|camera_control|network_control|bluetooth_control|input_lang_control|image_quality_control|sound_mode_control|projection_control|media_center_control|demo_control|personalization_control|scene_mode_control|screen_safety_control|playback_control|ambient_light_control|system_settings_control|ai_picture_sound_control"""

# ===========================================================================
# 路由 —— 影视单域版（只评测 vod，无 audio/device/educ 干扰）
# ===========================================================================
ROUTE_SYSTEM_VOD = """你是影视内容的工具路由器。根据用户请求判断应调用哪个影视工具。只输出 JSON。

工具列表：
  vod_search — 精确检索/起播（明确标题/演员/分类/地区/年份/免费/奖项/出品方/频道等条件）
  vod_slow_search_data_search — 慢链路语义搜索（模糊/复杂/无法结构化的描述）
  vod_relate_search — 相关推荐（"类似XX的/像XX一样的"）
  vod_personalized_search — 个性化推荐（"推荐我喜欢的/根据我兴趣"）
  vod_history — 历史记录（"我看过的/最近看了什么"）

intent 判定规则：
  play — 有起播/观看意图动词（播放/放/起播/继续看/想看/要看/看/打开/收看）+ **具体片名/内容** → tool=vod_search
  search — 条件筛选/搜索（搜/搜索/找/有没有/查），或"想看/要看"+纯条件无具体片名 → tool=vod_search
  slow_search — 需语义理解的模糊检索 → tool=vod_slow_search_data_search
  relate — 找相似内容 → tool=vod_relate_search
  personalized — 纯个性化推荐 → tool=vod_personalized_search
  history — 查观看历史 → tool=vod_history

play vs search 关键区分：
  - "播放琅琊榜" → play（有具体片名）
  - "我想看戏曲铡美案" → play（有具体片名"铡美案"）
  - "我想看韩剧" → search（无具体片名，只有分类条件）
  - "我要看免费的影片" → search（无具体片名，只有条件）
  - "播放言情剧" → search（无具体片名，"言情剧"是分类不是片名）
  - "看免费高分综艺" → search（纯条件筛选）

slow_search 命中条件（任一即 slow_search）：
  - 场景/情绪/人群/节日推荐（适合XX看的/适合XX时候看的）
  - 版本词（八三版/修复版/高清版——标签系统不支持的描述）
  - 剧情/台词/名场面出处（"那句台词出自哪""骑蓝色大鸟的电影"）
  - 剧集问答（更新了几集/票房多少）、待播/时效
  - 复杂模糊描述

注意：
  - "播放/放/看/想看/要看/打开/收看" + **具体片名** → play
  - 动词 + 纯条件无具体片名 → search（如"播放言情剧""我要看免费电影""看免费高分综艺"）
  - "搜索/搜/找/有没有" → search
  - "适合女生看的电影" 有 gender 可结构化但标签系统未必支持 → slow_search

输出：{domain:"vod", intent, tool, confidence}"""


# ===========================================================================
# IR 生成
# ===========================================================================
IR_SYSTEM = """你是影视媒资查询解析器。把用户请求解析成布尔查询 IR。只产出 IR JSON，后续由编译器落地。

IR 结构：
{{
  "domain": "{domain}",
  "action": "search|play",
  "query": <节点>,
  "sort": [{{"key":"rate|hot|new|play","order":"asc|desc"}}],
  "playback": {{"series":2,"video_index":3}}  // 仅 play 时可选
}}

节点形态：
  {{"and":[...]}} / {{"or":[...]}} / {{"not":节点}}
  {{"field":"F","value":"X"}}
  {{"field":"F","values":["X","Y"],"op":"and|or"}}  // 同字段多值
  {{"field":"F","value":0|1}}  // 状态字段
  {{"field":"F","range":{{"from":"...","to":"..."}}}}  // 范围，开区间用"*"

关键规则：
  1. action 判定——**以路由阶段为准**：
     - 若路由判定 intent=play，action 填 "play"。
     - 若路由判定 intent=search，action 填 "search"。
     - 不需要自行重新判定 action，编译器会以路由为准覆盖。
  2. playback 字段——**仅当用户显式说了集数/季/部才填**：
     - "第N季/第N部" → series:N；"第M集/第M期" → video_index:M
     - "第X分钟/X分Y秒" → voiceStartPos:秒数（注意转换为秒）
     - **禁止**：用户没提到集数时绝对不填 playback，不要默认加 series:1/video_index:1。
  3. category——**必须从用户表达中明确推断**，不要自行脑补：
     - 电影/片/影片/大片/XX片(武侠片/鬼片/抗战片) → category:"电影"
     - 电视剧/剧/连续剧/XX剧(韩剧/偶像剧/言情剧/谍战剧) → category:"电视剧"
     - 综艺/综艺节目 → category:"综艺"
     - 纪录片 → category:"纪录片"
     - 戏曲 → category:"戏曲"
     - 短视频 → category:"短视频"
     - 花絮 → category:"花絮"
     - **不要推断**的情况：用户没说出任何上述分类词时不加 category：
       "庆余年第二季" → 不加 category（虽然你知道它是电视剧）
       "陈翔六点半" → 不加 category（它是什么由检索决定）
       "小品XX/相声XX" → 不加 category:综艺，用 tag:小品/tag:相声
     - 注意："XX剧"一律有category:"电视剧"——"韩剧/偶像剧/言情剧" = category:电视剧 + area/tag
  4. tag 值——**保持用户的原始表达词**，不做同义替换：
     - 用户说"抗战" → tag:"抗战"（不要替换为"战争"）
     - 用户说"抗日" → tag:"抗日"（不要替换为"战争"）
     - 用户说"言情" → tag:"言情"（不要替换为"爱情"）
     - 用户说"破案" → tag:"破案"（不要替换为"悬疑"）
     - 用户说"恋爱综艺" → tag:"恋爱体验"
     - 用户说"鬼片" → tag:"恐怖"（"鬼片"不是标准tag，需要映射）
     - 用户说"搞笑" → tag:"喜剧"（"搞笑"不是标准tag，需要映射）
  5. tag vs 其他字段——**字段选择规则**（experience bank）：
     - "真人版/TV版/院线版/导演剪辑版/删减版" → field:tag（不是 language！）
     - "秦腔/京剧/豫剧/越剧/黄梅戏" → field:tag（戏曲子类型，不加 category:戏曲）
     - "小品/相声/脱口秀" → field:tag（综艺子类型，不加 category:综艺）
     - "人文/自然/美食/探险" → field:tag（纪录片子类型，不加 category:纪录片）
     - "辩论赛/晚会/春晚" → field:tag
     - **不是 tag 的（不要放 tag）**：
       "老版/新版" → 版本描述，走 slow_search（无法结构化）
       "民间小调" → 不是标准 tag，走 title 或 slow_search
       "花絮" → category:"花絮"（不是 tag）
       "友情出演" → 不是 tag，走 slow_search 或忽略
       "新剧" → sort:new:desc（不是 tag，是排序意图）
       "泰语版" → language:"泰语"（不是 tag）
  6. role vs title 区分：
     - 虚拟角色/剧中人物名（如孙悟空、詹姆斯·邦德、哪吒、吴石将军）→ field:role
     - 作品名 → field:title
     - 真实演员 → field:actor
  7. 同字段多值 → values+op：
     - "港台" → area:["香港","台湾"],op:"or"
     - "恐怖悬疑" → tag:values:["恐怖","悬疑"],op:"and"
     - "金秀贤、金智媛" → actor:values:["金秀贤","金智媛"]
  8. 排除（不要/除了）→ not。多个排除项 → not(or([...]))。
  9. 日期 release_year 格式 yyyyMMdd；年龄 age 整数；评分 rate 0-10（开区间用"*"）。
  10. sort 规则（只在用户**明确表达排序意图**时才加）：
     - "好看的/好评/评分高" → sort rate:desc
     - "最新的/新出的/最近" → sort new:desc
     - "热门/最火/很火" → sort hot:desc
     - "播放量高" → sort play:desc
     - 用户没提排序意图时**不加** sort（"好看的" 只是修饰不一定要排序）。
  11. 值大小写：公司/供应商名保持原始大小写（BBC/TVB/芒果TV）。
  12. title 不要带数字后缀——如果用户说"战狼2/封神第一部"，把数字拆到 playback.series：
     - "战狼2" → title:"战狼", playback.series:2
     - "封神第一部" → title:"封神", playback.series:1
     - 但如果数字是作品名本身的一部分（如"2012/1917"）则保留。
  13. voiceStartPos 转换：
     - "第N分钟/N分" → voiceStartPos: N*60
     - "N分M秒" → voiceStartPos: N*60+M
     - "第N秒" → voiceStartPos: N
  14. 只用下面清单中的字段，不要编造字段名。
  15. "全集"不等于 is_over=1，不要加 is_over。只有"已完结/连载中/没更完"才用 is_over。
  16. "评分高/高分/豆瓣评分高" → sort:rate:desc（不要加 rate 范围！range 只用于"X分以上/X到Y分"）。
      反之，"7分以上/高于8分" → 用 rate range，**不加 sort**。
  17. 知名公司/厂牌字段选择：
      - "笑果文化" → comedy_brand:"笑果文化"（不是 company）
      - "XX小说改编" → writer:XX
      - "好莱坞" → company:"好莱坞"（不是 area！好莱坞是制片体系不是地区）
      - "湖南卫视/山东卫视/中央一台" → channel（不是 company）
      - "TVB" → company:"tvb"（不是 vender_name）
  18. "韩剧/美剧/日剧/泰剧/港剧" → area + category:"电视剧"（"韩剧"=韩国+电视剧，不是 tag）
  19. relate 工具（"类似XX的/像XX的/跟XX相近的"）→ 必须把作品名提取到 title。
  20. "请播/播" = "播放" = 强 play 动词。"生万物第3集第10分钟" 有明确片名+集数→play。
  21. 多值 operator：
      - "A和B/A、B" → operator:"and"
      - "A或B/A还是B" → operator:"or"
      - 同一字段多值默认 and（除非逻辑上是选择关系如地区"港台"→or）
  22. playback 字段区分——**极其重要**：
      - "第N集/第N期" → video_index:N
      - "第N季/第N部" → series:N
      - "第N分钟/N分/N分M秒" → voiceStartPos:秒数（N×60 或 N×60+M）
      - 不要混淆！"第3集第10分钟" = video_index:3 + voiceStartPos:600
      - "第一部第34分" = series:1 + voiceStartPos:2040
  23. "专题片/纪实/实录" 是描述词，不是 category。后面跟的是 title。
  24. "电视台/央视/卫视" 泛指时不填 channel（只有具体频道名如"湖南卫视"才填）。
  25. "最近/新上的/新出的" → sort:new:desc 就够，不要加 release_time range。
  26. "方言" → language:"方言"（不是 tag）。"情景喜剧" → tag:"情景喜剧"（不是 category）。
  27. "XX节目" 中"节目"暗示 category:"综艺"（如"脱口秀节目"→综艺+tag:脱口秀）。

{domain} 可用字段：
{field_list}
{sort_hint}
只输出 JSON。"""

# ===========================================================================
# Audio / Device slot-fill
# ===========================================================================
AUDIO_SYSTEM = """你是电视端有声内容助手。根据用户请求填写有声搜索/播放参数。只输出 JSON。

输出：{{"tool":"audio_search|audio_chat_qa", "query":"核心内容", "play_mode":"search|play|screen_off_play", "screen_mode":"normal|screen_standby"}}

规则：query 提取核心搜索内容；play_mode 按用户动词判定；当前在播内容问答用 audio_chat_qa。"""

DEVICE_SYSTEM = """你是电视端设备控制助手。根据用户请求填写设备控制参数。只输出 JSON。

输出：{{"tool":"<工具名>", "operation":"操作类型", "object":"控制对象", "value":"参数值"}}

设备 tool 表：volume_control|power_control|signal_source_control|screen_display_control|camera_control|network_control|bluetooth_control|input_lang_control|image_quality_control|sound_mode_control|projection_control|media_center_control|demo_control|personalization_control|scene_mode_control|screen_safety_control|playback_control|ambient_light_control|system_settings_control|ai_picture_sound_control

operation 归一：调高/大点→提高；调低/小点→降低；打开/开启→打开；关闭/关掉→关闭；调到/设为→设置。"""


# ===========================================================================
# Prompt 构造函数
# ===========================================================================
def _render_fewshot(query: str, obj: dict) -> str:
    return f"用户：{query}\n→ {json.dumps(obj, ensure_ascii=False)}"


def route_system_prompt(vod_only: bool = False) -> str:
    """路由 system prompt。vod_only=True 时只输出影视域 prompt（评测用）。"""
    base = ROUTE_SYSTEM_VOD if vod_only else ROUTE_SYSTEM
    shots = _ROUTE_FEWSHOTS_VOD if vod_only else _ROUTE_FEWSHOTS
    shot_text = "\n\n".join(_render_fewshot(q, r) for q, r in shots)
    return base + "\n\n# 示例：\n" + shot_text


def route_observation(query: str, memory_hint: str = "") -> str:
    """路由阶段 user observation。"""
    obs = f"用户请求：{query}"
    if memory_hint:
        obs += f"\n上下文：{memory_hint}"
    obs += "\n\n输出路由 JSON。"
    return obs


def ir_observation(query: str, domain: str, memory_hint: str = "",
                   few_shots: list[tuple[str, dict]] | None = None,
                   intent: str | None = None,
                   use_experience_bank: bool = True) -> str:
    """IR 阶段 user observation = IR_SYSTEM + few-shot + 请求。

    Args:
        use_experience_bank: 是否注入 Experience Bank prompt 层规则。
            False 时只保留基础 IR 结构说明 + 字段列表，不注入领域规则和 few-shot。
            用于对比实验或当规则通过 RAG 按需注入时。
    """
    sort_keys = sort_keys_for_domain(domain)
    sort_hint = f"排序键：{', '.join(sort_keys)}" if sort_keys else ""
    system = IR_SYSTEM.format(
        domain=domain,
        field_list=_field_catalog(domain),
        sort_hint=sort_hint,
    )
    obs = system

    if use_experience_bank:
        shots = few_shots or _IR_FEWSHOTS.get(domain, [])
        if shots:
            shot_text = "\n\n".join(_render_fewshot(q, ir) for q, ir in shots)
            obs += "\n\n# 示例：\n" + shot_text

    obs += f"\n\n用户请求：{query}"
    if memory_hint:
        obs += f"\n上下文：{memory_hint}"
    if intent and "play" in intent:
        obs += "\n（路由判定为播放意图，action 填 play。只有用户显式说了集数才加 playback。）"
    elif intent and intent == "search":
        obs += "\n（路由判定为搜索意图，action 填 search。不要加 playback。）"
    obs += "\n输出 IR JSON。"
    return obs


def audio_observation(query: str, memory_hint: str = "") -> str:
    """有声域 slot-fill observation。"""
    obs = AUDIO_SYSTEM
    shots = "\n\n".join(_render_fewshot(q, r) for q, r in _AUDIO_FEWSHOTS)
    if shots:
        obs += "\n\n# 示例：\n" + shots
    obs += f"\n\n用户请求：{query}"
    if memory_hint:
        obs += f"\n上下文：{memory_hint}"
    obs += "\n输出 JSON。"
    return obs


def device_observation(query: str, memory_hint: str = "") -> str:
    """设备域 slot-fill observation。"""
    obs = DEVICE_SYSTEM
    shots = "\n\n".join(_render_fewshot(q, r) for q, r in _DEVICE_FEWSHOTS)
    if shots:
        obs += "\n\n# 示例：\n" + shots
    obs += f"\n\n用户请求：{query}"
    if memory_hint:
        obs += f"\n上下文：{memory_hint}"
    obs += "\n输出 JSON。"
    return obs


def repair_observation(errs: list[str]) -> str:
    """IR 校验失败时回灌 observation。"""
    return "IR 有误，请修正后重新输出 JSON：\n- " + "\n- ".join(errs)


# ===========================================================================
# 辅助
# ===========================================================================
def _field_catalog(domain: str) -> str:
    lines = []
    for f in fields_for_domain(domain):
        tag = {Kind.EXACT: "精确", Kind.STATUS: "状态", Kind.RANGE: "范围"}[f.kind]
        lines.append(f"  {f.canonical}[{tag}]: {f.desc}")
    return "\n".join(lines)


# ===========================================================================
# Few-shots —— 路由
# ===========================================================================
def _r(domain: str, intent: str, tool: str, conf: float = 0.9) -> dict:
    return {"domain": domain, "intent": intent, "tool": tool, "confidence": conf}


# 影视单域路由 few-shot（精简，覆盖边界）
_ROUTE_FEWSHOTS_VOD: list[tuple[str, dict]] = [
    # search
    ("周润发的动作电影", _r("vod", "search", "vod_search")),
    ("NHK纪录片", _r("vod", "search", "vod_search")),
    ("高清欧美科幻电影", _r("vod", "search", "vod_search")),
    ("获得百花奖最佳女主角的电影", _r("vod", "search", "vod_search")),
    ("热门综艺", _r("vod", "search", "vod_search")),
    # play — 有具体片名
    ("播放甄嬛传第二季第三集", _r("vod", "play", "vod_search")),
    ("放琅琊榜", _r("vod", "play", "vod_search")),
    ("我想看京剧霸王别姬", _r("vod", "play", "vod_search")),
    ("我要看速度与激情第30分钟", _r("vod", "play", "vod_search")),
    ("收看浙江卫视的奔跑吧兄弟", _r("vod", "play", "vod_search")),
    # search — 有动词但无具体片名（纯条件筛选）
    ("打开付费的电视剧", _r("vod", "search", "vod_search")),
    ("播放科幻电视剧", _r("vod", "search", "vod_search")),
    ("我想看美剧", _r("vod", "search", "vod_search")),
    ("我要看免费的电视剧", _r("vod", "search", "vod_search")),
    ("看热门高分电影", _r("vod", "search", "vod_search")),
    # search（有集数但无播放动词→search）
    ("搜下庆余年第二季", _r("vod", "search", "vod_search")),
    # search（"找/搜索" → search）
    ("免费的武侠剧天龙八部", _r("vod", "search", "vod_search")),
    # slow_search
    ("骑着白马的王子的电影", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("西游记86版的", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("适合情侣看的电影", _r("vod", "slow_search", "vod_slow_search_data_search")),
    # relate
    ("类似流浪地球的电影", _r("vod", "relate", "vod_relate_search")),
    # personalized
    ("根据我的喜好推荐电影", _r("vod", "personalized", "vod_personalized_search")),
    # history
    ("我最近看了什么电视剧", _r("vod", "history", "vod_history")),
]

# 全域路由 few-shot
_ROUTE_FEWSHOTS: list[tuple[str, dict]] = [
    # vod
    ("刘德华的免费电影", _r("vod", "search", "vod_search")),
    ("播放甄嬛传第二季第三集", _r("vod", "play", "vod_search")),
    ("骑着蓝色大鸟的人的电影", _r("vod", "slow_search", "vod_slow_search_data_search")),
    ("类似流浪地球的电影", _r("vod", "relate", "vod_relate_search")),
    ("给我推荐点好看的", _r("vod", "personalized", "vod_personalized_search")),
    ("最近看过什么", _r("vod", "history", "vod_history")),
    ("最好看的少儿动画", _r("vod", "search", "vod_search")),
    # educ
    ("拼音启蒙儿歌", _r("educ", "search", "educ_search")),
    # audio
    ("播放三国演义评书", _r("audio", "audio_play", "audio_search")),
    ("熄屏听睡前故事", _r("audio", "audio_screen_off_play", "audio_search")),
    # device
    ("声音大一点", _r("device", "device_control", "volume_control")),
    ("暂停", _r("device", "device_control", "playback_control")),
    ("切换到HDMI1", _r("device", "device_control", "signal_source_control")),
]

# ===========================================================================
# Few-shots —— IR（覆盖易错模式）
# ===========================================================================
_IR_FEWSHOTS: dict[str, list[tuple[str, dict]]] = {
    "vod": [
        # 基础 search（注意 "我想看" → action 由路由决定，这里演示 search）
        ("我想看周星驰的免费喜剧",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "actor", "value": "周星驰"},
              {"field": "category", "value": "电影"},
              {"field": "tag", "value": "喜剧"},
              {"field": "fee", "value": 0}]}}),
        # XX片 → category:电影 + tag 保持原词
        ("陈凯歌导演的抗日片",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "director", "value": "陈凯歌"},
              {"field": "category", "value": "电影"},
              {"field": "tag", "value": "抗日"}]}}),
        # tag 保持原词
        ("内地言情剧",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "area", "value": "内地"},
              {"field": "category", "value": "电视剧"},
              {"field": "tag", "value": "言情"}]}}),
        # play（有明确"播放"动词 + 集数）
        ("播放琅琊榜第一季第五集",
         {"domain": "vod", "action": "play",
          "query": {"field": "title", "value": "琅琊榜"},
          "playback": {"series": 1, "video_index": 5}}),
        # play 无集数（不加 playback！）
        ("放甄嬛传",
         {"domain": "vod", "action": "play",
          "query": {"field": "title", "value": "甄嬛传"}}),
        # "我想看" + 具体片名 → play（由路由决定）
        ("我想看京剧贵妃醉酒",
         {"domain": "vod", "action": "play",
          "query": {"and": [
              {"field": "title", "value": "贵妃醉酒"},
              {"field": "tag", "value": "京剧"}]}}),
        # 戏曲子类型 → tag 不加 category:戏曲
        ("请播放豫剧花木兰",
         {"domain": "vod", "action": "play",
          "query": {"and": [
              {"field": "title", "value": "花木兰"},
              {"field": "tag", "value": "豫剧"}]}}),
        # 相声/小品 → tag 不加 category:综艺
        ("相声满腹经纶",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "tag", "value": "相声"},
              {"field": "title", "value": "满腹经纶"}]}}),
        # XX版 → tag 不是 language
        ("院线版的唐人街探案",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "title", "value": "唐人街探案"},
              {"field": "tag", "value": "院线版"}]}}),
        # 多值 OR（港台 → 香港+台湾）
        ("日韩爱情电影",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "area", "values": ["日本", "韩国"], "op": "or"},
              {"field": "category", "value": "电影"},
              {"field": "tag", "value": "爱情"}]}}),
        # 多值 tag
        ("免费的动作冒险电影",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "category", "value": "电影"},
              {"field": "tag", "values": ["动作", "冒险"], "op": "and"},
              {"field": "fee", "value": 0}]}}),
        # 多排除 → not(or(...))
        ("电视剧，不要古装、不要韩国",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "category", "value": "电视剧"},
              {"not": {"or": [
                  {"field": "tag", "value": "古装"},
                  {"field": "area", "value": "韩国"}]}}]}}),
        # sort 示例
        ("免费又热门的电影",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "category", "value": "电影"},
              {"field": "fee", "value": 0}]},
          "sort": [{"key": "hot", "order": "desc"}]}),
        # sort new
        ("最新上映的悬疑电视剧",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "category", "value": "电视剧"},
              {"field": "tag", "value": "悬疑"}]},
          "sort": [{"key": "new", "order": "desc"}]}),
        # title 数字拆分
        ("给我放变形金刚3",
         {"domain": "vod", "action": "play",
          "query": {"field": "title", "value": "变形金刚"},
          "playback": {"series": 3}}),
        # 角色名 → role 字段
        ("演孙悟空的电视剧",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "role", "value": "孙悟空"},
              {"field": "category", "value": "电视剧"}]}}),
        # voiceStartPos 转换
        ("我要看复仇者联盟第40分钟",
         {"domain": "vod", "action": "play",
          "query": {"field": "title", "value": "复仇者联盟"},
          "playback": {"voiceStartPos": 2400}}),
        # 不要推断 category
        ("最新一期的快乐大本营",
         {"domain": "vod", "action": "search",
          "query": {"field": "title", "value": "快乐大本营"},
          "sort": [{"key": "new", "order": "desc"}]}),
        # 多演员 and
        ("成龙、李连杰合作的电影",
         {"domain": "vod", "action": "search",
          "query": {"and": [
              {"field": "actor", "values": ["成龙", "李连杰"], "op": "and"},
              {"field": "category", "value": "电影"}]}}),
        # 第N集 + 第N分钟 区分
        ("看三体第5集第20分钟",
         {"domain": "vod", "action": "play",
          "query": {"field": "title", "value": "三体"},
          "playback": {"video_index": 5, "voiceStartPos": 1200}}),
    ],
    "educ": [
        ("免费的英语启蒙动画",
         {"domain": "educ", "action": "search",
          "query": {"and": [
              {"field": "children_second_genre", "value": "启蒙"},
              {"field": "language", "value": "英语"},
              {"field": "fee", "value": 0}]}}),
        ("4到7岁的数学动画",
         {"domain": "educ", "action": "search",
          "query": {"and": [
              {"field": "children_second_genre", "value": "数学"},
              {"field": "age", "range": {"from": 4, "to": 7}}]}}),
    ],
}

# ===========================================================================
# Few-shots —— Audio / Device
# ===========================================================================
_AUDIO_FEWSHOTS: list[tuple[str, dict]] = [
    ("播放三国演义评书",
     {"tool": "audio_search", "query": "三国演义评书", "play_mode": "play"}),
    ("熄屏听睡前故事",
     {"tool": "audio_search", "query": "睡前故事", "play_mode": "screen_off_play", "screen_mode": "screen_standby"}),
    ("这本书讲什么",
     {"tool": "audio_chat_qa", "query": "这本书讲什么", "play_mode": "search"}),
]

_DEVICE_FEWSHOTS: list[tuple[str, dict]] = [
    ("声音大一点", {"tool": "volume_control", "operation": "提高", "object": "音量"}),
    ("把音量调到30", {"tool": "volume_control", "operation": "设置", "object": "音量", "value": "30"}),
    ("关机", {"tool": "power_control", "operation": "关闭", "object": "电源"}),
    ("切换到HDMI1", {"tool": "signal_source_control", "operation": "切换", "object": "信号源", "value": "HDMI1"}),
    ("暂停", {"tool": "playback_control", "operation": "关闭", "object": "播放"}),
    ("亮度调到50", {"tool": "image_quality_control", "operation": "设置", "object": "亮度", "value": "50"}),
]
