"""Prompt 模板：意图拆分 + 路由阶段 + IR 生成阶段 + 有声/设备 slot-fill 阶段。

设计要点：
  * 意图拆分：判断用户请求是否包含多个独立意图，拆分为子请求。
  * 路由：显式判定树（域 -> 意图大类），输出结构化并受 schema 约束。
    - 支持全域路由（4 域）和影视单域路由（只输出 vod 工具）。
  * IR 生成：只注入选中 domain 的字段清单，few-shot 覆盖易错模式。
  * 有声 slot-fill：填 query + play_mode。
  * 设备 slot-fill：选工具 + 填 operation/object/value。
"""
from __future__ import annotations

import json

from registry import Kind, fields_for_domain, sort_keys_for_domain, DEVICE_TOOLS
from fewshot_loader import load_fewshots

# ===========================================================================
# 意图拆分 —— 多意图检测与分解
# ===========================================================================
INTENT_SPLIT_SYSTEM = """你是电视端 AI 请求分析器。判断用户请求是否包含多个**独立**意图，如果是则拆分为子请求。

判定规则：
  1. **独立意图**：两个子请求之间没有依赖关系，可以并行执行。
     - "调亮屏幕同时搜刘德华的电影" → 两个独立意图：设备控制 + 影视搜索
     - "把声音调大然后播放琅琊榜" → 两个独立意图：音量控制 + 播放
     - "帮我关灯再搜个恐怖片" → 两个独立意图：设备控制 + 搜索
  2. **单一意图**：只有一个请求目标（即使描述很长/条件很多）。
     - "刘德华和梁朝伟的免费动作电影" → 单意图（多条件搜索）
     - "播放琅琊榜第二季第三集" → 单意图（带集数播放）
     - "最近有什么好看的韩剧" → 单意图
     - "类似甄嬛传的宫廷剧" → 单意图
  3. **连接词信号**（帮助判断多意图）：
     - "同时/然后/再/还要/另外/顺便/并且" + 跨域操作 → 多意图
     - "而且/并且" + 同域不同操作 → 可能多意图
  4. **不拆的情况**：
     - 条件叠加（"免费的高分港台动作片"）→ 不拆，是一个搜索
     - 修饰关系（"把声音调到适合看恐怖片的程度"）→ 不拆，是一个设备控制
     - 先后顺序的单一目标（"搜一下庆余年然后播放第一集"）→ 不拆，是搜+播一个流程

拆分时保持每个子请求语义完整、自包含，能独立理解。
只输出 JSON。"""

def intent_split_system_prompt() -> str:
    """意图拆分 system prompt。"""
    shots = load_fewshots("intent_split")
    shot_text = "\n\n".join(_render_fewshot(q, r) for q, r in shots)
    return INTENT_SPLIT_SYSTEM + "\n\n# 示例：\n" + shot_text


def intent_split_observation(query: str, memory_hint: str = "") -> str:
    """意图拆分 user observation。"""
    obs = f"用户请求：{query}"
    if memory_hint:
        obs += f"\n上下文：{memory_hint}"
    obs += "\n\n判断是否多意图并输出 JSON。"
    return obs


# ===========================================================================
# 路由 —— 全域版（vod/educ/audio/device）
# ===========================================================================
ROUTE_SYSTEM = """你是电视端 AI 助手的工具路由器。根据用户请求判断应调用哪个工具。只输出 JSON。

第一步 判断领域 domain：
  - device：设备控制——音量、电源、信号源、屏幕、摄像头、网络、蓝牙、画质、声音模式、投影、播放控制等。
  - audio：有声内容——听书、评书、相声、广播剧、有声故事等音频专辑（不含歌曲/音乐）。
  - educ：少儿/儿童向内容——**动画片/动漫/卡通/动画电影一律算 educ**（无论是否点明年龄），
    以及儿歌、绘本、点读、启蒙/早教、认知、亲子、幼儿/宝宝/小朋友内容；
    含具体动画 IP（如 汪汪队/小猪佩奇/超级飞侠/奥特曼/宝宝巴士/熊出没/芭比/艾莎公主/小羊肖恩 等）。
  - vod：真人影视——电影/电视剧/综艺/纪录片/戏曲等**非动画**内容。拿不准且非动画时默认 vod。

第二步 按 domain 判断 intent + tool：

domain=device → intent=device_control，tool 从设备工具表选。
domain=audio → intent∈{audio_search, audio_play, audio_screen_off_play, audio_chat_qa}，tool=audio_search|audio_chat_qa。
domain=educ → 见下方 educ 判定规则。
domain=vod → 见下方 vod 判定规则。

=== educ intent 判定规则（与 vod 同构）===
【search】有具体动画名/明确条件（年龄/语种/免费/分类/集数）或"播放XX动画" → tool=educ_search
  - "播放汪汪队第2集"、"我要看宝宝巴士"、"适合2岁小孩看的卡通"、"免费的英语启蒙动画"
【slow_search】模糊/语义描述、个性化推荐(根据我的喜好/按我喜好)、剧情/台词出处 → tool=educ_slow_search_data_search
  - "有没有亲子温情的动画视频"、"讲述两只熊和伐木工搞笑打闹的动画片"
  - "根据我的喜好推荐动画片"、"推荐我喜欢的动漫"、"适合我看的少儿动画"
  - "按我家娃的喜好推荐"、"找几部精准我喜好的少儿作品"
  注意：个性化推荐（我的喜好/我喜欢的/符合我偏好）→ slow_search（不是 relate）
【relate】**必须有参考作品名**："像XX那样的/类似XX的/和XX同一类型的动画" → tool=educ_relate_recommend
  - "有没有像小猪佩奇那样语速慢的英文动画"、"和消防员山姆同一类型的动画片"
  - 判断标准：必须提到一个具体作品名作为参考基准
【history】少儿观看/收藏历史（"回到刚才看的绘本/历史收藏里的XX"）→ tool=educ_history

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
【slow_search】需语义理解、无法直接结构化的模糊检索 → tool=vod_fuzzy_search：
  - 场景/情绪/人群/节日推荐
  - 版本词/画质词（如"八三版""修复版""高清版"——标签不支持的）
  - 剧情/台词/名场面出处查询
  - 剧集问答（集数/更新状态/票房）、待播/时效
  - 复杂模糊描述
  - **间接/关系引用**：通过人物关系描述（XX的老公/老婆/儿子/女儿/师傅/同学/搭档/前妻/妈妈/爸爸/哥哥/弟弟）来指代演员/导演/人物，需要知识推理才能确定具体人名
    - "孙俪老公的电影"、"黄晓明前妻的电视剧"、"成龙儿子的片"、"周杰伦老婆演的"
【search】其余可直接检索的明确条件（默认） → tool=vod_search：
  - 明确名称/演员/分类/地区/年份/免费/出品方/频道/奖项
  - 热度/口碑排序

第三步 输出 {domain, intent, tool, confidence}。

设备 tool 表：numeric_adjust|power_control|timer_control|source_switch|playback_control|video_picture_save_share|mode_control|screen_layout|screen_lift_rotation|demo_control|display_control|audio_control|smart_camera|network_control|screensaver_control|common_control|solve_picture_sound_problem_control"""

# ===========================================================================
# 路由 —— 影视单域版（只评测 vod，无 audio/device/educ 干扰）
# ===========================================================================
ROUTE_SYSTEM_VOD = """你是影视内容的工具路由器。根据用户请求判断应调用哪个影视工具。只输出 JSON。

工具列表：
  vod_search — 精确检索/起播（明确标题/演员/分类/地区/年份/免费/奖项/出品方/频道等条件）
  vod_fuzzy_search — 模糊搜索（模糊/复杂/无法结构化的描述）
  vod_relate_search — 相关推荐（"类似XX的/像XX一样的"）
  vod_personalized_search — 个性化推荐（"推荐我喜欢的/根据我兴趣"）
  vod_history — 历史记录（"我看过的/最近看了什么"）

intent 判定规则：
  play — 有起播/观看意图动词（播放/放/起播/继续看/想看/要看/看/打开/收看）+ **具体片名/内容** → tool=vod_search
  search — 条件筛选/搜索（搜/搜索/找/有没有/查），或"想看/要看"+纯条件无具体片名 → tool=vod_search
  slow_search — 需语义理解的模糊检索 → tool=vod_fuzzy_search
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
  - **间接/关系引用**：通过人物关系（XX的老公/老婆/儿子/女儿/师傅/同学/搭档/前妻/妈妈/爸爸）来指代演员/导演，需知识推理
    - "孙俪老公的电影"、"黄晓明前妻的电视剧"、"成龙儿子的片"、"周杰伦老婆演的"

注意：
  - "播放/放/看/想看/要看/打开/收看" + **具体片名** → play
  - 动词 + 纯条件无具体片名 → search（如"播放言情剧""我要看免费电影""看免费高分综艺"）
  - "搜索/搜/找/有没有" → search
  - "适合女生看的电影" 有 gender 可结构化但标签系统未必支持 → slow_search

输出：{domain:"vod", intent, tool, confidence}"""

# ===========================================================================
# 路由 —— 少儿单域版（只评测 educ，无 vod/audio/device 干扰）
# ===========================================================================
ROUTE_SYSTEM_EDUC = """你是少儿/儿童内容的工具路由器。根据用户请求判断应调用哪个少儿工具。只输出 JSON。

工具列表：
  educ_search — 精确检索/起播（明确动画名/年龄/语种/免费/分类/集数等条件）
  educ_slow_search_data_search — 慢链路语义搜索（模糊/语义描述、个性化推荐(我的喜好)、剧情出处）
  educ_relate_recommend — 相关推荐（"像XX那样的/类似XX的/和XX同一类型的动画"，必须有参考作品名）
  educ_history — 历史记录（"回到刚才看的绘本/历史收藏里的XX"）

intent 判定规则：
  play — 有起播/观看意图动词（播放/放/起播/继续看/想看/要看/看/打开）+ **具体动画名/内容名** → tool=educ_search
    - "播放汪汪队立大功第2集" → play
    - "我要看宝宝巴士" → play
  search — 条件筛选/搜索，或"想看/要看"+纯条件无具体名 → tool=educ_search
    - "适合2岁小孩看的卡通"、"免费的英语启蒙动画"、"最好看的少儿动画"
  slow_search — 需语义理解的模糊检索 / 个性化推荐 → tool=educ_slow_search_data_search
    - "有没有亲子温情的动画视频"、"讲述两只熊和伐木工搞笑打闹的动画片"
    - 个性化推荐（我的喜好/推荐我喜欢的/按我家娃喜好）→ slow_search
  relate — 找相似内容（**必须有参考作品名**） → tool=educ_relate_recommend
    - "有没有像小猪佩奇那样语速慢的英文动画"、"和消防员山姆同一类型的动画片"
  history — 查观看/收藏历史 → tool=educ_history

play vs search 关键区分：
  - "播放汪汪队立大功第2集" → play（有具体动画名+集数）
  - "我要看宝宝巴士" → play（有具体动画名）
  - "播放小羊肖恩大电影" → play（有具体动画名）
  - "适合2岁看的卡通" → search（无具体动画名，只有条件）
  - "我要看免费的动画" → search（无具体动画名，只有条件）

slow_search 命中条件（任一即 slow_search）：
  - 场景/情绪/亲子推荐
  - 剧情/内容描述（"讲述XX的动画"）
  - 个性化推荐（我的喜好/推荐我喜欢的/根据我兴趣/按我家娃喜好）

注意：
  - 动画/动漫/卡通/动画电影一律在 educ 域
  - 个性化推荐（"根据我的喜好推荐动画片"）→ slow_search（不是 relate）
  - relate 必须提到一个具体作品名作为参考基准

输出：{domain:"educ", intent, tool, confidence}"""


# ===========================================================================
# IR 生成 —— 影视 (vod) 专用
# ===========================================================================
IR_SYSTEM_VOD = """你是影视媒资查询解析器。把用户请求解析成布尔查询 IR。只产出 IR JSON，后续由编译器落地。

IR 结构：
{{
  "domain": "vod",
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

vod 可用字段：
{field_list}
{sort_hint}
只输出 JSON。"""


# ===========================================================================
# IR 生成 —— 少儿 (educ) 专用
# ===========================================================================
IR_SYSTEM_EDUC = """你是少儿媒资查询解析器。把用户请求解析成少儿布尔查询 IR。只产出 IR JSON，后续由编译器落地。

**注意：少儿域的 IR 结构与影视(vod)不同，不要混用！**

IR 结构：
{{
  "query": <节点>,
  "sort": {{"rate":{{"order":"desc"}}}},  // 可选，对象格式，键=rate|hot|new
}}

**无 domain、无 action、无 playback 字段！**

节点形态：
  {{"and":[...]}} / {{"or":[...]}} / {{"not":节点}}
  {{"field":"F","value":"X"}}
  {{"field":"F","values":["X","Y"],"op":"and|or"}}  // 同字段多值
  {{"field":"is_fee","value":0|1}}  // 状态字段
  {{"field":"age_range","from":"3","to":"6"}}  // 年龄范围（三种模式："3 TO 6" / "* TO 6" / "6 TO *"）
  {{"field":"release_year","from":"20200101","to":"20231231"}}  // 时间范围，yyyyMMdd

关键规则：
  1. **无 action/playback**：少儿域不区分 search/play，不使用 series/video_index/voiceStartPos。
     即使用户说"播放XX"，也只输出 query（不加 action）。
  2. **季/部/集 处理**：
     - "XX第N季" → AND[title:"XX", title:"第N季"]
     - "XX第N部" → AND[title:"XX", title:"第N部"]
     - "XX第N集" → title:"XX"（**忽略集数**，不要把"第N集"放进 title）
     - "汪汪队立大功第四季第2集" → AND[title:"汪汪队立大功", title:"第四季"]
     - "唐老鸭第29集" → title:"唐老鸭"（忽略集数）
     - "萌鸡小队第6集" → title:"萌鸡小队"（忽略集数）
  3. **content_type 使用规则**：当用户**明确说了"动画片/动画/卡通"**时，输出 content_type。
     - "XX动画片" → content_type:"动画"
     - "看卡通" → content_type:"卡通"
     - "凡人修仙传动画" → title:"凡人修仙传" + content_type:"动画"
     - 用户没有提到"动画/动画片/卡通"时，不要自行添加 content_type。
     - **"动漫"→ content_type:"动画"**（不是 children_second_genre）。
  4. **role 字段——极其谨慎使用**：
     - educ 中 **绝大多数情况只用 title**，不要加 role。
     - "播放光头强" → title:"光头强"（不加 role）
     - "播放丑小鸭" → title:"丑小鸭"（不加 role）
     - "播放奥特曼" → title:"奥特曼"（不加 role）
     - 原则：**如果不确定是否要 role，只用 title**。
  5. **语言值归一**：英语/英文版 → "英文"；中文/中文版/国语 → "普通话"。
  6. **genre 层级选择**：
     - children_second_genre 用于**大分类**：儿歌、故事、科普、绘本、数学、玩具、动画电影、音乐
     - children_third_genre 用于**具体题材/风格**：科幻、冒险、搞笑、魔法、恐龙、益智、古诗词、情绪管理、汉字、热血、校园、治愈
     - "启蒙" → 不要用（除非用户只说"启蒙动画"且没有其他可提取的关键词）
  7. **title 抽取规则**：
     - title 是**作品名/IP 名**，不是描述词。保持用户说的完整名称。
     - ✓ "播放宝宝巴士奇妙学古诗" → title:"宝宝巴士奇妙学古诗"（完整保留）
     - ✓ "播放土豆逗严肃科普" → title:"土豆逗严肃科普"（完整保留）
     - ✓ "消防车动画片" → title:"消防车"
     - ✗ "免费高分动画" → 不加 title（"动画"不是作品名）
     - ✗ "热血动漫" → 不加 title（"热血"是题材→children_third_genre）
     - 判断标准：该词能否作为一部具体动画作品的名字？不能→不放 title。
     - **JOJO 系列统一**：宝贝JOJO/超级宝贝JOJO → title:"超级宝贝jojo"（小写）
  8. **"全集" 不影响 title**：
     - "迷你特工队全集" → title:"迷你特工队全集"（全集是标题的一部分保留）
     - "XX大电影" → title:"XX大电影"（大电影是标题的一部分保留，不拆）
  9. **免费/不要VIP** → is_fee:0（状态字段，0=免费，1=付费）。
  10. **公司/品牌 → company**：
     - "迪士尼" → company:"迪士尼"（不是 title）
     - "优酷/酷喵/哔哩哔哩" → company:"优酷"等（不是 title）
  11. **"国产" → country**：
     - "国产动画" → country:["内地","中国澳门","中国香港","中国台湾"]（不要写 country:"中国"）
  12. **release_year 范围**：格式 yyyyMMdd。
     - "2023年" → from:"20230101", to:"20231231"
     - "XX年以后" → from:"XX0101", to:"*"
  13. **sort 规则**（少用；只在用户**明确表达排序意图**时才加）：
     - "好看的/好评/评分高" → sort:{{"rate":{{"order":"desc"}}}}
     - "最新的/新出的" → sort:{{"new":{{"order":"desc"}}}}
     - "热门/最火" → sort:{{"hot":{{"order":"desc"}}}}
     - 用户没提排序意图时**不加** sort。
  14. **排除（不要/除了）→ not**：多个排除项 → not(or([...]))。
  15. **同字段多值 → values+op**：
     - "科幻冒险" → children_third_genre: values:["科幻","冒险"], op:"and"
     - 同一字段多值默认 and（除非逻辑上是选择关系→or）
  16. 只用下面清单中的字段，不要编造字段名。

educ 可用字段（16 精确 + 1 状态 + 2 范围）：
{field_list}
只输出 JSON。"""

# ===========================================================================
# Audio / Device slot-fill
# ===========================================================================
AUDIO_SYSTEM = """你是电视端有声内容助手。根据用户请求填写有声搜索/播放参数。只输出 JSON。

输出：{{"tool":"audio_search|audio_chat_qa", "query":"核心内容", "play_mode":"search|play|screen_off_play", "screen_mode":"normal|screen_standby"}}

规则：query 提取核心搜索内容；play_mode 按用户动词判定；当前在播内容问答用 audio_chat_qa。"""

DEVICE_SYSTEM = """你是电视端设备控制助手。根据用户请求选择设备工具并填写参数。只输出 JSON。

输出格式：
  常规控制：{{"tool":"<工具名>", "operation":"操作", "object":"控制对象", "value":"参数值", "date_time":"定时时间"}}
    - value：无参数值时不填（如"打开背光分区""关机"）。
    - date_time：仅定时开关机/熄屏场景填，如"30分钟""22:00"；否则不填。
  音画问题（用户描述"效果不对/不满意，帮我调"类画质或音效问题）：
    {{"tool":"solve_picture_sound_problem_control", "intent":"<诊断意图>"}}  只填 intent。

=== 17 个工具怎么选（按语义匹配）===
- numeric_adjust 幅值控制：音量/低音/清晰度/对比度/色度/亮度/氛围灯亮度/分辨率/倍速/刷新率 的提高/降低/设置/查询。
    "声音大一点""音量调到30""亮度高一点""对比度设为50"。注意"声音/音量大小"→object填"音量"。
- power_control 开关机：开机、关机、重启（**无定时**）。
- timer_control 定时控制：**带时间**的定时开机/关机/熄屏、查剩余关机时长。有 date_time 就用它。
- source_switch 信号源切换：切到 HDMI/typec/usb/VGA/机顶盒/数字电视/switch/ps5 等。
- playback_control 播放控制：播放/停止/退出/重播/快进/快退/跳转/上一个/下一个/循环/随机播放。
- video_picture_save_share：视频或图片的保存/删除/分享(微信/抖音)。
- mode_control 模式控制：图像模式/声音模式/音效模式/混响模式/护眼模式/整机模式 的**切换**（带模式名）。
- screen_layout 画面布局：分屏/全屏/小屏/画面放大缩小。
- screen_lift_rotation：横竖屏切换、屏幕旋转、支架升降、竖屏短视频。
- demo_control 演示：关键词含 demo/演示/展示/体验。
- display_control 画质背光**开关**：打开/关闭具名画质或背光功能（背光分区/全程HDR/运动补偿/MiniLED/低蓝光护眼等）。
- audio_control 音质音效**开关**：打开/关闭具名音质或音效功能（杜比全景声/DTS/waves音效/立体音效/电视音频增强等）。
- smart_camera：摄像头/雷达/AI视觉相关（全景云台摄像头/智慧之眼/坐姿检测/AI云健身等）。
- network_control 网络：无线/有线网络、WiFi、热点、网络测速。WiFi/无线网络设置→object常填"网络设置"或"无线网络"。
- screensaver_control 屏保：屏保/壁纸/熄屏/亮屏/待机/睡眠/ai壁纸。
- common_control 兜底：不属于以上的其他开关——主页/设置/蓝牙/语言/输入法/主题/系统更新/恢复出厂/媒体中心/私有云/商场模式设置/文件导入等。
- solve_picture_sound_problem_control：用户**描述音画问题**（颜色太浓/太淡/偏色/暗部发黑/不清晰/噪点/运动不流畅/不护眼/人声听不清/没环绕感/低音差）→ 填 intent。

=== 参数约定（重要）===
- 关机/开机：power_control，operation="打开"，object="关机"或"开机"（"关掉电视/把电视关了"都→object="关机"）。
- 定时："30分钟后关机"→timer_control operation="打开" object="关机" date_time="30分钟"；"晚上10点关机"→date_time="22:00"（时间转24小时制）。
- 信号源："切换到HDMI1"→source_switch operation="设置" object="信号源" value="HDMI1"。
- 快进快退："快进10秒"→playback_control operation="快进" value="10秒"（**不填object**）。
- 模式："声音模式切换到影院模式"→mode_control operation="设置" object="声音模式" value="影院模式"。
- display_control/audio_control/demo_control/smart_camera/screensaver_control/common_control 的 object 直接填用户说的功能名（如"背光分区""waves音效""demo演示""智慧之眼""ai壁纸""nas私有云"），一般不填 value。
- **object/value 照抄用户原话里的功能名/信号源，逐字保留、不要增删或简写**（如"待机选项"不要写成"待机"，"安装耳机语音服务"不要写成"耳机语音服务"，"VGA模式"保留"模式"）。
- 打开/关闭具名画质或背光功能且**不含"模式"二字**→display_control（真实色彩还原/环境光感智能/自动对焦等）；含"模式"的模式切换→mode_control。
- "静音/把声音关掉/音量开关"→numeric_adjust（object 填"静音"或"音量"）。

intent 候选：colors_oversaturated(色彩偏浓) colors_washed_out(色彩偏淡) cold_color_cast(偏冷) red_color_cast(偏红) warm_color_cast(偏暖) dark_areas_crushed(暗场发黑) highlights_overexposed(亮部过曝) lack_of_clarity(不清晰) image_noise_obvious(噪点明显) motion_not_smooth(运动不流畅) motion_not_sharp(运动不清晰) poor_eye_comfort(不护眼) image_mode_movie(电影模式) flickering_brightness(亮度忽明忽暗) image_mode_auto(自动图像模式) voice_not_clear(人声听不清) lack_stereo_surround(无环绕感) bass_effect_poor(低音差)

operation 归一：调高/大点→提高；调低/小点→降低；打开/开启/启动→打开；关闭/关掉/退出→关闭；调到/设为→设置。"""


# ===========================================================================
# Prompt 构造函数
# ===========================================================================
def _render_fewshot(query: str, obj: dict) -> str:
    return f"用户：{query}\n→ {json.dumps(obj, ensure_ascii=False)}"


def route_system_prompt(vod_only: bool = False, educ_only: bool = False) -> str:
    """路由 system prompt。vod_only/educ_only=True 时只输出对应单域 prompt（评测用）。"""
    if vod_only:
        base = ROUTE_SYSTEM_VOD
        shots = load_fewshots("route", "vod")
    elif educ_only:
        base = ROUTE_SYSTEM_EDUC
        shots = load_fewshots("route", "educ")
    else:
        base = ROUTE_SYSTEM
        shots = load_fewshots("route", "full")
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
    """IR 阶段 user observation —— vod 和 educ 使用完全独立的 prompt。

    Args:
        use_experience_bank: 是否注入 Experience Bank prompt 层规则和 few-shot。
            False 时只保留基础 IR 结构说明 + 字段列表。
    """
    if domain == "educ":
        return _ir_observation_educ(query, memory_hint, few_shots, intent, use_experience_bank)
    return _ir_observation_vod(query, memory_hint, few_shots, intent, use_experience_bank)


def _ir_observation_vod(query: str, memory_hint: str = "",
                        few_shots: list[tuple[str, dict]] | None = None,
                        intent: str | None = None,
                        use_experience_bank: bool = True) -> str:
    """影视 (vod) IR prompt —— 使用 IR_SYSTEM_VOD。"""
    sort_keys = sort_keys_for_domain("vod")
    sort_hint = f"排序键：{', '.join(sort_keys)}" if sort_keys else ""
    system = IR_SYSTEM_VOD.format(
        field_list=_field_catalog("vod"),
        sort_hint=sort_hint,
    )
    obs = system

    if use_experience_bank:
        shots = few_shots or load_fewshots("ir", "vod")
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


def _ir_observation_educ(query: str, memory_hint: str = "",
                         few_shots: list[tuple[str, dict]] | None = None,
                         intent: str | None = None,
                         use_experience_bank: bool = True) -> str:
    """少儿 (educ) IR prompt —— 使用 IR_SYSTEM_EDUC，完全独立于 vod。"""
    system = IR_SYSTEM_EDUC.format(
        field_list=_educ_field_catalog(),
    )
    obs = system

    if use_experience_bank:
        shots = few_shots or load_fewshots("ir", "educ")
        if shots:
            shot_text = "\n\n".join(_render_fewshot(q, ir) for q, ir in shots)
            obs += "\n\n# 示例：\n" + shot_text

    obs += f"\n\n用户请求：{query}"
    if memory_hint:
        obs += f"\n上下文：{memory_hint}"
    obs += "\n输出 IR JSON。"
    return obs


def audio_observation(query: str, memory_hint: str = "") -> str:
    """有声域 slot-fill observation。"""
    obs = AUDIO_SYSTEM
    shots = load_fewshots("slot_fill", "audio")
    if shots:
        shot_text = "\n\n".join(_render_fewshot(q, r) for q, r in shots)
        obs += "\n\n# 示例：\n" + shot_text
    obs += f"\n\n用户请求：{query}"
    if memory_hint:
        obs += f"\n上下文：{memory_hint}"
    obs += "\n输出 JSON。"
    return obs


def device_observation(query: str, memory_hint: str = "") -> str:
    """设备域 slot-fill observation。"""
    obs = DEVICE_SYSTEM
    shots = load_fewshots("slot_fill", "device")
    if shots:
        shot_text = "\n\n".join(_render_fewshot(q, r) for q, r in shots)
        obs += "\n\n# 示例：\n" + shot_text
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


def _educ_field_catalog() -> str:
    """少儿域字段清单——使用 tool_schema 落地名（与 vod 的 canonical 分离）。"""
    # 对齐 educ_search_all tool_schema 0729-v1 的字段命名
    _EDUC_FIELDS = [
        ("title", "精确", "媒资作品标题(开放型)。作品本身的名字，如小猪佩奇、汪汪队。与 role(人物角色) 区分。"),
        ("content_type", "精确", "媒资表现形式(固定候选)：动画、卡通、动画片。仅用户明确提到时才填。"),
        ("children_second_genre", "精确", "蛛网二级分类(固定候选)：动漫、电影、绘本、儿歌、玩具、真人、音频、宠物、早教、科普、英语、国学、数学、才艺、故事、家长课堂、经典、国学启蒙、数学启蒙、动画电影、漫画、音乐。"),
        ("children_third_genre", "精确", "蛛网三级分类(固定候选)：搞笑、工程车、科幻、历史、励志、魔幻、热血、温馨治愈、益智动画、冒险、校园、古诗词、情绪管理、恐龙等。"),
        ("training_objectives", "精确", "少儿培养目标(固定候选)：宝宝学数字、识字、英语启蒙、安全教育等。"),
        ("role", "精确", "角色/人物(开放型)。极少使用，绝大多数情况只用 title。"),
        ("multiple_intelligences", "精确", "少儿八大智能(固定候选)：人际交往、空间想象、品格习惯、识物认知、数字推理、音乐节奏、肢体运动、语言沟通、学说话。"),
        ("country", "精确", "国家/地区(开放型)。与 language 区分——country 是地理概念。"),
        ("company", "精确", "出品公司(开放型)。如迪士尼、时光传媒。"),
        ("language", "精确", "语言(固定候选)：英文、普通话、日语、粤语等。"),
        ("gender", "精确", "受众性别(固定候选)：男、女、男孩、女孩等。"),
        ("festival", "精确", "节日(固定候选)：春节、儿童节、圣诞节等。"),
        ("prize", "精确", "奖项(固定候选)：凯迪克大奖、国际安徒生奖等。"),
        ("sub_prize", "精确", "子奖项(开放型)。与 prize 配合使用。"),
        ("features", "精确", "音效(固定候选)：杜比、杜比音效。"),
        ("grade", "精确", "年级/学段：小班、中班、大班、一年级~六年级。"),
        ("is_fee", "状态", "免付费标识：0=免费，1=付费。"),
        ("age_range", "范围", "年龄范围。格式：from/to 整数，如 from:\"3\" to:\"6\"。开区间用 \"*\"。"),
        ("release_year", "范围", "发布时间范围。格式：from/to yyyyMMdd，如 from:\"20200101\" to:\"20231231\"。开区间用 \"*\"。"),
    ]
    lines = []
    for name, kind, desc in _EDUC_FIELDS:
        lines.append(f"  {name}[{kind}]: {desc}")
    return "\n".join(lines)

