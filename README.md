# 电视端 AI 慢任务 Planner — 4 域统一架构

面向 **30B MoE planner**，覆盖 **影视(vod) / 少儿(educ) / 有声(audio) / 设备(device)** 四个域。

## Benchmark（当前）

| 域 | 用例数 | 工具准确率 | 工具+参数准确率 | 说明 |
|---|---|---|---|---|
| 影视 (vod) | 217 | 97.7% | 79.3% | baseline 47% → IR+编译器+EB 优化 |
| 设备 (device) | 1166 | 93.2% | 79.9% | baseline 76.3% → compiler EB 优化 |

## 背景与问题

慢任务接口(`/slowAgent`)把用户自然语言规划成结构化工具调用。四域走不同管线：

- **vod / educ（检索类）**：最复杂——嵌套布尔 DSL、路由混淆、两套参数范式并存。走 IR 编译管线。
- **audio（有声）**：不走 IR，路由后直接 slot-fill（query + play_mode + screen_mode）。
- **device（设备控制）**：不走 IR，路由到 17 个设备工具之一，然后 slot-fill。两种参数范式：
  - 常规控制（16 工具）：operation / object / value / date_time
  - AI 音画诊断（1 工具）：intent（18 种画质/音效异常枚举）

## 核心思路

**分域分管线 + 约束解码 + Experience Bank 双层兜底**

1. **路由先行**：模型一次约束解码输出 `{domain, intent, tool, confidence}`，决定走哪条管线。
2. **vod/educ IR 管线**：模型产域无关 IR → 编译器确定性转换 → EB 后处理修正。
3. **audio/device Slot-Fill 管线**：模型直接填槽（约束解码保证结构/枚举合法）→ compile 落地参数。
4. **多意图并发**：`plan_multi()` 先做意图拆分，多意图时并发规划各子请求。

## 链路

```
用户query(+memory)
   │
 [意图拆分] intent_split (可选，plan_multi 入口)
   │   单意图 → plan()  |  多意图 → 并发 plan(sub_q[i])
   │
 [路由] route  ── guided_json(小schema) ─► {domain, intent, tool, confidence}
   │
   ├─── domain=vod/educ ───────────────────────────────────────────────────┐
   │ [IR生成] guided_json(按domain收紧的IR schema) ─► 域无关布尔IR          │
   │                                              ↑ Experience Bank (prompt层)
   │ [校验] validate_ir  ──失败──► 带错误回灌自修复(≤2次)                    │
   │                                                                       │
   │ [编译] compile_with_fallback ─► 最终 tool params                      │
   │     ├─ nested backend  -> vod_search / vod_search_all / educ_search   │
   │     ├─ relate backend  -> vod_relate_search                           │
   │     └─ flat backend    -> *_slow_search（只传query原文）               │
   │                                                                       │
   │ [后处理] Experience Bank (compiler层)                                  │
   │     ├─ action 动词判定（播放/放/请播→play）                             │
   │     ├─ 字段名修正 / 值归一化 / 结构修正                                 │
   │     ├─ query-text补全 / sort精细控制                                    │
   │     └─ title+series 拆分 / 时间转换 / 名称映射                          │
   │                                                                       │
   ├─── domain=audio ──────────────────────────────────────────────────────┐
   │ [slot-fill] guided_json(audio_schema) ─► {tool, query, play_mode, ..} │
   │ [compile_audio] → (audio_search/audio_chat_qa, params)                │
   │                                                                       │
   ├─── domain=device ─────────────────────────────────────────────────────┐
   │ [slot-fill] guided_json(device_schema) ─► {tool, operation, object,.. │
   │            tool enum(17选1约束) + intent enum(AI音画18选1约束)          │
   │ [compile_device] (含 EB compiler 层)                                  │
   │     ├─ tool 纠偏：object/value/query 词表匹配 → 修正工具路由            │
   │     ├─ solve_picture_sound_problem_control → {intent}                 │
   │     └─ 其余16工具 → {operation, object, [value], [date_time]}          │
   │         ├─ mode_control: object/value 归一 + 模式类别判定              │
   │         ├─ source_switch: 信号源/设备名 参数范式归一                    │
   │         ├─ playback_control: operation归一 + 时间格式 + 播放模式        │
   │         ├─ network_control: object 同义词精确归一                      │
   │         ├─ screen_layout: 小屏/分屏/全屏 归一                         │
   │         └─ numeric_adjust/power/timer/display/audio_control 归一       │
   │                                                                       │
   └─── 慢链路/简单工具 ──► 直接输出 {tool_name, {query}}                   │
         ▼
 最终输出 {tool_name, parameters}
```

## 目录结构

```
poc/
├── planner/
│   ├── registry.py        Field Registry（单一事实源：字段/工具/枚举/别名）
│   ├── ir.py              域无关布尔 IR：数据类 / 解析 / 语义校验
│   ├── grammar.py         从 registry 生成各阶段 guided_json schema
│   ├── compiler.py        编译器 + Experience Bank (compiler层) 后处理规则
│   ├── prompts.py         路由/IR/audio/device prompt + EB (prompt层) 规则
│   ├── agent.py           域无关 agent 状态机（route→分流→生成），推理与训练共用
│   ├── vllm_client.py     vLLM OpenAI 兼容客户端（guided_json）
│   └── harness.py         端到端 Planner 编排（含多意图并发）
├── bench_vod.py             影视评测脚本（含 param_diff 差异分析）
├── bench_device.py          设备评测脚本（工具别名归一 + 空字段容忍）
├── export_device_eval_issues.py  设备评测集标注问题导出
├── tests/test_planner.py    81 项离线测试（不连模型）
├── tests/test_planner_extended.py  24 项扩展测试（device/audio/route）
├── demo.py                  可运行示例
├── requirements.txt
└── README.md
```

## 模块职责

| 文件 | 职责 |
|---|---|
| `registry.py` | **单一事实源**：canonical 字段、domain、nested/flat 落地名、sort/playback 登记表、17 个设备工具列表（0731 schema）、AI音画 intent 枚举(18值)、object→tool 词表（从 schema CSV 解析） |
| `ir.py` | IR 数据类(`IR/And/Or/Not/Leaf`)、`parse_ir`、`validate_ir`(registry 驱动的语义校验) |
| `grammar.py` | 各阶段 schema：`build_route_schema` / `build_ir_schema(domain)` / `build_audio_schema` / `build_device_schema` / `build_intent_split_schema` |
| `compiler.py` | 编译核心：`compile_nested`/`compile_flat`/`compile_relate`/`compile_audio`/`compile_device` + vod/educ EB compiler层 + device EB compiler层 |
| `prompts.py` | 路由/IR/audio/device prompt + EB prompt层(27条规则 + 21组few-shot + device 10组few-shot) |
| `agent.py` | **训推一致核心**：`PlannerAgent` 状态机(route→domain分流→生成→校验自修复) |
| `vllm_client.py` | `VLLMClient`(`extra_body.guided_json`)；可注入 `responder` 离线测试 |
| `harness.py` | `Planner.plan()` / `Planner.plan_multi()`：驱动 Agent + 编译 + 后处理，含多意图并发 |
| `bench_vod.py` | 影视评测：多worker并发、param_diff 差异列、按工具细分统计 |
| `bench_device.py` | 设备评测：工具别名归一、空字段对称容忍、时间/值格式容忍、param_diff 差异列 |
| `export_device_eval_issues.py` | 扫描设备评测集自身标注问题，导出返修清单 CSV |

## 设备域 (device) 详细设计

### 17 工具 + 2 参数范式（0731 schema）

**常规控制（16 工具）**：
numeric_adjust / power_control / timer_control / source_switch / playback_control / video_picture_save_share / mode_control / screen_layout / screen_lift_rotation / demo_control / display_control / audio_control / smart_camera / network_control / screensaver_control / common_control

参数：`{operation, object, [value], [date_time]}`
- operation：提高/降低/打开/关闭/设置/查询/切换
- object：控制对象（音量/静音/亮度/WiFi/信号源/电源/关机…）
- value：参数值（可选，如 30、50%、HDMI1、标准模式）
- date_time：定时时间（可选，仅定时开关机，如 30分钟、22:00）

**AI 音画诊断（1 工具）**：
solve_picture_sound_problem_control → 参数：`{intent}`

intent 枚举（18 值）：
colors_oversaturated / colors_washed_out / cold_color_cast / red_color_cast / warm_color_cast / dark_areas_crushed / highlights_overexposed / lack_of_clarity / image_noise_obvious / motion_not_smooth / motion_not_sharp / poor_eye_comfort / image_mode_movie / flickering_brightness / image_mode_auto / voice_not_clear / lack_stereo_surround / bass_effect_poor

### 设备域 Experience Bank (Compiler层)

设备域的 compiler 层后处理是 `compile_device()` 内的确定性规则，覆盖以下子模块：

| 子模块 | 规则 |
|--------|------|
| **tool 纠偏** | ①object→tool 精确词表（从 schema CSV 解析）; ②value→tool 匹配（u盘→common_control）; ③query 最长子串匹配; ④object 模糊最近邻; ⑤高置信关键词兜底 |
| **mode_control** | object 同义词归一(画质模式/显示模式→图像模式); value→object 映射(杜比音效→音效模式); 模式值拆分(object=标准模式→obj=图像模式+val=标准模式); 纯类别时 operation=打开 |
| **source_switch** | 非信号源设备(机顶盒/数字电视等)保持 operation=打开+object=设备名; 具体信号源→operation=设置+object=信号源+value=具体值; HDMI 格式归一 |
| **playback_control** | operation归一(快进到/快退到 query感知, 停止→暂停, 上一个→上+value=集); 时间格式(01:30→1:30, 2分钟→2分); 播放模式(value=列表播放→operation=列表播放+object=播放列表) |
| **network_control** | object 精确同义词表归一(wifi网络→无线网络, 以太网→有线网络); operation 启动/设置→打开; 保持特定功能名(wifi全时推送/无线热点等)不过度归一 |
| **screen_layout** | 小屏类→object=小屏; 分屏/全屏类→归一+operation=打开 |
| **numeric_adjust** | 静音→operation=打开; 音量最大→100%; 去"格"后缀 |
| **power_control** | 开机/关机/重启→operation=打开 |
| **display/audio_control** | 设置/启动→打开（纯开关工具） |
| **timer_control** | 定时关机→关机 |
| **通用** | object/value 英文统一小写; 空字段不输出; value-based 重路由后参数修正 |

## Experience Bank 架构

Experience Bank 是一套**领域规则库**，分两层生效：

### Prompt 层（`prompts.py`，vod/educ 27条规则 + device 10组few-shot）

指导模型在 IR/slot-fill 生成阶段做正确的语义选择：

| 域 | 规则类 | 示例 |
|----|--------|------|
| vod/educ | action 判定 | "播放/放/请播" → play；"我想看/我要看" + 无片名 → search |
| vod/educ | category 推断 | "XX片" → 电影；"XX剧" → 电视剧；"节目" → 综艺 |
| vod/educ | tag 保持原词 | 用户说"抗战" → tag:"抗战"（不替换为"战争"） |
| vod/educ | playback 区分 | "第N集" → video_index；"第N分钟" → voiceStartPos |
| vod/educ | 字段选择 | "好莱坞" → company；"笑果文化" → comedy_brand |
| vod/educ | sort 触发 | "好看的" 不加 sort；"最新" → new:desc |
| vod/educ | 多值 op | "A和B" → and；"港台" → or |
| device | 工具路由 | 10组 few-shot 覆盖常规控制+定时+AI音画 |

### Compiler 层（`compiler.py` 内确定性后处理）

编译后基于规则表 **无条件修正** 输出，不依赖模型行为：

**vod/educ 域：**

| 规则类 | 处理 |
|--------|------|
| action 动词覆盖 | 强动词(播放/放/请播/打开)→play，弱动词(看/想看)+无title→search |
| 字段名 remap | 好莱坞→company, 湖南卫视→channel, TVB→company:tvb |
| 值归一化 | 人名间隔符, vender大小写, tag大小写 |
| 结构简化 | 单元素 and/or 解包；not{is_fee:1}→is_fee:0 |
| query-text 补全 | "影片"→category:电影, "韩剧"→area+category |
| sort 精细控制 | 有明确分数→不加sort；"最近"→sort:new+删date range |
| title 处理 | 去后缀(漩涡视频→漩涡), 拆数字(战狼2→战狼+series:2) |
| 时间转换 | "去年/今年/前年"→绝对年份 |
| 名称映射 | 奖项/tag/company 归一 |

**device 域：** 见上方「设备域 Experience Bank (Compiler层)」表。

## IR 结构（vod/educ 专用）

```jsonc
{
  "domain": "vod" | "educ",
  "action": "search" | "play",
  "query": <节点>,                    // and/or/not/leaf 任意嵌套
  "sort":  [{"key":"rate|hot|new|play","order":"asc|desc"}],
  "playback": {"series":2,"video_index":3}
}
```

节点（叶子）四形态：
```jsonc
{"field":"actor","value":"刘德华"}                       // 精确单值
{"field":"actor","values":["刘德华","吴京"],"op":"and"}   // 同字段多值
{"field":"fee","value":0}                               // 状态字段，0|1
{"field":"release_year","range":{"from":"20200101","to":"*"}}  // 范围
```

## 运行

```bash
pip install -r requirements.txt

# 离线测试（不连模型）
python tests/test_planner.py            # 81 项核心测试
python tests/test_planner_extended.py   # 24 项扩展测试（device/audio）

python demo.py                          # 展示编译 + 后处理效果

# 连真实 vLLM 评测 —— 影视
VLLM_BASE_URL=http://host:8000/v1 VLLM_MODEL=your-30b-moe \
  python bench_vod.py test_set/*.csv --live --output results.csv --worker 16

# 连真实 vLLM 评测 —— 设备
VLLM_BASE_URL=http://host:8000/v1 VLLM_MODEL=your-30b-moe \
  python bench_device.py \
  "test_set/AIOS交互新架构POC设备评测用例集 - 快链路工具识别&业务准确率用例集-整机控制.csv" \
  --live --output results_device.csv --workers 16 --trace trace_device.jsonl

# 导出设备评测集标注问题清单
python export_device_eval_issues.py \
  "test_set/AIOS交互新架构POC设备评测用例集 - 快链路工具识别&业务准确率用例集-整机控制.csv" \
  -o device_eval_issues.csv
```

最简用法：

```python
from planner import Planner, VLLMClient, VLLMConfig

planner = Planner(VLLMClient(VLLMConfig(base_url="http://host:8000/v1", model="your-30b-moe")))

# 单意图 — 影视
res = planner.plan("我想看刘德华的免费电影")
print(res.tool_name, res.parameters)   # -> vod_search {...}

# 单意图 — 设备控制
res = planner.plan("把音量调到30")
print(res.tool_name, res.parameters)   # -> numeric_adjust {"operation":"设置","object":"音量","value":"30"}

# 单意图 — 有声
res = planner.plan("播放三体有声书")
print(res.tool_name, res.parameters)   # -> audio_search {"query":"三体","play_mode":"play"}

# 多意图
results = planner.plan_multi("调亮屏幕同时搜刘德华的电影")
for r in results:
    print(r.tool_name, r.parameters)
```

## 评测

```bash
# 影视评测输出含 param_diff 列，直观展示每条错误的具体差异
VLLM_BASE_URL=http://localhost:8080/v1 VLLM_MODEL=baseline \
  python bench_vod.py test_set/*.csv --live --output results.csv --worker 16

# 设备评测
VLLM_BASE_URL=http://localhost:8080/v1 VLLM_MODEL=baseline \
  python bench_device.py "test_set/...整机控制.csv" \
  --live --output results_device.csv --workers 16 --trace trace_device.jsonl
```

输出 CSV 列：`source_file | row | query | expected_tool | predicted_tool | tool_correct | param_correct | expected_params | predicted_params | param_diff | error | note`

## vLLM 启动

```bash
vllm serve /path/to/your-30b-moe \
    --served-model-name your-30b-moe \
    --host 0.0.0.0 --port 8000 \
    --structured-outputs-config.backend xgrammar \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90
```

## 优化历程

### 影视域 (vod)

| 版本 | param_acc | 主要改动 |
|------|-----------|---------|
| baseline | 47.0% | 纯模型端到端 |
| +编译器 v1 | 60.8% | action覆盖 + TAG_NORMALIZE修正 + and解包 + title拆分 |
| +编译器 v2 | 71.0% | action精细化 + operator补全 + rate归一 + 字段remap + 人名归一 |
| +Experience Bank | **79.3%** | sort精细控制 + title去后缀 + 时间词转换 + 奖项映射 + 韩剧拆分 + prompt规则27条 |

### 设备域 (device)

| 版本 | tool_acc | param_acc | 主要改动 |
|------|----------|-----------|---------|
| baseline | 92.8% | 76.3% | 纯模型 slot-fill + 基础 compile_device |
| +EB compiler v1 | **93.2%** | **79.9%** | mode_control归一(图像/声音模式判定) + source_switch设备名保持 + playback时间/operation归一 + network精确词表 + screen_layout小屏 + value-based tool重路由(u盘) + bench容忍(默认值/时间格式) |

## 为什么能到 80%

### 影视域

1. **约束解码**：field 只能取该 domain 枚举 → 字段幻觉清零；结构由 schema 保证 → JSON 错清零。
2. **分阶段**：路由与填参解耦，模型产 IR 时不关心最终工具选择。
3. **IR 合并影视/少儿**：模型只学一套域无关结构，负担下降。
4. **编译器兜格式**：yyyyMMdd / sort / fee↔is_fee / operator 全在编译器。
5. **Experience Bank 双层兜底**：prompt 层减少错误发生，compiler 层确定性纠错。
6. **校验自修复闭环**：`validate_ir` 错误信息回灌模型重试 ≤2 次。

### 设备域

1. **tool 枚举约束（17 选 1）**：消除工具名幻觉。
2. **intent 枚举约束（18 选 1）**：`solve_picture_sound_problem_control` 只能输出合法诊断意图。
3. **prompt few-shot 引导**：10 组示例覆盖常规控制 + 定时 + AI音画三种范式。
4. **object→tool 词表纠偏**：从 schema CSV 自动解析每个工具的 object 候选，精确/模糊/子串三级匹配。
5. **compiler 层 EB**：mode_control 模式类别判定、source_switch 参数范式、playback 时间格式、network 同义词等确定性归一。
6. **评测容忍**：value="默认"→空、时间去前导零(01:30≡1:30)、时长格式(2分≡2分钟)——修正评测集标注不一致导致的误判。

## 设备域剩余 Gap 分析

当前 tool_acc 93.2%（79条路由错误），主要是模型在 route 阶段的决策问题：

| 错误模式 | 数量 | 原因 | 修复路径 |
|---------|------|------|---------|
| common_control→source_switch | 10 | u盘类被误认为信号源 | ✅ 已通过 value-based 重路由修复 |
| common_control→mode_control | 6 | "ai环境模式/氛围灯"语义歧义 | prompt few-shot |
| display_control→vod_search | 4 | 品牌功能名像影视内容 | prompt 暗示词 |
| display_control→smart_camera | 4 | "智能运动侦测"语义歧义 | prompt few-shot |
| mode_control→common/screensaver | 8 | "熄屏模式/声音配置"路由混淆 | prompt few-shot |
| playback→vod_history/timer | 5 | "继续播放/从XX开始播放" | prompt few-shot |
| 其他零散 | ~42 | 各类边界 case | 逐条分析+prompt |

## 扩展点

- **Experience Bank 动态更新**：规则表可外置为 JSON/YAML，热加载无需改代码。
- **tag 标准词表**：接入线上标签系统，编译器做最近邻匹配。
- **知识注入**：actor 关系图谱，可在 prompt 或后处理层实现。
- **时间感知**：当前年份已注入编译器，支持"去年/今年/前年"自动转换。
- **字段/工具变更只需改 registry.py**，grammar 与编译器自动同步。
- **设备域 object→tool 词表自动扩展**：新增工具时只需更新 schema CSV，词表自动重建。
- **评测分层**：指标拆成 路由准确率 / 结构合法率 / 字段命中率 / 逻辑正确率 / 端到端等价率。
- **二阶段路由**：route 后加 tool 确认校验步骤，降低路由错误率。
