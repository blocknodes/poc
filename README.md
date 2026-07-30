# 电视端 AI 慢任务 Planner — IR 层 + 编译器 + Experience Bank + vLLM 约束解码

面向 **30B MoE planner**，目标：把工具调用准确率稳定拉到 **80%+**。

当前 benchmark（217 条影视评测用例）：
- 工具准确率 **97.7%**
- 工具+参数准确率 **79.3%**（从 baseline 47% 优化而来）

## 背景与问题

慢任务接口(`/slowAgent`)要把用户自然语言规划成结构化的工具调用步骤(`steps[]`)。可用工具按域分两套(影视 `vod_*` / 少儿 `educ_*`)，检索类工具存在三重难点，直接让模型端到端填参很难过 80%：

1. **路由混淆**：精确检索 / 慢链路模糊 / 相关推荐 / 个性化 / 历史 / 切片，语义高度接近。
2. **嵌套布尔 DSL**：`vod_search` / `educ_search` 要生成任意嵌套的 `and/or/not + field/values/operator`，裸生成易出 JSON 结构错、字段名幻觉、operator 语义反。
3. **两套参数范式并存**：精确检索是嵌套 JSON，慢链路是扁平字符串 mini-DSL(`"刘德华 AND 吴京"` / `"20200101 TO *"` / `"latest-3-Y"`)，模型容易串味。

## 核心思路

**三层架构：模型生成 IR → 编译器确定性转换 → Experience Bank 兜底修正**

1. **模型只产出域无关 IR**（受 vLLM 约束解码保证结构合法）
2. **编译器**把 IR 落地成具体工具参数，同时执行确定性后处理规则
3. **Experience Bank**（领域规则库）分两处生效：
   - **Prompt 层**：指导模型正确生成 IR（few-shot + 规则描述）
   - **Compiler 层**：编译后确定性修正（不依赖模型行为）

## 链路

```
用户query(+memory)
   │
 [路由] route  ── guided_json(小schema) ─► {domain, intent, tool, confidence}
   │
 [IR生成] guided_json(按domain收紧的IR schema) ─► 域无关布尔IR
   │                                              ↑ Experience Bank (prompt层)
 [校验] validate_ir  ──失败──► 带错误回灌自修复(≤2次)
   │
 [编译] compile_with_fallback ─► 最终 tool params
   │     ├─ nested backend  -> vod_search / vod_search_all / educ_search
   │     └─ flat backend    -> *_slow_search（只传query原文）
   │
 [后处理] Experience Bank (compiler层)
   │     ├─ action 动词判定（播放/放/请播→play）
   │     ├─ 字段名修正（好莱坞→company, 湖南卫视→channel）
   │     ├─ 值归一化（人名间隔符, TVB→tvb, tag/奖项映射）
   │     ├─ 结构修正（单元素and解包, not is_fee→is_fee:0）
   │     ├─ query-text补全（影片→category, 韩剧拆分, 时间词转换）
   │     └─ title+series 拆分（战狼2→战狼+series:2）
   ▼
 最终输出 {tool_name, parameters}
```

## 目录结构

```
poc/
├── planner/
│   ├── registry.py      Field Registry（单一事实源）
│   ├── ir.py            域无关布尔 IR：数据类 / 解析 / 语义校验
│   ├── grammar.py       从 registry 生成 vLLM guided_json schema
│   ├── compiler.py      编译器 + Experience Bank (compiler层) 全部后处理规则
│   ├── prompts.py       路由/IR prompt + Experience Bank (prompt层) 领域规则
│   ├── agent.py         域无关 agent 状态机（route→IR→自修复），推理与训练共用
│   ├── vllm_client.py   vLLM OpenAI 兼容客户端（guided_json）
│   └── harness.py       端到端 Planner 编排
├── bench_vod.py           影视评测脚本（含 param_diff 差异分析）
├── tests/test_planner.py  81 项离线测试（不连模型）
├── demo.py                可运行示例
├── requirements.txt
└── README.md
```

## 模块职责

| 文件 | 职责 |
|---|---|
| `registry.py` | **单一事实源**：canonical 字段、所属 domain、nested/flat 落地名与格式、sort/playback 登记表 |
| `ir.py` | IR 数据类(`IR/And/Or/Not/Leaf`)、`parse_ir`、`validate_ir`(registry 驱动的语义校验) |
| `grammar.py` | `build_ir_schema(domain)` / `build_route_schema()`：按 domain 收紧 field 枚举的 draft-07 schema |
| `compiler.py` | 编译核心 + **Experience Bank (compiler层)**：双后端编译 + 全部确定性后处理规则 |
| `prompts.py` | 路由/IR prompt + **Experience Bank (prompt层)**：27 条领域规则 + 21 组 few-shot |
| `agent.py` | **训推一致核心**：`PlannerAgent` 状态机(route→IR→校验自修复) |
| `vllm_client.py` | `VLLMClient`(`extra_body.guided_json`)；可注入 `responder` 离线测试 |
| `harness.py` | `Planner.plan()`：驱动 Agent + 编译 + 后处理完整流程 |
| `bench_vod.py` | 评测脚本：多worker并发、param_diff 差异列、按工具细分统计 |

## Experience Bank 架构

Experience Bank 是一套**领域规则库**，分两层生效：

### Prompt 层（`prompts.py` 内 27 条规则）

指导模型在 IR 生成阶段做正确的语义选择：

| 规则类 | 示例 |
|--------|------|
| action 判定 | "播放/放/请播" → play；"我想看/我要看" + 无片名 → search |
| category 推断 | "XX片" → 电影；"XX剧" → 电视剧；"节目" → 综艺 |
| tag 保持原词 | 用户说"抗战" → tag:"抗战"（不替换为"战争"） |
| tag vs 其他字段 | "真人版/TV版" → tag；"秦腔/小品" → tag；"方言" → language |
| playback 区分 | "第N集" → video_index；"第N分钟" → voiceStartPos |
| 字段选择 | "好莱坞" → company；"笑果文化" → comedy_brand；"金庸改编" → writer |
| sort 触发 | "好看的" 不加 sort；"最新" → new:desc；有明确分数不加 sort |
| 多值 op | "A和B" → and；"港台" → or |

### Compiler 层（`compiler.py` 内确定性后处理）

编译后基于规则表 **无条件修正** 输出，不依赖模型行为：

| 规则类 | 处理 |
|--------|------|
| action 动词覆盖 | 从原文提取动词，强动词(播放/放/请播/打开)→play，弱动词(看/想看)+无title→search |
| 字段名 remap | 好莱坞→company, 湖南卫视→channel, TVB→company:tvb |
| 值归一化 | 人名间隔符(汤姆克鲁斯→汤姆·克鲁斯), vender(芒果TV→芒果tv), tag大小写(DVD版→dvd版) |
| 结构简化 | 单元素 and/or 解包；not{is_fee:1}→is_fee:0 |
| query-text 补全 | "影片"→category:电影, "韩剧"→area+category, "节目"→category:综艺 |
| sort 精细控制 | 有明确分数→不加sort；"好看的"→删sort；"最近"→sort:new+删date range |
| title 处理 | 去后缀(漩涡视频→漩涡), 拆数字(战狼2→战狼+series:2) |
| 时间转换 | "去年/今年/前年"→绝对年份 |
| 名称映射 | 奖项(香港金像奖→香港电影金像奖), tag(辩论赛→辩论), company(TVB→tvb) |
| rate 归一 | 开区间"*"→10.0, from转float |
| operator 补全 | values 节点始终输出 operator 字段 |

## IR 结构

```jsonc
{
  "domain": "vod" | "educ",
  "action": "search" | "play",                        // 可选，默认 search
  "query": <节点>,                                     // and/or/not/leaf 任意嵌套
  "sort":  [{"key":"rate|hot|new|play","order":"asc|desc"}],  // 可选
  "playback": {"series":2,"video_index":3}            // 可选，仅 vod play
}
```

节点（叶子）四形态：

```jsonc
{"field":"actor","value":"刘德华"}                       // 精确单值
{"field":"actor","values":["刘德华","吴京"],"op":"and"}   // 同字段多值
{"field":"fee","value":0}                               // 状态字段(fee/is_over)，0|1
{"field":"release_year","range":{"from":"20200101","to":"*"}}  // 范围，开区间用 "*"
```

## 编译示例

"刘德华和吴京都演的、非恐怖、2020年后、免费电影，按评分降序" →

```json
{"action": "search",
 "query": {"and": [
   {"field": "actor", "values": ["刘德华", "吴京"], "operator": "and"},
   {"field": "category", "value": "电影"},
   {"not": {"field": "tag", "value": "恐怖"}},
   {"field": "release_time", "from": "20200101", "to": "*"},
   {"field": "is_fee", "value": 0}
 ]},
 "sort": {"rate": {"order": "desc"}},
 "retext": "刘德华和吴京都演的非恐怖2020年后免费电影按评分降序"}
```

## 运行

```bash
pip install -r requirements.txt

python tests/test_planner.py        # 81 项离线测试（不连模型）
python demo.py                      # 展示编译 + 后处理效果

# 连真实 vLLM 评测
VLLM_BASE_URL=http://host:8000/v1 VLLM_MODEL=your-30b-moe \
  python bench_vod.py test_set/*.csv --live --output results.csv --worker 16
```

最简用法：

```python
from planner import Planner, VLLMClient, VLLMConfig

planner = Planner(VLLMClient(VLLMConfig(base_url="http://host:8000/v1", model="your-30b-moe")))
res = planner.plan("我想看刘德华的免费电影")
print(res.tool_name, res.parameters)   # -> vod_search {...}
```

## 评测

```bash
# 评测输出含 param_diff 列，直观展示每条错误的具体差异
VLLM_BASE_URL=http://localhost:8080/v1 VLLM_MODEL=baseline \
  python bench_vod.py test_set/*.csv --live --output results1.csv --worker 16
```

输出 CSV 列：`source_file | row | query | expected_tool | predicted_tool | tool_correct | param_correct | expected_params | predicted_params | param_diff | error | note`

`param_diff` 列示例：
- `action:search→play`
- `多余字段:tag=戏曲; 缺失字段:category=戏曲`
- `值不同:tag(应=抗战,pred=战争)`
- `op不同:actor(应=and,pred=or)`

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

说明：
- `--structured-outputs-config.backend xgrammar`：可选，v0.9 及更早用 `--guided-decoding-backend xgrammar`
- `--max-model-len 4096`：**必带**，本 planner prompt 很短，4096 足够
- 启动后接口即为 `http://<host>:8000/v1`

### 显存 OOM 排查

1. 加/调小 `--max-model-len`（最有效）
2. 调低 `--gpu-memory-utilization`（0.90 → 0.80）
3. 调小 `--max-num-seqs`（256 → 64）
4. `--kv-cache-dtype fp8`
5. 加大 `--tensor-parallel-size`

## 为什么能到 80%

1. **约束解码**：field 只能取该 domain 枚举 → 字段幻觉清零；结构由 schema 保证 → JSON 错清零。
2. **分阶段**：路由与填参解耦，模型产 IR 时不关心最终工具选择。
3. **IR 合并影视/少儿**：模型只学一套域无关结构，负担下降。
4. **编译器兜格式**：yyyyMMdd / sort / fee↔is_fee / operator 全在编译器。
5. **Experience Bank 双层兜底**：prompt 层减少错误发生，compiler 层确定性纠错。
6. **校验自修复闭环**：`validate_ir` 错误信息回灌模型重试 ≤2 次。

## 优化历程

| 版本 | param_acc | 主要改动 |
|------|-----------|---------|
| baseline | 47.0% | 纯模型端到端 |
| +编译器 v1 | 60.8% | action覆盖 + TAG_NORMALIZE修正 + and解包 + title拆分 |
| +编译器 v2 | 71.0% | action精细化(强/弱动词) + operator补全 + rate归一 + 字段remap + 人名归一 |
| +Experience Bank | **79.3%** | sort精细控制 + title去后缀 + 时间词转换 + 奖项映射 + 韩剧拆分 + prompt规则27条 |

剩余错误主要是：标注存疑(~5条)、外部知识推理(~2条)、模型语义理解(~8条)、数据问题(~2条)。

## 扩展点

- **Experience Bank 动态更新**：规则表可外置为 JSON/YAML，热加载无需改代码。
- **tag 标准词表**：接入线上标签系统，编译器做最近邻匹配（辩论赛→辩论）。
- **知识注入**：actor 关系图谱（孙俪→邓超），可在 prompt 或后处理层实现。
- **时间感知**：当前年份已注入编译器，支持"去年/今年/前年"自动转换。
- **字段词表变更**只需改 `registry.py`，grammar 与编译器自动同步。
- **评测分层**：指标拆成 路由准确率 / 结构合法率 / 字段命中率 / 逻辑正确率 / 端到端等价率。
