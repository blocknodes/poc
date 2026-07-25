# 电视端 AI 慢任务 Planner — IR 层 + 双后端编译器 + vLLM 约束解码

面向 **30B MoE planner**，目标：把工具调用准确率稳定拉到 **80%+**。

## 背景与问题

慢任务接口(`/slowAgent`)要把用户自然语言规划成结构化的工具调用步骤(`steps[]`)。可用工具按域分两套(影视 `vod_*` / 少儿 `educ_*`)，检索类工具存在三重难点，直接让模型端到端填参很难过 80%：

1. **路由混淆**：精确检索 / 慢链路模糊 / 相关推荐 / 个性化 / 历史 / 切片，语义高度接近。
2. **嵌套布尔 DSL**：`vod_search` / `educ_search` 要生成任意嵌套的 `and/or/not + field/values/operator`，裸生成易出 JSON 结构错、字段名幻觉、operator 语义反。
3. **两套参数范式并存**：精确检索是嵌套 JSON，慢链路是扁平字符串 mini-DSL(`"刘德华 AND 吴京"` / `"20200101 TO *"` / `"latest-3-Y"`)，模型容易串味。

## 核心思路

**不让模型直接生成各工具的最终参数**，而是先路由、再让模型产出一份 **域无关的布尔查询 IR**（受 vLLM 约束解码保证结构合法），最后由**编译器**把 IR 落地成具体工具的 parameters。影视/少儿的字段命名差异、以及「精确检索嵌套 JSON vs 慢链路扁平字符串」两套格式差异，全部从模型身上剥离，下沉到 **Field Registry + 编译器**。

## 链路

```
用户query(+memory)
   │
 [路由] route  ── guided_json(小schema) ─► {domain, intent, tool, confidence}
   │
 [IR生成] guided_json(按domain收紧的IR schema) ─► 域无关布尔IR
   │
 [校验] validate_ir  ──失败──► 带错误回灌自修复(≤2次)
   │
 [编译] compile_with_fallback ─► 最终 tool params
        ├─ nested backend  -> vod_search / educ_search
        └─ flat backend    -> *_slow_search（不可无损编译时自动回退到 *_search）
```

## 目录结构

```
poc/
├── planner/
│   ├── registry.py      Field Registry（单一事实源）
│   ├── ir.py            域无关布尔 IR：数据类 / 解析 / 语义校验
│   ├── grammar.py       从 registry 生成 vLLM guided_json schema
│   ├── compiler.py      nested / flat 双后端 + 可编译性检查 + 回退
│   ├── prompts.py       路由判定树 / IR 生成 prompt / few-shot（单 system + observation 文本）
│   ├── agent.py         域无关 agent 状态机（route→IR→自修复），推理与训练共用
│   ├── vllm_client.py   vLLM OpenAI 兼容客户端（guided_json）
│   └── harness.py       端到端 Planner 编排
├── tests/test_planner.py  12 项离线测试（不连模型）
├── demo.py                可运行示例
├── requirements.txt
└── README.md
```

## 模块职责

| 文件 | 职责 |
|---|---|
| `registry.py` | **单一事实源**：canonical 字段、所属 domain、nested/flat 落地名与格式、sort/playback 登记表。字段词表变更只改这里 |
| `ir.py` | IR 数据类(`IR/And/Or/Not/Leaf`)、`parse_ir`、`validate_ir`(registry 驱动的语义校验) |
| `grammar.py` | `build_ir_schema(domain)` / `build_route_schema()`：按 domain 收紧 field 枚举的 draft-07 schema |
| `compiler.py` | `compile_nested` / `compile_flat` / `can_compile_flat` / `compile_with_fallback` |
| `prompts.py` | `route_system_prompt` / `route_observation` / `ir_observation` / `repair_observation`，few-shot 以文本内嵌 |
| `agent.py` | **训推一致核心**：`PlannerAgent` 状态机(route→IR→校验自修复) + `IR_TOOLS` + `loads_lenient`，无模型调用 |
| `vllm_client.py` | `VLLMClient`(`extra_body.guided_json`)；可注入 `responder` 离线测试 |
| `harness.py` | `Planner.plan()`：驱动 `PlannerAgent`，加 client 生成/约束解码/编译 |

## IR 结构

```jsonc
{
  "domain": "vod" | "educ",
  "action": "search" | "play",                        // 可选，默认 search
  "query": <节点>,                                     // and/or/not/leaf 任意嵌套
  "sort":  [{"key":"rate|hot|new|play","order":"asc|desc"}],  // 可选
  "playback": {"series":2,"videoIndex":3}             // 可选，仅 vod
}
```

节点（叶子）四形态：

```jsonc
{"field":"actor","value":"刘德华"}                       // 精确单值
{"field":"actor","values":["刘德华","吴京"],"op":"and"}   // 同字段多值，op 默认 or
{"field":"fee","value":0}                               // 状态字段(fee/is_over)，0|1
{"field":"release_year","range":{"from":"20200101","to":"*"}}  // 范围，开区间用 "*"
```

## 编译示例：一份 IR → 两个后端

IR（"刘德华和吴京都演、非恐怖、2020年后、免费，按评分降序"）编译结果：

→ `vod_search`（nested）：

```json
{"query":{"and":[
  {"field":"actor","values":["刘德华","吴京"],"operator":"and"},
  {"field":"category","value":"电影"},
  {"not":{"field":"tag","value":"恐怖"}},
  {"field":"release_year","from":"20200101","to":"*"},
  {"field":"fee","value":0}]},
 "sort":{"rate":{"order":"desc"}}}
```

→ `vod_slow_search_data_search`（flat）：

```json
{"actor":"刘德华 AND 吴京","category":"电影","tag":"NOT 恐怖",
 "date":"20200101 TO *","is_fee":"0","rate":"desc"}
```

编译到 educ 时 `fee` 自动映射为 `is_fee`。跨字段 OR / 嵌套 NOT 等 flat 无法无损表达时，`compile_with_fallback` 自动回退到 `*_search`。

## 运行

```bash
pip install -r requirements.txt

python tests/test_planner.py        # 12 项离线测试（不连模型）
python demo.py                      # 展示同一 IR 编译到 nested/flat 及回退

# 连真实 vLLM（需以 guided decoding 后端启动）
VLLM_BASE_URL=http://host:8000/v1 VLLM_MODEL=your-30b-moe python demo.py --live
```

最简用法：

```python
from planner import Planner, VLLMClient, VLLMConfig

planner = Planner(VLLMClient(VLLMConfig(base_url="http://host:8000/v1", model="your-30b-moe")))
res = planner.plan("我想看刘德华的免费电影")
print(res.tool_name, res.parameters)   # -> vod_search {...}
```

## vLLM 启动

本 planner 依赖 vLLM 的约束解码（structured outputs）。**vLLM ≥ 0.11 起结构化输出默认开启**，backend 默认 `auto`（对 JSON schema 会自动选 xgrammar），通常无需任何额外参数即可用。

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

- `--structured-outputs-config.backend xgrammar`：**可选**，显式指定约束解码后端（也可用 `guidance`）；不写则用默认 `auto`。需与客户端 `VLLMConfig.guided_backend` 一致。
- 若报 `unrecognized arguments: --structured-outputs-config.backend`，说明是更早的 vLLM：v0.9 及更早用 `--guided-decoding-backend xgrammar`；实在不确定就**直接省略该参数**用默认后端。
- `--served-model-name`：要与 `VLLMConfig.model` / `VLLM_MODEL` 保持一致。
- `--tensor-parallel-size` / `--enable-expert-parallel`：按 30B MoE 的显卡数与专家并行需求调整（单卡可去掉这两项）。
- `--max-model-len 4096`：**必带**。不指定时 vLLM 按模型默认最大上下文（30B 常见 32K/128K）预分配 KV cache，极易 OOM。本 planner 的 prompt（路由 + IR 生成）很短，4096 足够；few-shot 很多时再上调。
- 启动后接口即为 `http://<host>:8000/v1`，填入 `VLLM_BASE_URL`。

### 显存 OOM 排查

按代价从小到大依次尝试：

1. **加 / 调小 `--max-model-len`**：本 planner 4096 就够，这是最有效的一招。
2. **调低 `--gpu-memory-utilization`**（如 0.90 → 0.80）：给权重加载/激活留余量；若是加载阶段就 OOM 尤其有效。
3. **调小 `--max-num-seqs`**（并发序列数，如 256 → 64）：直接减少 KV cache 峰值。
4. **`--kv-cache-dtype fp8`**：KV cache 用 fp8，显存近乎减半（精度影响小）。
5. **加大 `--tensor-parallel-size`**：把权重/KV 摊到更多卡上（需多卡）。

> 版本差异（重要）：
> - **启动参数**：`--structured-outputs-config.backend`（v0.21）↔ `--guided-decoding-backend`（v0.9 及更早）。
> - **请求参数**：v0.11 起 `guided_json` 已废弃，统一用 `structured_outputs: {"json": <schema>}`。本仓库 `vllm_client.py` 默认走新格式，可用 `VLLMConfig.request_format="guided"` 切回旧格式兼容老服务。

## 为什么这样能上 80%

1. **约束解码**：field 只能取该 domain 枚举 → 字段幻觉清零；结构由 schema 保证 → JSON/结构错清零。剩余错误集中到「语义选择」，便于针对性迭代。
2. **分阶段**：路由与填参解耦，且模型产 IR 时不关心最终选哪个 search 工具——精确 vs 慢链路由路由决定，同一 IR 喂不同后端。选错 search 工具从「参数全错」降级为「换后端重编译」。
3. **IR 合并影视/少儿**：模型只学一套域无关结构，负担与出错面同时下降。
4. **编译器兜格式**：yyyyMMdd / `TO` 区间 / sort 位置 / `fee↔is_fee` 全在编译器，模型不碰。
5. **可编译性检查 + 回退**：flat 表达力弱（跨字段仅隐式 AND），遇到跨字段 OR / 嵌套 NOT / educ 日期等不可无损编译时**自动回退到 `*_search`**，绝不静默丢逻辑。
6. **校验自修复闭环**：`validate_ir` 的错误信息回灌模型重试 ≤2 次。

## 扩展点

- 非检索类工具（relate / personalized / history / clip）目前只路由、参数留空（`harness.py` 已留 slot），可各自加独立 slot-filler。
- 字段词表变更只需改 `registry.py`，grammar 与编译器自动同步。
- 建评测集时按工具分层，指标拆成：**路由准确率 / 结构合法率 / 字段命中率 / 逻辑正确率 / 端到端等价率**，用于定位错误落在哪一层。
