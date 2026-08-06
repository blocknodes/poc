# slime — 慢任务轻量规划服务

轻量级慢任务接口，接收用户自然语言请求，通过 enhanced_vllm 中间层进行意图路由和参数填槽，返回结构化工具调用。

## 启动

```bash
# 基本启动（连接 enhanced_vllm 中间层）
VLLM_BASE_URL=http://localhost:9000/v1 VLLM_MODEL=your-30b-moe \
  python slime/server.py --port 8082

# debug 模式（打印每次 LLM 调用的详细信息和耗时）
VLLM_BASE_URL=http://localhost:9000/v1 VLLM_MODEL=your-30b-moe \
  DEBUG=1 python slime/server.py --port 8082
```

## 接口

**POST** `/slowAgent/poc_1784284048243`

### 请求参数

```json
{
  "traceId": "请求追踪ID",
  "deviceId": "设备ID",
  "data": {
    "query": "用户自然语言请求（必填）",
    "toolList": [
      {
        "toolName": "vod_search",
        "description": "影视搜索"
      },
      {
        "toolName": "numeric_adjust",
        "description": "数值调节"
      }
    ],
    "enableMultiIntent": false,
    "memory": {
      "shortMemory": [
        {"query": "上一轮用户说的话", "answer": "上一轮系统回答"}
      ]
    }
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `traceId` | string | 否 | 请求追踪 ID，原样返回 |
| `deviceId` | string | 否 | 设备 ID，原样返回 |
| `data.query` | string | **是** | 用户自然语言请求 |
| `data.toolList` | array | 否 | 可用工具列表。传入时路由阶段**只能从这些工具中选择**（硬约束）；不传则不限制 |
| `data.toolList[].toolName` | string | 是 | 工具名称 |
| `data.enableMultiIntent` | boolean | 否 | 是否开启多意图拆分。`true`=先做意图拆分再并发规划；`false`或不传=直接单意图规划（省一次 LLM 调用） |
| `data.memory` | object | 否 | 上下文记忆 |
| `data.memory.shortMemory` | array | 否 | 短期对话历史（取最近 3 轮） |

### 响应参数

**成功响应** (code=200)：

```json
{
  "traceId": "原样返回",
  "deviceId": "原样返回",
  "code": 200,
  "message": "success",
  "data": {
    "planId": "plan_a1b2",
    "schemaVersion": "1.0",
    "planType": "execute",
    "steps": [
      {
        "id": "s1",
        "toolName": "vod_search",
        "parameters": {
          "action": "search",
          "query": {"and": [{"field": "actor", "value": "刘德华"}]},
          "retext": "刘德华的电影"
        },
        "dependsOn": [],
        "retext": "刘德华的电影"
      }
    ],
    "planConfidence": 0.95
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | int | 200=成功，400=参数错误，500=内部错误 |
| `message` | string | 状态描述 |
| `data.planId` | string | 本次规划唯一 ID |
| `data.schemaVersion` | string | 固定 "1.0" |
| `data.planType` | string | 固定 "execute" |
| `data.steps` | array | 规划步骤列表（多意图时可能有多个） |
| `data.steps[].id` | string | 步骤 ID（s1, s2, ...） |
| `data.steps[].toolName` | string | 工具名称（保证在 toolList 范围内） |
| `data.steps[].parameters` | object | 工具参数（结构因工具而异） |
| `data.steps[].dependsOn` | array | 依赖步骤（当前固定为空） |
| `data.steps[].retext` | string | 原始 query |
| `data.planConfidence` | float | 规划置信度 (0~1) |

**错误响应**：

```json
{
  "traceId": "...",
  "deviceId": "...",
  "code": 500,
  "message": "错误信息",
  "data": {"steps": []}
}
```

## 各工具参数结构

### vod 域（影视）

工具：`vod_search` / `vod_fuzzy_search` / `vod_search_all` / `vod_relate_search` / `vod_history`

```json
{
  "action": "search|play",
  "query": {"and": [{"field": "actor", "value": "刘德华"}, {"field": "category", "value": "电影"}]},
  "sort": [{"key": "rate", "order": "desc"}],
  "playback": {"series": 2, "video_index": 3},
  "retext": "原始query文本"
}
```

### device 域（设备控制）

工具：`numeric_adjust` / `power_control` / `timer_control` / `source_switch` / `playback_control` / `mode_control` / `screen_layout` / ... (17个)

```json
{
  "operation": "提高|降低|打开|关闭|设置|查询|切换",
  "object": "音量|亮度|电源|WiFi|...",
  "value": "30|标准模式|HDMI1",
  "date_time": "30分钟|22:00"
}
```

### audio 域（有声）

工具：`audio_search` / `audio_chat_qa`

```json
{
  "query": "三体",
  "play_mode": "play|search",
  "screen_mode": "screen_off|normal"
}
```

## 架构

```
客户端 → slime(8082) → enhanced_vllm(9000) → vLLM(8000)
```

- **slime**：接口适配层，提取 toolList/memory，驱动规划流程，构造响应
- **enhanced_vllm**：中间层，注入完整 prompt+few-shot+EB 规则，约束解码，后处理编译
- **vLLM**：模型推理引擎

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_BASE_URL` | `http://localhost:9000/v1` | enhanced_vllm 地址 |
| `VLLM_MODEL` | `baseline` | 模型名 |
| `VLLM_TIMEOUT` | `60` | 请求超时(秒) |
| `DEBUG` | 空 | 设为 `1` 开启 debug 日志 |
