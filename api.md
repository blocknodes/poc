# 慢任务接口_v2 API文档
## 一、接口基础信息
| 项 | 内容 |
| ---- | ---- |
| 接口名称 | 慢任务接口_v2 |
| 创建人 | yangshuxin2 |
| 状态 | 未完成 |
| 更新时间 | 2026-07-22 20:43:53 |
| 请求方式 | POST |
| 接口路径 | /slowAgent/poc_1784284048243 |
| Mock调试地址 | http://10.18.229.83:3000/mock/1304/slowAgent/poc_1784284048243 |

## 二、请求参数
### 2.1 请求 Headers
| 参数名称 | 参数值 | 是否必须 | 示例 | 备注 |
| ---- | ---- | ---- | ---- | ---- |
| Content-Type | application/json | 是 | - | 请求体JSON格式 |

### 2.2 请求 Body 入参
| 字段名 | 数据类型 | 是否必须 | 默认值 | 字段说明 |
| ---- | ---- | ---- | ---- | ---- |
| traceId | string | 必须 | - | 请求追踪号，便于日志排查 |
| deviceId | string | 必须 | - | 设备唯一ID |
| deviceType | string | 必须 | - | 设备类型标识 |
| data | object | 必须 | - | 业务核心数据对象 |
| └─ query | string | 必须 | - | 用户原始请求文本 |
| └─ tvMode | string | 必须 | - | 设备运行状态枚举<br>0-普通电视模型<br>1-音箱<br>2-月光模式<br>3-双屏<br>4-网络机顶盒<br>5-小聚聊天室<br>6-息屏模式 |
| └─ segment | object | 必须 | - | 语句分词解析信息 |
| └─ memory | object | 非必须 | - | 记忆数据容器 |
| &nbsp;&nbsp;&nbsp;└─ shortMemory | object[] | 非必须 | - | 短期记忆数组，存储历史5轮对话 |
| &nbsp;&nbsp;&nbsp;└─ longMemory | object | 非必须 | - | 长期记忆，包含用户画像信息 |
| └─ toolList | object[] | 必须 | - | 可用工具定义列表 |
| &nbsp;&nbsp;&nbsp;└─ toolName | string | 必须 | - | 工具名称 |
| &nbsp;&nbsp;&nbsp;└─ description | string | 必须 | - | 工具功能描述 |
| &nbsp;&nbsp;&nbsp;└─ parameters | object | 必须 | - | 工具入参定义 |
| └─ toolHistory | object[] | 非必须 | - | 历史工具执行记录数组 |
| &nbsp;&nbsp;&nbsp;└─ stepId | string | 必须 | - | 执行步骤唯一ID |
| &nbsp;&nbsp;&nbsp;└─ toolName | string | 必须 | - | 调用工具名称 |
| &nbsp;&nbsp;&nbsp;└─ parameters | object | 必须 | - | 本次工具调用参数 |
| &nbsp;&nbsp;&nbsp;└─ result | object | 必须 | - | 工具执行返回结果 |
| └─ timestamp | string | 必须 | - | 请求时间戳 |

## 三、接口返回数据
| 字段名 | 数据类型 | 是否必须 | 默认值 | 字段说明 |
| ---- | ---- | ---- | ---- | ---- |
| traceId | string | 必须 | - | 透传请求传入的追踪ID |
| deviceId | string | 必须 | - | 透传设备ID |
| code | number | 必须 | - | 响应状态码<br>200=成功响应<br>400=请求参数缺失/非法<br>500=服务器内部异常 |
| message | string | 必须 | - | 响应文本描述 |
| data | object | 必须 | - | 业务返回主体 |
| └─ steps | object[] | 必须 | - | 模型规划执行步骤，仅code=200时有数据 |
| &nbsp;&nbsp;&nbsp;└─ stepId | string | 非必须 | - | 步骤编号 |
| &nbsp;&nbsp;&nbsp;└─ toolName | string | 非必须 | - | 待执行工具名称 |
| &nbsp;&nbsp;&nbsp;└─ parameters | object | 非必须 | - | 工具调用参数 |

## 四、底部信息
- 文档来源：YApi 接口管理平台
- 仓库地址：GitHub YMFE YApi
- 反馈渠道：Github Issues / Github Pull Requests
- 平台版本：1.10.3
- 版权：Copyright © 2018-2026 YMFE
