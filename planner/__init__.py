"""电视端 AI 慢任务 planner —— IR 层 + 编译器 + Experience Bank + vLLM 约束解码。

覆盖 4 域：影视(vod) / 少儿(educ) / 有声(audio) / 设备控制(device)。

模块：
  registry    Field Registry（单一事实源）+ 各域工具与 slot 定义
  ir          域无关布尔查询 IR + 校验（vod/educ 检索类）
  grammar     从 registry 生成 vLLM guided_json schema（IR / audio / device / route）
  compiler    nested/flat 双后端 + Experience Bank compiler层 + audio/device 编译
  prompts     路由/IR/audio/device prompt 模板 + Experience Bank prompt层
  agent       域无关 agent 状态机（训推一致）
  vllm_client vLLM OpenAI 兼容客户端（支持 guided_json）
  harness     端到端编排 Planner
"""
from .compiler import ExperienceBankConfig, get_experience_bank_config, set_experience_bank_config
from .harness import Planner, PlanResult
from .vllm_client import VLLMClient, VLLMConfig

__all__ = [
    "Planner", "PlanResult", "VLLMClient", "VLLMConfig",
    "ExperienceBankConfig", "get_experience_bank_config", "set_experience_bank_config",
]
