"""电视端 AI 慢任务 planner —— IR 层 + 双后端编译器 + vLLM 约束解码 harness。

模块：
  registry   Field Registry（单一事实源）
  ir         域无关布尔查询 IR + 校验
  grammar    从 registry 生成 vLLM guided_json schema
  compiler   nested/flat 双后端 + 可编译性检查
  prompts    路由/IR 生成 prompt
  vllm_client vLLM OpenAI 兼容客户端（支持 guided_json）
  harness    端到端编排 Planner
"""
from .harness import Planner, PlanResult
from .vllm_client import VLLMClient, VLLMConfig

__all__ = ["Planner", "PlanResult", "VLLMClient", "VLLMConfig"]
