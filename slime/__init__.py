"""slim —— 自包含的轻量 planner 包。"""
try:
    from .vllm_client import VLLMClient, VLLMConfig
    from .planner import Planner, PlanResult
except ImportError:
    from vllm_client import VLLMClient, VLLMConfig
    from planner import Planner, PlanResult

__all__ = ["VLLMClient", "VLLMConfig", "Planner", "PlanResult"]
