"""统一配置加载 —— 读取 .env 文件 + 环境变量，提供类型安全的配置对象。

加载优先级（高→低）：
    1. 已设置的环境变量（export / CLI 前缀）
    2. 项目根目录 .env 文件
    3. 代码内默认值

用法：
    from config import cfg

    # 所有配置项直接属性访问
    cfg.vllm_base_url     # str
    cfg.retrieve_enabled  # bool
    cfg.bench_workers     # int
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _find_env_file() -> Optional[Path]:
    """从当前文件所在目录向上查找 .env 文件。"""
    # 优先在 poc/ 目录找
    here = Path(__file__).resolve().parent
    env_path = here / ".env"
    if env_path.exists():
        return env_path
    # 再向上一级
    env_path = here.parent / ".env"
    if env_path.exists():
        return env_path
    return None


def _load_env_file(path: Path) -> dict[str, str]:
    """解析 .env 文件（支持注释、空行、带引号值）。"""
    result: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 去引号
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            result[key] = value
    return result


def _get(env_dict: dict[str, str], key: str, default: str = "") -> str:
    """获取配置值：环境变量优先 > .env 文件 > default。"""
    return os.environ.get(key) or env_dict.get(key) or default


def _bool(val: str) -> bool:
    """字符串转布尔。"""
    return val.lower() in ("1", "true", "yes", "on")


def _int(val: str, default: int = 0) -> int:
    """字符串转整数。"""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _float(val: str, default: float = 0.0) -> float:
    """字符串转浮点数。"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


@dataclass
class Config:
    """统一配置对象，所有字段从 .env + 环境变量加载。"""

    # ---- vLLM ----
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "your-30b-moe"
    vllm_api_key: str = "EMPTY"
    vllm_timeout: float = 60.0
    vllm_request_format: str = "structured_outputs"

    # ---- 检索层 ----
    retrieve_enabled: bool = False
    retrieve_base_url: str = "http://10.18.231.63:31935"
    retrieve_tool_k: int = 3
    retrieve_parameter_k: int = 6
    retrieve_value_k: int = 6
    retrieve_timeout: float = 5.0

    # ---- Planner 行为 ----
    planner_stub: bool = False
    planner_vod_only: bool = False
    planner_educ_only: bool = False
    planner_max_repairs: int = 2

    # ---- Experience Bank ----
    eb_enabled: bool = True
    eb_prompt_enabled: bool = True
    eb_compiler_enabled: bool = True
    eb_action_override: bool = True
    eb_field_remap: bool = True
    eb_value_normalize: bool = True
    eb_sort_control: bool = True
    eb_query_text_complete: bool = True
    eb_title_process: bool = True
    eb_time_convert: bool = True
    eb_name_normalize: bool = True
    eb_structure_simplify: bool = True

    # ---- 评测 ----
    bench_workers: int = 16
    bench_output: str = ""
    bench_trace: str = ""

    # ---- 服务端 ----
    server_host: str = "0.0.0.0"
    server_port: int = 8082


def load_config() -> Config:
    """加载配置：先读 .env 文件，再用环境变量覆盖。"""
    env_file = _find_env_file()
    env_dict: dict[str, str] = {}
    if env_file:
        env_dict = _load_env_file(env_file)

    return Config(
        # vLLM
        vllm_base_url=_get(env_dict, "VLLM_BASE_URL", "http://localhost:8000/v1"),
        vllm_model=_get(env_dict, "VLLM_MODEL", "your-30b-moe"),
        vllm_api_key=_get(env_dict, "VLLM_API_KEY", "EMPTY"),
        vllm_timeout=_float(_get(env_dict, "VLLM_TIMEOUT", "60"), 60.0),
        vllm_request_format=_get(env_dict, "VLLM_REQUEST_FORMAT", "structured_outputs"),

        # 检索层
        retrieve_enabled=_bool(_get(env_dict, "RETRIEVE_ENABLED", "0")),
        retrieve_base_url=_get(env_dict, "RETRIEVE_BASE_URL", "http://10.18.231.63:31935"),
        retrieve_tool_k=_int(_get(env_dict, "RETRIEVE_TOOL_K", "3"), 3),
        retrieve_parameter_k=_int(_get(env_dict, "RETRIEVE_PARAMETER_K", "6"), 6),
        retrieve_value_k=_int(_get(env_dict, "RETRIEVE_VALUE_K", "6"), 6),
        retrieve_timeout=_float(_get(env_dict, "RETRIEVE_TIMEOUT", "5"), 5.0),

        # Planner
        planner_stub=_bool(_get(env_dict, "PLANNER_STUB", "0")),
        planner_vod_only=_bool(_get(env_dict, "PLANNER_VOD_ONLY", "0")),
        planner_educ_only=_bool(_get(env_dict, "PLANNER_EDUC_ONLY", "0")),
        planner_max_repairs=_int(_get(env_dict, "PLANNER_MAX_REPAIRS", "2"), 2),

        # Experience Bank
        eb_enabled=_bool(_get(env_dict, "EB_ENABLED", "1")),
        eb_prompt_enabled=_bool(_get(env_dict, "EB_PROMPT_ENABLED", "1")),
        eb_compiler_enabled=_bool(_get(env_dict, "EB_COMPILER_ENABLED", "1")),
        eb_action_override=_bool(_get(env_dict, "EB_ACTION_OVERRIDE", "1")),
        eb_field_remap=_bool(_get(env_dict, "EB_FIELD_REMAP", "1")),
        eb_value_normalize=_bool(_get(env_dict, "EB_VALUE_NORMALIZE", "1")),
        eb_sort_control=_bool(_get(env_dict, "EB_SORT_CONTROL", "1")),
        eb_query_text_complete=_bool(_get(env_dict, "EB_QUERY_TEXT_COMPLETE", "1")),
        eb_title_process=_bool(_get(env_dict, "EB_TITLE_PROCESS", "1")),
        eb_time_convert=_bool(_get(env_dict, "EB_TIME_CONVERT", "1")),
        eb_name_normalize=_bool(_get(env_dict, "EB_NAME_NORMALIZE", "1")),
        eb_structure_simplify=_bool(_get(env_dict, "EB_STRUCTURE_SIMPLIFY", "1")),

        # 评测
        bench_workers=_int(_get(env_dict, "BENCH_WORKERS", "16"), 16),
        bench_output=_get(env_dict, "BENCH_OUTPUT", ""),
        bench_trace=_get(env_dict, "BENCH_TRACE", ""),

        # 服务端
        server_host=_get(env_dict, "SERVER_HOST", "0.0.0.0"),
        server_port=_int(_get(env_dict, "SERVER_PORT", "8082"), 8082),
    )


# 模块级单例：导入即可用
cfg = load_config()


def make_vllm_config() -> "VLLMConfig":
    """从统一配置构造 VLLMConfig 对象。"""
    from planner import VLLMConfig
    return VLLMConfig(
        base_url=cfg.vllm_base_url,
        model=cfg.vllm_model,
        api_key=cfg.vllm_api_key,
        timeout=cfg.vllm_timeout,
        request_format=cfg.vllm_request_format,
    )


def make_retrieve_config() -> Optional["RetrieveConfig"]:
    """从统一配置构造 RetrieveConfig 对象，RETRIEVE_ENABLED=0 时返回 None。"""
    if not cfg.retrieve_enabled:
        return None
    from planner import RetrieveConfig
    return RetrieveConfig(
        base_url=cfg.retrieve_base_url,
        tool_k=cfg.retrieve_tool_k,
        parameter_k=cfg.retrieve_parameter_k,
        value_k=cfg.retrieve_value_k,
        timeout=cfg.retrieve_timeout,
    )


def make_eb_config() -> "ExperienceBankConfig":
    """从统一配置构造 ExperienceBankConfig 对象。"""
    from planner import ExperienceBankConfig
    return ExperienceBankConfig(
        enabled=cfg.eb_enabled and cfg.eb_compiler_enabled,
        action_override=cfg.eb_action_override,
        field_remap=cfg.eb_field_remap,
        value_normalize=cfg.eb_value_normalize,
        sort_control=cfg.eb_sort_control,
        query_text_complete=cfg.eb_query_text_complete,
        title_process=cfg.eb_title_process,
        time_convert=cfg.eb_time_convert,
        name_normalize=cfg.eb_name_normalize,
        structure_simplify=cfg.eb_structure_simplify,
    )


def make_planner(client=None) -> "Planner":
    """从统一配置一站式构造 Planner 实例。

    Args:
        client: 可选，若不传则根据配置自动创建 VLLMClient。

    Returns:
        配置好的 Planner 实例。
    """
    from planner import Planner, VLLMClient, set_experience_bank_config

    # Experience Bank
    set_experience_bank_config(make_eb_config())

    # VLLMClient
    if client is None:
        if cfg.planner_stub:
            from demo import make_stub
            client = VLLMClient(responder=make_stub())
        else:
            client = VLLMClient(make_vllm_config())

    # Retrieve
    retrieve_config = make_retrieve_config()

    return Planner(
        client,
        max_repairs=cfg.planner_max_repairs,
        vod_only=cfg.planner_vod_only,
        educ_only=cfg.planner_educ_only,
        use_eb_prompt=cfg.eb_enabled and cfg.eb_prompt_enabled,
        use_retrieve=cfg.retrieve_enabled,
        retrieve_config=retrieve_config,
    )
