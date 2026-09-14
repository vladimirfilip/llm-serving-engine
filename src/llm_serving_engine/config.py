"""Engine and server configuration. Env vars override dataclass defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")


@dataclass
class ModelConfig:
    model_name_or_path: str = "unsloth/Llama-3.2-3B-Instruct"
    device: str = "cuda"  # ModelRunner falls back to "cpu" if unavailable
    dtype: str = "float16"
    quantize: str = "none"  # "none" or "int8"
    use_custom_kernels: bool = False  # use the hand-written Triton kernels instead of HF's

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            model_name_or_path=_env_str("LLM_MODEL", cls.model_name_or_path),
            device=_env_str("LLM_DEVICE", cls.device),
            dtype=_env_str("LLM_DTYPE", cls.dtype),
            quantize=_env_str("LLM_QUANTIZE", cls.quantize),
            use_custom_kernels=_env_bool("LLM_USE_CUSTOM_KERNELS", cls.use_custom_kernels),
        )


@dataclass
class KVCacheConfig:
    """Sizing inputs for the block pool. num_blocks is derived from these fields:
    (memory_budget_bytes) / (block_size * 2 * n_kv_heads * head_dim * dtype_bytes).
    """

    block_size: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    n_layers: int = 32
    dtype_bytes: int = 2
    gpu_memory_utilization: float = 0.85  # fraction of free memory reserved for the KV pool

    @classmethod
    def from_env(cls) -> "KVCacheConfig":
        return cls(
            block_size=_env_int("LLM_BLOCK_SIZE", cls.block_size),
            n_kv_heads=_env_int("LLM_N_KV_HEADS", cls.n_kv_heads),
            head_dim=_env_int("LLM_HEAD_DIM", cls.head_dim),
            n_layers=_env_int("LLM_N_LAYERS", cls.n_layers),
            dtype_bytes=_env_int("LLM_DTYPE_BYTES", cls.dtype_bytes),
            gpu_memory_utilization=_env_float(
                "LLM_GPU_MEM_UTIL", cls.gpu_memory_utilization
            ),
        )

    def bytes_per_token(self) -> int:
        # 2 for K and V; n_kv_heads, not n_heads, since GQA shares K/V across head groups.
        return 2 * self.n_kv_heads * self.head_dim * self.dtype_bytes * self.n_layers

    def num_blocks(self, free_memory_bytes: int) -> int:
        budget = int(free_memory_bytes * self.gpu_memory_utilization)
        return budget // (self.bytes_per_token() * self.block_size)

    @classmethod
    def from_model(cls, hf_config: object) -> "KVCacheConfig":
        """Reads n_kv_heads/head_dim/n_layers off the loaded model's own config instead of
        requiring them kept in sync by hand — swapping model size/family (e.g. Llama-3.2-1B
        vs. Llama-3-8B) must not silently mis-size the KV pool. num_key_value_heads falls
        back to num_attention_heads for non-GQA models; env vars still override if set.
        """
        d = cls()
        n_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
        head_dim = getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
        )
        return cls(
            block_size=_env_int("LLM_BLOCK_SIZE", d.block_size),
            n_kv_heads=_env_int("LLM_N_KV_HEADS", n_kv_heads),
            head_dim=_env_int("LLM_HEAD_DIM", head_dim),
            n_layers=_env_int("LLM_N_LAYERS", hf_config.num_hidden_layers),
            dtype_bytes=_env_int("LLM_DTYPE_BYTES", d.dtype_bytes),
            gpu_memory_utilization=_env_float("LLM_GPU_MEM_UTIL", d.gpu_memory_utilization),
        )


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    output_queue_maxsize: int = 64

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=_env_str("LLM_HOST", cls.host),
            port=_env_int("LLM_PORT", cls.port),
            output_queue_maxsize=_env_int("LLM_OUTPUT_QUEUE_MAXSIZE", cls.output_queue_maxsize),
        )


@dataclass
class EngineConfig:
    model: ModelConfig
    kv_cache: KVCacheConfig
    server: ServerConfig
    token_budget: int = 4096  # mirrors scheduler.TOKEN_BUDGET; kept here so it's one env knob
    scheduler: str = "continuous"  # "continuous" (ContinuousBatchedScheduler) or "static"
    static_batch_size: int = 8  # mirrors scheduler.BATCH_SIZE; only used when scheduler == "static"
    max_concurrent_sequences: int = 64  # mirrors scheduler.MAX_CONCURRENT_SEQUENCES
    kv_allocator: str = "paged"  # "paged" (BlockAllocator) or "contiguous" (ContiguousAllocator)

    @classmethod
    def from_env(cls) -> "EngineConfig":
        return cls(
            model=ModelConfig.from_env(),
            kv_cache=KVCacheConfig.from_env(),
            server=ServerConfig.from_env(),
            token_budget=_env_int("LLM_TOKEN_BUDGET", cls.token_budget),
            scheduler=_env_str("LLM_SCHEDULER", cls.scheduler),
            static_batch_size=_env_int("LLM_STATIC_BATCH_SIZE", cls.static_batch_size),
            max_concurrent_sequences=_env_int(
                "LLM_MAX_CONCURRENT_SEQUENCES", cls.max_concurrent_sequences
            ),
            kv_allocator=_env_str("LLM_KV_ALLOCATOR", cls.kv_allocator),
        )
