"""Engine and server configuration. Env vars override dataclass defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .scheduling.scheduler import BATCH_SIZE, MAX_CONCURRENT_SEQUENCES, TOKEN_BUDGET


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
    device: str = "cuda"  # "cpu" when CUDA is unavailable
    dtype: str = "float16"
    quantize: str = "none"  # "none" or "int8"
    # Batched Triton attention on the hot path. On by default, so a CPU or non-Llama
    # deployment fails at load time; the per-sequence HF forward is opt-in.
    use_custom_kernels: bool = True
    # Replays pure-decode iterations as one captured CUDA graph (paged KV only). A graph
    # that fails its post-capture self-check is dropped and decode runs eagerly.
    use_cuda_graphs: bool = True

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            model_name_or_path=_env_str("LLM_MODEL", cls.model_name_or_path),
            device=_env_str("LLM_DEVICE", cls.device),
            dtype=_env_str("LLM_DTYPE", cls.dtype),
            quantize=_env_str("LLM_QUANTIZE", cls.quantize),
            use_custom_kernels=_env_bool("LLM_USE_CUSTOM_KERNELS", cls.use_custom_kernels),
            use_cuda_graphs=_env_bool("LLM_USE_CUDA_GRAPHS", cls.use_cuda_graphs),
        )


@dataclass
class KVCacheConfig:
    """Sizing inputs for the block pool: num_blocks = memory budget / (block_size *
    bytes_per_token)."""

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
        # 2 for K and V. Under GQA a group of query heads shares one K/V head, so K/V
        # storage scales with n_kv_heads.
        return 2 * self.n_kv_heads * self.head_dim * self.dtype_bytes * self.n_layers

    def num_blocks(self, free_memory_bytes: int) -> int:
        budget = int(free_memory_bytes * self.gpu_memory_utilization)
        return budget // (self.bytes_per_token() * self.block_size)

    @classmethod
    def from_model(cls, hf_config: object, dtype_bytes: int | None = None) -> "KVCacheConfig":
        """Pool shape read off the loaded model's config, and dtype_bytes off its loaded
        dtype (e.g. `model.dtype.itemsize`), so the sizing always matches the pool the
        model runner allocates. A model without num_key_value_heads has one K/V head per
        query head. Env vars override every field.
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
            dtype_bytes=_env_int(
                "LLM_DTYPE_BYTES", dtype_bytes if dtype_bytes is not None else d.dtype_bytes
            ),
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
    token_budget: int = TOKEN_BUDGET
    scheduler: str = "continuous"  # "continuous" (ContinuousBatchedScheduler) or "static"
    static_batch_size: int = BATCH_SIZE  # only used when scheduler == "static"
    max_concurrent_sequences: int = MAX_CONCURRENT_SEQUENCES
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
