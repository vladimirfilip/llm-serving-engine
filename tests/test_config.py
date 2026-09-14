from llm_serving_engine.config import EngineConfig, KVCacheConfig, ModelConfig, ServerConfig


def test_defaults_construct():
    cfg = EngineConfig.from_env()
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.kv_cache, KVCacheConfig)
    assert isinstance(cfg.server, ServerConfig)
    assert cfg.token_budget == 4096
    assert cfg.scheduler == "continuous"
    assert cfg.static_batch_size == 8
    assert cfg.max_concurrent_sequences == 64
    assert cfg.kv_allocator == "paged"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt2")
    monkeypatch.setenv("LLM_PORT", "9001")
    monkeypatch.setenv("LLM_TOKEN_BUDGET", "1024")
    monkeypatch.setenv("LLM_SCHEDULER", "static")
    monkeypatch.setenv("LLM_STATIC_BATCH_SIZE", "4")
    monkeypatch.setenv("LLM_MAX_CONCURRENT_SEQUENCES", "16")
    monkeypatch.setenv("LLM_KV_ALLOCATOR", "contiguous")
    cfg = EngineConfig.from_env()
    assert cfg.model.model_name_or_path == "gpt2"
    assert cfg.server.port == 9001
    assert cfg.token_budget == 1024
    assert cfg.scheduler == "static"
    assert cfg.static_batch_size == 4
    assert cfg.max_concurrent_sequences == 16
    assert cfg.kv_allocator == "contiguous"


def test_kv_cache_sizing_uses_kv_heads_not_query_heads():
    # GQA: bytes/token must scale with n_kv_heads, independent of any query-head count.
    cfg = KVCacheConfig(block_size=16, n_kv_heads=8, head_dim=128, n_layers=32, dtype_bytes=2)
    expected = 2 * 8 * 128 * 2 * 32
    assert cfg.bytes_per_token() == expected


def test_num_blocks_respects_memory_budget():
    cfg = KVCacheConfig(
        block_size=16, n_kv_heads=8, head_dim=128, n_layers=1, dtype_bytes=2,
        gpu_memory_utilization=1.0,
    )
    bytes_per_token = cfg.bytes_per_token()
    free_memory = bytes_per_token * 16 * 10  # exactly 10 blocks worth
    assert cfg.num_blocks(free_memory) == 10


class _FakeGQAConfig:
    num_hidden_layers = 16
    num_attention_heads = 32
    num_key_value_heads = 8
    hidden_size = 2048
    head_dim = 64


class _FakeMHAConfig:
    # No GQA / explicit head_dim fields — must fall back to num_attention_heads and
    # hidden_size // num_attention_heads rather than raising.
    num_hidden_layers = 12
    num_attention_heads = 12
    hidden_size = 768


def test_kv_cache_from_model_derives_gqa_shape_from_hf_config():
    # Shaped like Llama-3.2-1B: n_kv_heads/head_dim differ from the Llama-3-8B defaults.
    cfg = KVCacheConfig.from_model(_FakeGQAConfig())
    assert cfg.n_layers == 16
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 64


def test_kv_cache_from_model_falls_back_without_gqa_fields():
    cfg = KVCacheConfig.from_model(_FakeMHAConfig())
    assert cfg.n_layers == 12
    assert cfg.n_kv_heads == 12
    assert cfg.head_dim == 64


def test_kv_cache_from_model_env_override_wins(monkeypatch):
    monkeypatch.setenv("LLM_N_KV_HEADS", "4")
    cfg = KVCacheConfig.from_model(_FakeGQAConfig())
    assert cfg.n_kv_heads == 4
    assert cfg.head_dim == 64  # unrelated field still derived
