import json

import pytest

from bench.modelspec import ModelSpec

LLAMA_31_8B = dict(
    hidden_size=4096, intermediate_size=14336, num_hidden_layers=32, num_attention_heads=32,
    num_key_value_heads=8, vocab_size=128256,
)


@pytest.fixture
def spec(tmp_path) -> ModelSpec:
    (tmp_path / "config.json").write_text(json.dumps(LLAMA_31_8B))
    return ModelSpec.from_dir(tmp_path, "bfloat16")


def test_kv_bytes_scale_with_kv_heads_not_query_heads(spec):
    assert spec.head_dim == 128
    assert spec.kv_bytes_per_token == 131072
    assert spec.kv_bytes_per_tok_layer * spec.n_layers == spec.kv_bytes_per_token


def test_decode_weight_bytes_match_the_known_checkpoint_size(spec):
    assert spec.weight_bytes_decode == pytest.approx(15.01e9, rel=0.005)


def test_bound_tok_s_falls_as_context_grows(spec):
    assert spec.bound_tok_s(0, 672) == pytest.approx(672e9 / spec.weight_bytes_decode)
    assert spec.bound_tok_s(8192, 672) < spec.bound_tok_s(512, 672)


def test_prefill_flops_grow_quadratically_only_in_attention(spec):
    matmul_only = 2 * spec.n_layers * spec.layer_matmul_params * 1024 + 2 * spec.vocab * spec.d
    attention = spec.n_layers * 2 * 1024 * 1024 * spec.n_heads * spec.head_dim
    assert spec.prefill_flops(1024) == matmul_only + attention


def test_explicit_head_dim_overrides_the_derived_one():
    spec = ModelSpec.from_config({**LLAMA_31_8B, "head_dim": 64}, "bfloat16")
    assert spec.head_dim == 64


def test_capture_env_serialises_the_model_spec(tmp_path):
    from bench.config import Config, load_config
    from bench.env import ClockLock, capture_env

    (tmp_path / "config.json").write_text(json.dumps(LLAMA_31_8B))
    cfg = load_config()
    cfg = Config(cfg.hardware, cfg.model | {"local_path": str(tmp_path)}, cfg.suite)
    env = capture_env(cfg, ClockLock(True, "", 1897), {"bw_read_gbs": 600.0, "bw_copy_gbs": 500.0})
    assert env["model"]["spec"]["n_kv_heads"] == 8
    assert env["bw_read_pct_of_datasheet"] == pytest.approx(100 * 600 / 672)
    assert env["peak_tflops_bf16_at_locked_clock"] == pytest.approx(61.75 * 1897 / 2510)
    json.dumps(env)
