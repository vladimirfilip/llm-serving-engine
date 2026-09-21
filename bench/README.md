# bench

A benchmark harness for the single-GPU serving engine ("ours") against vLLM, SGLang and
TensorRT-LLM: one model, one dtype, one GPU, open-loop load, honest tail latency. Everything runs
as `python -m bench <command>` from the repo root, in the engine's own environment (`.venv/`).
The baselines run as subprocesses from their own environments and are never imported.

## Commands

```
bench env-check                      package inventory, GPU state, clock lock
bench prepare-data [--synthetic]     ShareGPT pool, WikiText-2, correctness prompts
bench bwprobe                        measured read/copy bandwidth (no engine running)
bench reference [--score ENGINE]     HF reference generations and perplexity, or score an engine
bench tune --engines ...             pick each engine's token budget (highest sharegpt tok/s)
bench run <suite> --engines ...      one suite: correctness probe sweep single_stream kernels
                                     nsys memory scheduler ablation precision coldstart soak
bench all --engines ... [--soak] [--quick]
bench report  --run-id ID            report.md and plots for a run
bench publish --run-id ID [--write-readme] [--force]
```

Run flags: `--run-id`, `--quick`, `--force`, `--allow-unlocked`, `--client-procs N`,
`--model-path`, `--config-dir`. A finished phase writes `<phase>/DONE` and is skipped on a rerun
with the same `--run-id`; sweeps and the capacity grid also resume at the point or cell. A phase
whose engine failed a MUST check is `partial`, not done.

## Setup

1. **Engine environment.** `pip install -e '.[dev,bench]'` in `.venv/`. `bench env-check` prints
   every package the harness imports, present or missing. Added to the optional `bench` group in
   `pyproject.toml`: numpy, pandas, pyarrow, pyyaml, aiohttp, orjson, matplotlib, scipy, psutil,
   nvidia-ml-py, datasets, lm-eval. **Missing or unusable in this environment:**
   - `flash-attn`: not installed (needs a source build; no wheel for torch 2.14 / sm_120).
   - `flashinfer-python`: installed, but its kernels JIT-compile and the compile fails here
     (`ptxas` rejects the PTX version the vendored headers generate). The kernel suite records it
     as failed at run time, by name, in `kernels/summary.json` and the report.
2. **Baselines.** `bench/scripts/setup_baselines.sh` creates `~/.bench-envs/{vllm,sglang,trtllm}`
   with `uv` and installs the versions in `bench/baselines.lock` (vllm 0.29.0, sglang 0.5.20,
   tensorrt_llm 1.2.1). Every launch flag in the engine YAMLs was checked against `--help` of the
   pinned versions, and each engine was launched and queried on this machine (all three return the
   same greedy completion as ours). What it took on an RTX 5070 (sm_120, driver 580):
   - **vLLM:** `VLLM_USE_FLASHINFER_SAMPLER=0`; its FlashInfer sampler JIT-compiles kernels,
     which do not build here.
   - **SGLang:** `--attention-backend=triton --sampling-backend=pytorch` (FlashInfer's JIT does not
     build here), `CUDA_HOME` and `PATH` into its `nvidia/cu13` tree, plus the `lib64` and
     `libcudart.so` links the script creates. It returns no token ids through the OpenAI
     endpoint, so its token-level correctness metrics are n/a.
   - **TensorRT-LLM:** system `libopenmpi3`; torch and torchvision `+cu130`; nvcc, nvvm, crt and
     cccl 13.0 matching that torch; the same two links; and a `sitecustomize` shim
     (`configs/shims/trtllm`) because its startup probe calls a `pynvml` confidential-compute
     query that segfaults on this driver. It rejects the OpenAI `logprobs` field and has no stats
     endpoint, so its token-level correctness metrics and KV/queue plots are n/a.
3. **Model and data.** `hf download unsloth/Llama-3.2-3B-Instruct --local-dir /models/Llama-3.2-3B-Instruct`,
   then `bench prepare-data`. `--synthetic` builds random datasets for CPU-only runs and stamps
   every report.
4. **GPU permissions.** Locking clocks needs root (`nvidia-smi -lgc/-lmc`). Undo it with
   `nvidia-smi -rgc -rmc`.

## Configs and the sizing decisions

`configs/hardware.yaml`, `model.yaml`, `suite.yaml` and `engines/*.yaml` hold every number the
harness uses; nothing else hard-codes a dimension, path, GPU name or SLO.

| Decision | Value | Why |
|---|---|---|
| Model | Llama-3.2-3B-Instruct, bf16 | the 8B model does not fit a 12 GB card |
| `max_model_len` | 16384 | vLLM refuses 32768 at 0.90 utilisation (weights leave ~30k KV tokens); `ctx32k` disables itself |
| `max_num_seqs` | 64 | at 256 the engine's own piecewise-graph self-check fails and it falls back to eager |
| `gpu_mem_util` | 0.90 | for ours it caps *total* device use (`LLM_DEVICE_MEM_UTIL`), like the baselines |
| Locked clock | 1900 MHz (the driver settles on its 1897 bin), memory 14001 MHz | held under a 250 W limit at ~124 W |
| Datasheet peaks | 672 GB/s, 61.75 TFLOP/s bf16 dense | 192-bit x 28 Gbps; 988 FP4-sparse AI TOPS / 16 |

The peaks were cross-checked on the card: the read probe reaches 94.8% of 672 GB/s, and a bf16
GEMM at the locked clock reaches 45 TFLOP/s against 61.75 x 1897/2510 = 46.7 TFLOP/s.
`peak_tflops_bf16_dense` is quoted at the boost clock, so MFU against it understates what the
locked clock allows; `env.json` also records `peak_tflops_bf16_at_locked_clock`.

## What was implemented for ours

- `engines/ours.yaml`: launches `bench.engines.ours_server` in the harness interpreter; engine
  settings go in as `LLM_*` environment variables (dtype, token budget, `max_num_seqs`,
  `LLM_DEVICE_MEM_UTIL`).
- `engines/ours_server.py`: the OpenAI-style `/v1/completions` wrapper (streaming SSE, one event
  per token, usage event, `logprobs=1`, `stop`, `skip_special_tokens`, `ignore_eos`),
  `/health`, `/internal/stats`, `/internal/score`.
- `kernels/ours_kernels.py`: the engine's own Triton kernels and `nn.Linear` calls, called the way
  the engine calls them.
- Engine hooks (`src/llm_serving_engine`): `submit_tokens`; per-token logprobs recorded only for
  requests that ask; `SamplingParams.ignore_eos`; `InferenceEngine.stats()` (H1) with
  `ModelRunner.memory_bytes()`; `ENGINE_NVTX=1` ranges `step`, `schedule`, `prepare_inputs`,
  `forward`, `sample`, `postprocess` (H2; a shared no-op context when unset); `ModelRunner.score`;
  `KVCacheConfig.device_memory_fraction`; `TokenizerWrapper` special-token flag.
- `configs/engines/ours_ablation.yaml` (H4): five cumulative steps built from flags the engine
  already has.

## A full, publishable run

```
python -m bench prepare-data
bash bench/scripts/setup_baselines.sh
python -m bench all --engines ours,vllm,sglang,trtllm --soak    # 5-7 h per engine plus 4 h soak
python -m bench publish --run-id <RUN_ID> --write-readme
```

`publish` refuses a `--quick` run, unlocked clocks, a dirty engine tree, an unevaluated or failed
correctness gate, an invalid sharegpt point, or an unfinished required phase. `--force` overrides
with a visible line in the README block.

## Where the spec's design met reality

Each item is a place the real interface or hardware forced a change; metric definitions are
unchanged unless stated.

- **Engine edits.** The engine needed changes beyond the marked hooks (token-id submission,
  logprobs, `ignore_eos`, `score`, the memory cap). Default behaviour is unchanged and the original
  test suite passes unmodified apart from one test that patched a function signature.
- **Ablation order.** Paged KV runs only on the fused kernels, and graphs replay only against a
  paged pool, so the steps are naive, +continuous batching, +fused kernels, +paged KV, +CUDA graphs.
- **Burst check compares gaps with the mean gap, not the median.** Once most gaps are burst gaps the
  median is one of them and nothing falls below a fraction of it, so the median criterion cannot
  fail a mock with `burst: 4`, which the spec requires it to.
- **Client null-server check.** 64 streams, send lag p99 < 5 ms, jitter p99 < 3 ms
  (`client_check` in `suite.yaml`) instead of 512 streams and 1 ms. This engine streams at most 64
  at once, and on this 9-vCPU KVM guest a raw-socket client (no HTTP library, no parsing) already
  sees 1.2 ms jitter p99 at 64 streams and 5.5 ms at 256, so 1 ms is below what the host can
  deliver to any client. ITL p99 therefore carries a ~2 ms host-noise floor here.
- **Client timing.** SSE events are timestamped per network chunk, and without logprobs token events
  are counted by substring rather than parsed; per-event CPU had to drop for the check above.
- **TTFT unit test.** The client-versus-mock test compares a long and a short prompt in one session
  (within 2 ms) because aiohttp plus loopback adds ~2 ms of fixed TTFT on this VM.
- **`skip_special_tokens: false`** is in every engine's `extra_body`. With `ignore_eos` a model keeps
  emitting special tokens after its natural end, which decode to empty text and would otherwise
  make the token-accounting check disable ITL for every engine.
- **GPU work in child processes.** Clock verification, the bandwidth probe, the HF reference and the
  kernel suite run via `bench.gpu_tasks`; a CUDA context in the harness would show as a compute
  process and block every engine launch. Server processes die with the harness (`PDEATHSIG`).
- **GPU idle check** runs before each GPU engine launch, waits up to 30 s for our own previous task
  to go quiet, and is recorded in `checks.json`.
- **`max_num_seqs` and grid lengths.** Lengths above `max_model_len` are `not run`, not `failed`.
- **Quick ablation** uses 1 capacity repeat and 2 batch-1 repeats.
- **Precision variants and soak** run on ours only; the soak windows are fractions of the run so
  the 10-60 min / 60 min-end comparison holds at any duration.

## Tests

`pytest` runs everything, including `cuda` tests on a GPU; `pytest -m "not cuda"` is the CPU-only
set. `bench/tests/test_all_mock.py` runs `bench all --engines mock --quick` end to end (about five
minutes) with synthetic data. The mock engine (`engines/mock_server.py`) and null server
(`engines/null_server.py`) exist only to test the harness.
