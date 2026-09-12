# 10-Day Sprint — Jump Campus ML Research Engineer (Intern)

**Assumptions baked in:** you have CUDA fundamentals; strong OS / systems / architecture / C++; AI (Claude Code) writes all boilerplate; you build fast (ITCH feed handler ≈ 10 hrs). Calibrated to ~5–6 focused hrs/day. Single GPU is sufficient throughout.

---

## The bet

Your edge over other ML-RE candidates isn't modelling — it's **latency engineering at HFT grade.** Most people who can train a transformer cannot write a lock-free scheduler or reason about p999 tails and coordinated omission. You can (ITCH proves it). So the whole sprint produces **one flagship that transfers that skill directly to ML serving**, wrapped with unmistakable GPU-kernel signal.

**Flagship:** a low-latency, high-throughput **LLM inference engine, built like a trading system.** It hits four JD lines at once — *integrate ML into production where latency matters* · *high-throughput inference* · *GPU/accelerator programming* · *C/C++/Python/CUDA* — and it's the artifact a Jump interviewer remembers.

The training/HPC side of the JD you already cover with your existing **Distributed AI Training System** (parameter server, synchronous SGD, scaling analysis). Don't rebuild it — turn it into an interview talking point (see Day 10) and, only if you're ahead, do the [stretch](#stretch-only-if-ahead).

---

## Days at a glance

| Day | Focus | Your work | Output |
|---|---|---|---|
| 1 | Fused CUDA kernel | Write + profile 1 raw-CUDA fused kernel | `ncu`-profiled kernel vs baseline |
| 2 | FlashAttention-2 (Triton) | CS336 A2 pt.1 | FA-2 kernel, benchmarked vs SDPA |
| 3 | Engine skeleton + threading | Design the lock-free pipeline; MVP generate | Baseline p50/p99, threading model |
| 4–5 | Continuous batching + scheduler | The core systems work | In-flight batching, throughput uplift |
| 6 | Paged KV-cache | Block-paged memory manager | Memory-efficient long-context serving |
| 7 | Kernels + int8 on hot path | Wire in Day 1–2 kernels; quantize | End-to-end `nsys` profile |
| 8 | Latency engineering pass | Pin, pool, decompose, tail-hunt | p50/p99/p999 + per-stage breakdown |
| 9 | Benchmarks + writeup | Ablations, plots, README, blog | Throughput–latency Pareto, blog post |
| 10 | Interview prep | Narratives, whiteboard, gap answers | 3 crisp stories, prepped Q&A |

---

## Phase 1 — GPU kernel signal (Days 1–2)

Skip fundamentals. Produce two ML-relevant kernels that (a) prove you write real GPU code and (b) become hot-path components of the engine.

**Day 1 — one hand-written CUDA kernel.** Raw CUDA C++ for credibility (Triton alone can read as "never wrote a real kernel"). Pick one that the engine will actually use: a **fused dequant → int8 GEMM epilogue**, or **fused RMSNorm + residual add**. Profile in `ncu`; iterate coalescing / shared-mem / occupancy; record each speedup with the profile that justified it. *(AI: build harness, correctness tests, benchmark scaffold.)*

**Day 2 — FlashAttention-2 in Triton.** Do **CS336 Assignment 2, part 1** — it hands you a tested, gold-standard FA-2-in-Triton exercise. Benchmark vs PyTorch SDPA. Be able to explain *why* it's IO-aware (a stock interview question). *(AI: benchmarking + plotting.)*

**Output:** `gpu-kernels` repo — both kernels, benchmarks vs cuBLAS/SDPA, `ncu` profiles, one-paragraph optimization writeup each.
**Resume line:** *"Hand-wrote a fused int8-GEMM CUDA kernel and implemented FlashAttention-2 in Triton; N× over naive / matched SDPA."*

---

## Phase 2 — Flagship inference engine (Days 3–8)

Target a decode-heavy autoregressive model that fits one GPU (Llama-3.2-1B/3B, or a 7B in int8). The point isn't the model — it's the serving system around it. **Build it like the ITCH handler:** the same instincts (SPSC/MPSC rings, allocation-free hot path, cache-aware layout, honest tail measurement) are exactly what makes this stand out.

**Claude Code does all plumbing:** HTTP/gRPC server, tokenizer glue, model loading, config, the load generator, Prometheus/Grafana dashboards, the test suite, CI. **You own the systems core:** the threading model, the scheduler, batching policy, KV-cache manager, and the latency-critical paths. That division is what makes 6 days enough — and it's the part interviews probe.

**Day 3 — skeleton + threading model.** Stand up single-request `generate` and a baseline. Design the pipeline you know cold: a **network/IO thread ↔ scheduler thread ↔ GPU worker**, connected by **lock-free queues** (your SPSC ring, generalized to MPSC for ingress). Pool-allocate request objects; zero allocations on the steady-state path. Record baseline p50/p99.

**Days 4–5 — continuous (in-flight) batching + scheduler.** The core contribution. Admit and evict sequences from the running batch each step instead of static batch-and-wait; handle ragged sequence lengths; implement an admission/scheduling policy (FCFS first, then a length- or deadline-aware variant). **Measure throughput uplift vs static batching** — this is your headline number.

**Day 6 — paged KV-cache.** PagedAttention-style block paging: a block allocator, per-sequence block tables, no big contiguous per-request reservation. This is a memory-management problem, your wheelhouse. Show the max-concurrency / long-context win over naive contiguous KV.

**Day 7 — kernels + quantization on the hot path.** Wire in your Day 1–2 kernels (FA-2 for attention, fused int8 GEMM for the projection/MLP path). Add an int8 weight path. Profile end-to-end with `nsys`; find and kill the real bottleneck (likely launch overhead, H2D, or a sync point).

**Day 8 — latency engineering pass (your differentiator).** The day that makes this a *Jump* project. Pin threads / isolate cores; strip remaining hot-path allocations and false sharing; build a **coordinated-omission-aware load generator** (measure true p99/p999 under a fixed offered load, not just closed-loop). Produce a **per-stage latency decomposition** — enqueue → schedule → prefill → decode-step → detokenize → response — the LLM-serving analogue of your ITCH t0→t3. Hunt the tail.

**Output:** `llm-serving-engine` repo.
**Resume line:** *"Built a low-latency LLM inference engine (lock-free scheduling, continuous batching, paged KV-cache, int8 + custom CUDA/Triton kernels): X req/s at p99 < Y ms on one GPU, Z× over static batching; per-stage latency decomposition and coordinated-omission-aware benchmarking."*

---

## Phase 3 — Prove it and pitch it (Days 9–10)

**Day 9 — benchmarks + writeup.** The numbers are half the value; the *analysis* is the other half. Produce: a **throughput–latency Pareto** (sweep offered QPS), **p50/p99/p999 vs load**, and **ablations** — static vs continuous batching, contiguous vs paged KV, fp16 vs int8, with/without your fused kernel. Clean README with plots + a short blog post ("HFT-grade latency engineering for LLM inference"). GPU strategy: dev on Colab/Kaggle; headline run on the university GPU; ~£20–40 cloud spot (Lambda/RunPod) as insurance for one bigger-model or A100 number.

**Day 10 — interview prep.** With a strong resume + one killer project, interview performance is now the marginal lever.
- **Three 2-minute STAR narratives**, each ending in a number: the engine, the kernels, and the ITCH handler (still your best pure-systems story).
- **Whiteboard-ready:** memory hierarchy & coalescing, occupancy, why FlashAttention is IO-aware, continuous vs static batching, paged KV-cache, coordinated omission, roofline. Plus the classics you already know: lock-free queues, memory ordering (acquire/release), false sharing.
- **The scale gap, pre-answered.** If asked "how would you take this to hundreds of TBs / multi-node?", answer in the vocabulary of the **Ultra-Scale Playbook** (data/tensor/pipeline parallelism, ZeRO stages, hierarchical all-reduce, compute/comm overlap) and point to your existing distributed-training project as the training-side counterpart.
- **Low-volume Codeforces** (1–2 problems) as warm-up — you're already Expert; this is just staying sharp for a live-coding round.

---

## Stretch (only if ahead)

If the engine lands early, the highest-value add is modernizing your **existing** distributed-training repo rather than starting anything new: swap the parameter server for **ring all-reduce**, add **ZeRO-1** optimizer-state sharding (CS336 A2 part 2 gives you the pattern), and re-run the scaling curves. One day, reuses existing code, upgrades a resume bullet, and rounds out the training/HPC side.

---

## What this covers vs the JD

| JD requirement | Covered by |
|---|---|
| C / C++ / Python / CUDA | Kernels + engine (C++/CUDA/Python) |
| GPU / accelerator programming (CUDA, Triton, …) | Days 1–2 + hot-path kernels |
| Integrate ML into production where latency matters | Entire flagship |
| High-throughput inference, low latency | Continuous batching + latency pass |
| Build observable, performant, flexible ML systems | Metrics/dashboards + benchmarks + ablations |
| ML systems at large scale | Serving throughput now; training scale via existing repo + Day-10 vocabulary |
| PyTorch / DL library | Engine + CS336 A2 |
| Reduce research iteration cycle time | Framed in the engine's benchmarking/observability story |

---

## Triage (if a day slips)

Drop from the bottom: (5) int8/quant polish → (4) paged KV-cache → (3) Day-1 raw CUDA kernel (Triton FA-2 alone still shows GPU work) → **protect: continuous batching + latency pass + benchmarks** (the differentiator) → **never drop: a working engine MVP with an honest p99 number.**

---

## Resource index (verified current)

| Resource | Use | Link |
|---|---|---|
| CS336 (LM from Scratch) — Assignment 2 | FA-2 in Triton (+ DDP/ZeRO for stretch) | https://cs336.stanford.edu/ · repos: https://github.com/stanford-cs336 |
| Triton tutorials | Kernel authoring | https://triton-lang.org/ |
| GPU MODE lectures | FlashAttention / profiling reference | https://github.com/gpu-mode/lectures |
| Ultra-Scale Playbook | Scale vocabulary for interviews | https://huggingface.co/spaces/nanotron/ultrascale-playbook |
| Nsight Compute / Systems | `ncu` kernels, `nsys` end-to-end | NVIDIA Nsight |
