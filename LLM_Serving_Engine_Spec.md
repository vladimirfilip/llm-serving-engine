# LLM Serving Engine: Build Spec

Authoritative spec for Days 3–8 of the sprint, the flagship deliverable. Theory it leans on lives in `LLM_Systems_Technical_Reference.md` (continuous batching and PagedAttention: section 7; quantization: section 8; nsys and roofline: section 9; coordinated omission and per-stage latency: section 11) and this document doesn't re-derive it, it applies it. Kernels this engine wires in on Day 7 come from `FlashAttention2_Triton_Build_Guide.md` and Day 1's fused RMSNorm kernel.

**Division of labor**, stated once so it doesn't need repeating per section: you own the threading model, the scheduler, the KV-cache manager, and every latency-critical path. Claude Code owns the HTTP/gRPC server shell, tokenizer glue, model weight loading, the load generator's non-timing plumbing, dashboards, tests, and CI. The line is simple: if a bug in it would show up as a wrong *latency number*, you write it; if a bug in it would show up as a broken *endpoint*, delegate it.

**Target model:** one GPU, a decode-heavy autoregressive model, Llama-3.2-1B/3B or a 7B in int8. The model choice doesn't matter much to anything below; the engine around it does.

---

## 1. System architecture

Three threads, two queue hops, one design decision that justifies all of it.

```
                    ┌─────────────────────────────────────────────┐
  HTTP requests ──▶ │  IO thread (Python, asyncio)                 │
                    │  (Claude Code: server shell, tokenize)       │
                    │  owns: output_channels[seq_id] -> Queue      │
                    └───────────────────┬───────────────────────---┘
                                         │ MPSC ingress queue (C++, plain values)
                                         ▼
                    ┌───────────────────────────────────────────---┐
                    │  Scheduler thread  (yours, C++ SchedulerCore)│
                    │  - admission / eviction                      │
                    │  - KV block allocation                       │
                    │  - builds each iteration's batch-plan        │
                    │  - GIL released for the duration of step()   │
                    └──────┬─────────────────────────────▲────────-┘
                SPSC (C++) │                              │  SPSC (C++)
                    plan   ▼                              │  results
                    ┌───────────────────────────────────--┴────────┐
                    │  GPU worker thread (Python + wired kernels)  │
                    │  runs one forward iteration, returns tokens  │
                    │  dispatches tokens -> output_channels        │
                    └───────────────────────────────────────────---┘
```

Language boundary, stated plainly since it's easy to get backwards: the IO thread and the GPU worker's tensor-building glue are Python throughout. The scheduler and the two queues it touches are the part that moves to C++, and only once profiling says so, see below.

**Why three threads and not two.** The scheduler and the GPU worker could be the same thread: decide the batch, launch the kernels, block until done, repeat. The problem is that "block until done" wastes exactly the window where there's real CPU work to do, deciding *next* iteration's batch, draining the ingress queue, freeing blocks for sequences that just finished. Splitting them lets the scheduler prepare iteration N+1's plan while the GPU is still executing iteration N. This is the same overlap-compute-with-something-else instinct as async I/O, applied to the CPU-GPU boundary instead of the CPU-network boundary. It's a real, measurable throughput gain, and it's the reason this section exists before any of the others.

**On the queues, if you're building this in Python: the GIL changes what "lock-free" needs to mean, and it isn't automatically safe.** `collections.deque` and `queue.SimpleQueue` have GIL-atomic `append`/`popleft` at the C level, so for the SPSC hops (scheduler↔worker) a bare `deque` needs no additional lock; wrapping it in `threading.Lock` would be pure overhead. For MPSC ingress, `queue.SimpleQueue` is the correct off-the-shelf choice. The part that makes the three-thread architecture actually pipeline, not just look like it does on paper, is that PyTorch's CUDA calls release the GIL around genuinely blocking work (kernel launches return almost immediately; `stream.synchronize()` and similar are designed to drop the GIL while waiting so other threads can run Python bytecode in that window). This is the intended design, not an incidental detail, but it has had real historical bugs and inconsistencies in specific PyTorch codepaths, so treat it as something to *verify with your own `nsys` trace* (do you actually see scheduler-thread Python work overlapping the GPU worker's kernel execution window?), not something to assume blindly. If profiling shows the GIL is genuinely fighting you, the escape hatch is a C++/pybind11 extension for the scheduler and queues specifically, or running the scheduler as a separate OS process over shared memory. Don't reach for that on day one; reach for it only if the trace tells you to.

**The migration, precisely, once the trace does tell you to.** Writing the scheduler in C++ only delivers real parallelism if the C++ code actually drops the GIL while it runs. A pybind11 wrapper that takes Python objects, does work, and returns a Python object still holds the GIL for the whole call by default: faster per-call execution, same serialization as before. Two disciplines make the difference between that and a genuine fix.

State lives natively in C++, not as C++ code reaching back into Python lists and dicts: `running`, `waiting`, block tables, the free list, as `std::vector`/`std::deque`/plain structs owned by a C++ object, not wrapped Python containers. Then the GIL is released explicitly for the duration of real work (`py::gil_scoped_release` in pybind11) and reacquired only at the boundary, when data actually needs to cross. Queues carry plain value types across that boundary, not Python objects: a request crossing the ingress queue is primitive fields, seq id, token ids, sampling params, copied in, not a live refcounted Python object dragged across.

```cpp
class SchedulerCore {
public:
    void enqueue(Request req);            // called from the IO thread
    BatchPlan step();                     // called from the GPU-worker thread
    void report_results(IterResult res);  // called from the GPU-worker thread
};
```

```python
# GPU worker thread's loop, still Python
plan = scheduler.step()          # GIL released for the C++ work inside
tensors = build_tensors(plan)    # thin glue, stays Python
output = model.forward(tensors)  # GIL released for the actual CUDA work
scheduler.report_results(extract(output))
```

The discipline that matters most: cross the boundary once per iteration, not once per sequence. `step()` returning one batched plan for the whole iteration keeps the crossing cost fixed and small; a loop calling into C++ once per sequence in the batch reintroduces the exact per-call overhead this migration exists to remove, just relocated rather than eliminated. `build_tensors` and the result-extraction glue around `forward()` stay in Python for now, thin enough that whether they're worth moving is itself a question for the trace, not a guess made up front; if they turn out to matter, they're a natural extension of the same C++ layer, since the batch plan `step()` already produces is exactly what would get converted.

**Sequence state**, tracked by the scheduler for every request it knows about. Once the migration above happens, this is the shape of `SchedulerCore`'s native state, not a Python object:

```python
@dataclass
class Sequence:
    seq_id: int
    prompt_tokens: list[int]
    generated_tokens: list[int]
    block_table: BlockTable
    status: Literal["WAITING", "PREFILLING", "DECODING", "FINISHED"]
    prefill_progress: int          # tokens of the prompt processed so far
    sampling_params: SamplingParams
    arrival_time: float
    metrics: RequestMetrics        # section 6
```

No `output_channel` field. A live `asyncio.Queue` is a refcounted Python object with its own event-loop affinity; it has no business on a struct meant to live natively in C++, and it doesn't need to; the scheduler only ever needs `seq_id` to know who a result belongs to. Where the channel actually lives is next.

**Getting tokens back to the client.** Kept entirely on the Python side, in its own map, keyed by the same plain `seq_id` the C++ core already uses:

```python
output_channels: dict[int, asyncio.Queue] = {}   # seq_id -> per-request queue
```

The C++ core never sees this map. Flow: the IO thread creates a queue on request arrival, stores it under the new `seq_id`, and streams from it (SSE, one queue-get per chunk) while awaiting. The GPU worker thread, after each `forward()`, has `(seq_id, token)` pairs for the iteration and pushes each into its matching queue.

The part that isn't optional: `asyncio.Queue` isn't thread-safe to push into directly from a thread that isn't running its event loop, and the GPU worker thread is a plain OS thread, not the loop. `queue.put_nowait()` called directly from it is a race. The fix is `loop.call_soon_threadsafe(...)`, with the loop reference captured once at startup (one async IO thread handling all connections via coroutines, not one OS thread per connection, so there's exactly one loop to target).

A trap worth naming, because the natural-looking version of this fails silently: `call_soon_threadsafe` schedules a call and returns immediately, it doesn't run inline. Wrapping it in `try/except QueueFull` on the GPU worker thread catches nothing, since the exception happens later, on the event loop, inside a callback your `try` already returned past. The `except` has to live inside the function that actually runs on the loop:

```python
def _safe_put(q: asyncio.Queue, item):
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass   # slow client; drop rather than block the GPU worker

def dispatch_results(iter_results: list[tuple[int, int, bool]]):  # (seq_id, token, finished)
    for seq_id, token, finished in iter_results:
        q = output_channels.get(seq_id)
        if q is None:
            continue                      # client already disconnected
        loop.call_soon_threadsafe(_safe_put, q, token)
        if finished:
            loop.call_soon_threadsafe(_safe_put, q, DONE)
```

Bound the queue (`asyncio.Queue(maxsize=64)`) and drop on full rather than growing unbounded, for the same reason `TOKEN_BUDGET` exists in section 2: one slow HTTP client reading its stream too slowly should never stall the GPU worker's iteration for every other sequence decoding alongside it. That's head-of-line blocking again, one layer up, at the network boundary instead of the compute boundary.

Receiving side, the stream generator consumes until the sentinel and cleans up its own entry, which is what makes `if q is None: continue` above correct rather than a silent-drop bug on disconnect:

```python
async def sse_stream(seq_id, q):
    try:
        while True:
            item = await q.get()
            if item is DONE:
                break
            yield format_sse(item)
    finally:
        output_channels.pop(seq_id, None)
```

**Day 3 deliverable:** the three threads running, a single sequence going end to end through all of them (no batching, no paging yet, just a working pipe), and a baseline p50/p99 for one request at a time. That number is what every later day's improvement gets measured against.

---

## 2. The scheduler and continuous batching

The core algorithm, run once per iteration. Read it in the order it's written; the ordering encodes the priority policy.

```python
TOKEN_BUDGET = 4096   # tune this: total new tokens (prefill+decode) admitted per iteration

def scheduler_step(running: list[Sequence], waiting: deque[Sequence],
                    allocator: BlockAllocator) -> BatchPlan:
    budget = TOKEN_BUDGET
    plan = BatchPlan()

    # 1. Already-decoding sequences go first. They're mid-generation and a
    #    client is already waiting on their next token; stalling them to let
    #    a new request in is the exact head-of-line-blocking failure mode
    #    continuous batching exists to avoid, just inflicted on someone else.
    for seq in running:
        if seq.status == "DECODING":
            plan.add(seq, n_tokens=1)
            budget -= 1

    # 2. Sequences mid-chunked-prefill continue where they left off.
    for seq in running:
        if seq.status == "PREFILLING" and budget > 0:
            remaining = len(seq.prompt_tokens) - seq.prefill_progress
            chunk = min(remaining, budget)
            plan.add(seq, n_tokens=chunk, is_prefill_chunk=True)
            seq.prefill_progress += chunk
            budget -= chunk
            if seq.prefill_progress == len(seq.prompt_tokens):
                seq.status = "DECODING"   # finishes decoding next iteration

    # 3. Admit new work only with whatever budget and KV capacity remain.
    while waiting and budget > 0:
        seq = waiting[0]
        if not allocator.has_capacity(seq, tokens=min(len(seq.prompt_tokens), budget)):
            break   # don't skip the line; back off and try again next iteration
        chunk = min(len(seq.prompt_tokens), budget)
        allocator.ensure_capacity(seq.block_table, chunk)
        plan.add(seq, n_tokens=chunk, is_prefill_chunk=(chunk < len(seq.prompt_tokens)))
        seq.prefill_progress = chunk
        seq.status = "DECODING" if chunk == len(seq.prompt_tokens) else "PREFILLING"
        running.append(waiting.popleft())
        budget -= chunk

    return plan
```

**`TOKEN_BUDGET` is chunked prefill's entire mechanism**, stated as one number. A prefill without this cap would consume the whole iteration's compute and spike inter-token latency for every sequence currently decoding alongside it, since prefill is compute-bound and decode is memory-bound and mixing them unbounded means one iteration does a full prompt's worth of matmul while everyone else's next token waits behind it. Capping total tokens per iteration and letting a long prompt spill across several iterations' worth of chunks bounds that spike. Tune the number empirically: too high and you're back to the spike; too low and prefill for long prompts drags out, delaying time-to-first-token. Start around 2–4x your target batch size in decode tokens and adjust from what the Day 8 latency decomposition shows you.

**Result handling**, immediately on receiving the GPU worker's output for an iteration: append each sequence's new token, push it to that sequence's output channel, check EOS/`max_tokens`, and for anything finished, free its blocks and drop it from `running` before building the next plan. Do this before step 1 of the next `scheduler_step` call, not deferred, so a block frees the same iteration a sequence finishes rather than sitting reserved for one extra loop.

**Ablation to run once this works:** static batching (wait for a fixed batch, run every sequence to the max length in it, no admit/evict) versus this. That throughput delta is your Day 4–5 headline number.

---

## 3. Paged KV cache

**Physical storage**, one tensor per layer (or one tensor with a layer dimension, your call), shape:

```
(num_blocks, block_size, n_kv_heads, head_dim)     # separately for K and V
```

This layout is simple and buildable in the time available: contiguous per block, per token, per head, reasonably coalesced. Production engines apply more specialized layouts tuned to their specific attention kernel's access pattern; that's a real further optimization, not a correctness requirement, and out of scope here.

**Sizing the pool.** Bytes per token per layer is `2 (K and V) × n_kv_heads × head_dim × dtype_bytes` (reference doc, section 6, this is exactly the GQA-driven number, not the query head count). Multiply by `n_layers` for the full per-token cost, divide your memory budget by that and by `block_size` to get `num_blocks`. Reserve real headroom for the model weights themselves and for activation memory before allocating everything else to the KV pool.

**Block table and allocator:**

```python
@dataclass
class BlockTable:
    physical_blocks: list[int]   # logical block 0, 1, 2, ... → physical block id
    num_tokens: int

class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.block_size = block_size

    def has_capacity(self, seq: Sequence, tokens: int) -> bool:
        needed = ceil((seq.block_table.num_tokens + tokens) / self.block_size) \
                 - len(seq.block_table.physical_blocks)
        return needed <= len(self.free_blocks)

    def ensure_capacity(self, table: BlockTable, new_tokens: int):
        table.num_tokens += new_tokens
        blocks_needed = ceil(table.num_tokens / self.block_size)
        while len(table.physical_blocks) < blocks_needed:
            table.physical_blocks.append(self.free_blocks.popleft())

    def free(self, table: BlockTable):
        self.free_blocks.extend(table.physical_blocks)
        table.physical_blocks.clear()
```

**What the attention kernel needs from this.** Standard attention assumes a flat, contiguous KV buffer; paging means it has to gather from `block_table.physical_blocks` instead. This is the one piece of Day 6 that genuinely requires a modified kernel, not just data-structure bookkeeping, since the memory access is now indirect. Reusing FA-2's structure with block-indirect loads instead of a single contiguous load is the right starting point; writing this from scratch is a meaningfully bigger task than the block bookkeeping above, budget for it accordingly. If time is short, a correctness-first fallback is to gather each sequence's blocks into a temporary contiguous buffer before calling your existing FA-2 kernel unmodified, pay the gather cost, and only write the indirect-load kernel version if profiling shows the gather actually matters.

**Scope cut, stated plainly:** no preemption and no prefix sharing. If the pool runs out of free blocks, new admissions simply wait (the `has_capacity` check in section 2 already does this), and running sequences are never evicted mid-generation to make room. Real engines add both; both are real additional projects, not a missing bookkeeping detail, and out of scope for six days.

**Ablation:** max concurrent sequences (or max context length) under naive per-sequence max-length reservation versus under paging, at the same GPU memory budget. That gap is your Day 6 headline number.

---

## 4. Kernels and quantization on the hot path

Day 7 is integration, not new design. Two things to wire in, one thing to measure.

**Wiring in the kernels.** Whatever model-loading code Claude Code generates almost certainly builds a standard framework model (a HuggingFace `LlamaForCausalLM` or similar). The lowest-friction way to substitute your own kernels is to monkeypatch or subclass the specific submodules: replace the attention module's forward with a call into your FA-2 kernel (paged, per section 3) plus your Day-1 fused RMSNorm-and-residual kernel at each sublayer boundary. You don't need to rewrite the rest of the forward pass, only the two spots you've built custom kernels for.

**int8 on the hot path.** This is weight-only quantization (reference doc, section 8): quantize weights to int8 once at load time, dequantize on the fly at (or fused into) each matmul. It buys decode speed in direct proportion to bytes saved streaming weights from HBM, not from faster arithmetic, because decode is memory-bound (section 6), not compute-bound. Don't expect it to move prefill much; that's the compute-bound phase, where weight-only quantization has nothing to offer.

**Profiling workflow.** Run the full engine under `nsys` at a realistic batch size and sequence length mix. First question: does wall-clock decode-step time match what your kernel-level `ncu` benchmarks (from Day 1's kernel work) would predict? If not, the reference doc's section 9 answer applies directly: small-batch decode means many small kernel launches per step, and the gap is usually launch overhead or an unnecessary sync point, visible as *gaps* between GPU activity in the nsys timeline, not as a slow kernel. Fix what the trace actually shows, not what seems most interesting to optimize.

**If the trace confirms launch overhead specifically: CUDA graph capture, before considering a rewrite of the model-execution path.** The forward pass itself, the matmuls, the attention kernel, is compiled CUDA regardless of what language calls it; a C++ (or LibTorch) rewrite of this path wouldn't make the GPU compute faster, since the GPU doesn't know which language issued the launch. What both languages pay is per-call dispatch overhead into each op, argument marshaling, PyTorch's op dispatch, and decode does dozens of small launches per step. `torch.cuda.graph` capture records that launch sequence once and replays it as a single low-overhead launch, addressing the actual cost directly, inside Python, for a fraction of the effort a rewrite would take.

The complication: continuous batching means batch composition changes every iteration, and a graph needs fixed shapes. The standard answer, and what vLLM's own model runner does, is to capture a separate graph for each of a bounded set of batch sizes (1, 2, 4, 8, 16, 24, 32, ... up to some max) and pad the real batch up to the nearest captured size each iteration. It's a real effect, not a marginal one: a from-scratch decode implementation without graph capture, measured against vLLM with it on comparable hardware, showed roughly 1.9x to 3x throughput and a comparable drop in per-token latency, with the gap tracking fixed per-step launch count rather than per-step compute, exactly the signature of launch overhead rather than kernel quality. Try this before touching the model-execution language.

---

## 5. Latency engineering pass

Four concrete things, in the order they're worth doing.

**Thread pinning.** `os.sched_setaffinity(0, {core_id})` pins the *calling* thread's underlying OS thread to a specific core (Linux only). Pin the scheduler thread and the GPU worker's CPU-side driver thread to separate physical cores, ideally on the same NUMA node as the GPU. This reduces cross-core migration and keeps each thread's working set warm in its own cache, which matters most for the scheduler's per-iteration bookkeeping, the tightest, most frequently executed loop in the whole system.

**Hot-path allocation.** Pool-allocate `Sequence` and `RequestMetrics` objects (a free-list reused across requests) rather than constructing and garbage-collecting one per request. In Python this won't get you to zero allocations the way it would in C++, but it removes the specific allocator churn that shows up as latency spikes under load: reuse objects for the lifetime of the pool instead of letting the GC reclaim a burst of them at once.

**Per-stage latency decomposition.** Instrument every boundary, and store it per request so a bad tail can be attributed to a stage, not guessed at:

```python
@dataclass
class RequestMetrics:
    enqueue_time: float
    admit_time: float | None = None
    first_token_time: float | None = None
    token_times: list[float] = field(default_factory=list)
    done_time: float | None = None

    @property
    def schedule_latency(self): return self.admit_time - self.enqueue_time
    @property
    def prefill_latency(self): return self.first_token_time - self.admit_time
    @property
    def inter_token_latencies(self):
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]
```

**Coordinated-omission-aware load generator.** The one piece of infrastructure you should not delegate to Claude Code even though it looks like plumbing, because getting it subtly wrong silently reintroduces the exact measurement bug it exists to prevent. Open-loop, fixed arrival schedule, timestamps recorded *before* the send, not after:

```python
async def open_loop_load_gen(target_qps: float, duration_s: float, send_fn):
    next_send = time.monotonic()
    end = next_send + duration_s
    results = []
    while next_send < end:
        now = time.monotonic()
        if now < next_send:
            await asyncio.sleep(next_send - now)
        intended_send_time = next_send          # record BEFORE sending, this is the fix
        asyncio.create_task(_send_and_time(send_fn, intended_send_time, results))
        next_send += random.expovariate(target_qps)   # Poisson arrivals
    return results

async def _send_and_time(send_fn, intended_send_time, results):
    result = await send_fn()
    results.append({"latency": time.monotonic() - intended_send_time, **result})
```

The critical line is `intended_send_time`, recorded from the fixed schedule, not from whenever the request actually went out. Timing from actual-send would silently absorb sender-side delay into a smaller sample of requests rather than reporting it, and the resulting p99 lies to you by construction. Measuring from the intended schedule is what makes a stall show up as a cluster of high latencies instead of vanishing from the sample.

---

## 6. Benchmarks worth running

Every ablation below isolates exactly one thing this spec built, so each one should map to a specific section above:

- Throughput–latency Pareto: sweep offered QPS, plot p50/p99 vs. achieved throughput.
- Static vs. continuous batching (section 2).
- Contiguous max-length reservation vs. paged KV cache, at matched memory (section 3).
- fp16 vs. int8 weights, decode-step latency specifically (section 4).
- With and without your fused kernels, end-to-end (section 4).

---

## 7. Done checklist

- [ ] Single request travels IO → scheduler → GPU worker → back, end to end
- [ ] Continuous batching admits and evicts mid-flight, verified by watching batch composition change across iterations
- [ ] Chunked prefill: a long prompt's admission doesn't spike inter-token latency for concurrently decoding sequences (check this directly, don't assume it from the code)
- [ ] Paged KV cache: no fragmentation-driven OOM at a token count where the naive reservation baseline would have failed
- [ ] Day 1/2 kernels wired into the real forward pass, not just benchmarked standalone
- [ ] int8 weight path measurably faster on decode, roughly flat on prefill (if it isn't, something's wrong with which phase you're measuring)
- [ ] `nsys` trace taken, at least one real gap or stall identified and explained, not just captured
- [ ] Threads pinned; verified with `taskset -p` or equivalent, not just called
- [ ] Load generator is open-loop; a deliberate artificial stall shows up as a latency cluster, not as fewer completed requests
- [ ] Per-stage latency breakdown available per request, not just an aggregate p99
- [ ] If the scheduler was migrated to C++: verified with a trace that the GIL is actually released during `step()`, not just that the code compiles as C++
- [ ] Token delivery uses `call_soon_threadsafe`, never a direct cross-thread `put_nowait`
- [ ] Output queues are bounded; a deliberately slow client doesn't stall the GPU worker's iteration for other sequences

---

## 8. Failure modes, ranked by how much time they cost

**Throughput looks fine, p99 looks great, but the demo "feels" laggy under load.** The load generator is timing from actual-send instead of the fixed schedule, so stalls get absorbed instead of reported. You're not measuring what you think you're measuring; see section 5.

**Batch composition never changes between iterations.** Eviction isn't running, or it's deferred past when it should fire. Check that finished sequences are freed the same iteration their EOS is detected, in the result-handling step before `scheduler_step` is called again, not on some later pass.

**Latency spikes precisely when a long prompt arrives.** `TOKEN_BUDGET` is unset, too high, or the admission loop isn't actually chunking, it's admitting the whole prompt at once. Log `plan` per iteration and confirm chunk sizes are what you expect.

**KV pool exhausts far sooner than the sizing math predicted.** Almost always block-size accounting using query-head count instead of KV-head count (reference doc, section 6, this is the GQA distinction). Recompute from `n_kv_heads`, not `n_heads`.

**Scheduler thread doesn't seem to make progress while the GPU worker is running.** The GIL-release assumption from section 1 isn't holding for your specific codepath. Confirm with an `nsys` trace showing (or not showing) scheduler-thread CPU work overlapping the GPU worker's kernel window before reaching for a C++ rewrite; it may be a different bottleneck entirely (e.g., the scheduler itself blocked on something).

**int8 path is slower than fp16.** Almost certainly measured on prefill, or the dequantization itself is unfused and adds more overhead than the bandwidth savings recover. Isolate decode-step latency specifically before concluding the kernel is wrong.

**C++ scheduler shows no speedup over the Python version.** The GIL probably isn't actually being released. Check for `py::gil_scoped_release` around the real work specifically, not just present somewhere in the file, and check that the scheduler's state is native C++ containers rather than C++ code still marshaling Python lists and dicts on every call, which pays the GIL cost you moved to C++ to avoid.

**Tokens for one client silently stop arriving, no error anywhere.** Almost always the `try/except QueueFull` living in the wrong place, on the calling thread around `call_soon_threadsafe` itself, where it catches nothing, rather than inside the callback that actually runs on the event loop. Second most likely: the client disconnected and `output_channels` still holds a stale entry because the stream generator's `finally` didn't run.

**Everything works alone, nothing works together.** Integration order matters: get section 1 (skeleton) solid and load-tested before adding section 2 (batching); get batching solid before adding section 3 (paging); add section 4's kernels last, once the system around them is already correct, so a new wrong number has exactly one likely cause.
