# CLAUDE.md

## Project

A single-GPU LLM serving engine built like a trading system: lock-free queues, continuous
batching, paged KV cache, custom CUDA/Triton kernels on the hot path, and honest tail-latency
measurement.

Authoritative documents, in precedence order:

1. [LLM_Serving_Engine_Spec.md](LLM_Serving_Engine_Spec.md) — the design. Architecture, scheduler
   algorithm, allocator, benchmarks, failure modes. Follow it; it is not a sketch.
2. [LLM_Systems_Technical_Reference.md](LLM_Systems_Technical_Reference.md) — the theory the spec
   applies (continuous batching §7, quantization §8, nsys/roofline §9, coordinated omission §11).
3. [Jump_ML_RE_10Day_Sprint (1).md](Jump_ML_RE_10Day_Sprint%20(1).md) — day-by-day roadmap and
   triage order if a day slips.

When code and spec disagree, say so rather than silently picking one.

## Division of labor

**You write:** HTTP/SSE server shell, tokenizer glue, model weight loading, config, the load
generator's non-timing plumbing, metrics export and dashboards, tests, CI, plotting.

**The user writes:** the threading model, the scheduler, the KV-cache manager, the C++ core and its
queues, and the load generator's timing logic. Anything on a latency-critical path.

The line: a bug in your code shows up as a **broken endpoint**; a bug in theirs shows up as a
**wrong latency number**. Do not write the second kind unscaffolded — propose it and let the user
take it, or write a clearly-marked placeholder they will replace.

## Code style

> "Code is just a sequence of invariants and transformations."

- Terse and maximally readable. No line should need deciphering to see what it does.
- Name things after the invariant they maintain or the transformation they perform.
- Straight-line where possible: compute, then branch, then mutate. No cleverness that buys nothing.
- No defensive scaffolding nobody asked for: no speculative abstraction layers, no config knobs with
  one caller, no try/except that swallows a bug, no compatibility shims for versions we don't use.
- Type hints on signatures. Dataclasses for state. `dataclass(slots=True)` on anything per-request.
- Match the spec's names exactly (`Sequence`, `BlockTable`, `BlockAllocator`, `BatchPlan`,
  `RequestMetrics`, `TOKEN_BUDGET`, `output_channels`) so code and spec read as one document.
- Delete code rather than commenting it out. Git remembers.

## Comments

Comments supplement readable code; they are not a memory dump.

- Write them for what the code cannot say: why this invariant holds, why this ordering, units and
  layouts of tensors — grounded entirely in the code they sit next to.
- Never cite an external document as the reason for a shape: no "spec section N", no filenames like
  `LLM_Serving_Engine_Spec.md`, no `CLAUDE.md`, no "per the spec/design doc". If a constraint from
  one of those documents matters, restate the constraint itself in the comment — the code and the
  comment must stand on their own without the reader opening another file.
- No process or ownership narration either: no "user territory", "division of labor", "Day N
  deliverable", "TODO(user)", "placeholder for the scheduler owner" — say what the code does or
  raises (e.g. `NotImplementedError("int8 weight-only quantization")`), not who is meant to write it
  or why the org chart put it there.
- Do not narrate the code, log your debugging history, tag previous bugs, or leave "changed X to Y"
  notes. No `# TODO(claude)`, no section-banner ASCII art.
- One line usually suffices. If a comment needs a paragraph, the code is wrong.

## Invariants that are easy to break silently

These come from the spec's failure-mode list; check any code you touch against them.

- Tokens cross into the event loop only via `loop.call_soon_threadsafe`; never a direct
  `put_nowait` from the GPU worker thread.
- `except asyncio.QueueFull` lives *inside* the callback that runs on the loop, not around
  `call_soon_threadsafe`.
- Output queues are bounded and drop on full. A slow client never stalls an iteration.
- The stream generator's `finally` pops its `output_channels` entry.
- KV sizing uses `n_kv_heads`, never `n_heads` (GQA).
- Finished sequences are freed in result handling, before the next `scheduler_step`.
- Load-generator latency is measured from the *intended* send time, open-loop.

## Working agreements

- Python ≥3.10, `ruff` (line length 100), `pytest -q`. Run both before declaring work done.
- Venv at `.venv/`. Install with `pip install -e '.[dev]'`.
- Tests assert behaviour and invariants, not implementation detail. A scheduler test that pins the
  exact call sequence is worse than none.
- Benchmarks and ablations write raw results to disk; plots are regenerated from them, never
  hand-edited.
- Keep GPU-dependent code importable without a GPU, so tests and CI run on CPU.
- Commit only when asked.
