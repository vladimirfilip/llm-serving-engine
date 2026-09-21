"""GPU work that must not hold a CUDA context in the harness process: an engine cannot launch
while `nvidia-smi` shows a compute process. Each task runs in a child that exits before the
next engine starts. `python -m bench.gpu_tasks <task> '<json args>'` prints one JSON line."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path


def run_task(task: str, **args) -> dict:
    """Runs `task` in a subprocess and returns its JSON result."""
    result = subprocess.run([sys.executable, "-m", "bench.gpu_tasks", task, json.dumps(args)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gpu task {task} failed:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    from . import env

    task, args = sys.argv[1], json.loads(sys.argv[2])
    if task == "verify-clock":
        result = dataclasses.asdict(env.verify_clock_lock(args["hw"]))
    elif task == "bwprobe":
        result = env.bandwidth_probe()
    elif task in ("reference-generate", "reference-score", "reference-ppl"):
        from . import reference

        if "prompts_path" in args:  # too long for a command line: read the selected prompts here
            wanted = set(args.pop("ids"))
            args["prompts"] = [p for p in reference.read_jsonl(Path(args.pop("prompts_path")))
                               if p["id"] in wanted]
        result = {"reference-generate": reference.task_generate,
                  "reference-score": reference.task_score,
                  "reference-ppl": reference.task_ppl}[task](**args)
    elif task == "kernels":
        from .suites.kernels import task_kernels

        result = task_kernels(**args)
    else:
        raise SystemExit(f"unknown gpu task {task}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
