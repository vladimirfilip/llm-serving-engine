#!/usr/bin/env bash
# Creates one environment per baseline under ~/.bench-envs/<name> and installs the versions pinned
# in bench/baselines.lock. The baselines pin their own torch and CUDA builds, so they cannot share
# an environment with the engine or with each other. The harness only ever launches them as
# subprocesses with these interpreters.
set -euo pipefail

ENVS="${BENCH_ENVS:-$HOME/.bench-envs}"
LOCK="$(cd "$(dirname "$0")/.." && pwd)/baselines.lock"
PYTHON_VERSION="${BENCH_BASELINE_PYTHON:-3.12}"

command -v uv >/dev/null || python3 -m pip install --user --quiet uv
export PATH="$HOME/.local/bin:$PATH"

pinned() { grep -E "^$1==" "$LOCK" | head -1; }

install() {  # name, pip requirement, extra index (optional)
  local name="$1" requirement="$2" index="${3:-}"
  uv venv --python "$PYTHON_VERSION" --seed "$ENVS/$name"
  if [ -n "$index" ]; then
    uv pip install --python "$ENVS/$name/bin/python" "$requirement" --extra-index-url "$index"
  else
    uv pip install --python "$ENVS/$name/bin/python" "$requirement"
  fi
}

install vllm "$(pinned vllm)"
install sglang "$(pinned sglang)"
install trtllm "$(pinned tensorrt_llm)" https://pypi.nvidia.com
