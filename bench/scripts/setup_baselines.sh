#!/usr/bin/env bash
# Creates one environment per baseline under ~/.bench-envs/<name> and installs the versions pinned
# in bench/baselines.lock. The baselines pin their own torch and CUDA builds, so they cannot share
# an environment with the engine or with each other. The harness only ever launches them as
# subprocesses with these interpreters.
#
# Beyond the pinned packages, SGLang and TensorRT-LLM JIT-compile or link against a CUDA toolkit,
# and pip ships one inside the environment (site-packages/nvidia/cu13). It needs lib64 and an
# unversioned libcudart.so link, and TensorRT-LLM additionally needs the CUDA 13 builds of torch
# and torchvision, a matching nvcc, and the system MPI library. The engine YAMLs point CUDA_HOME
# and PATH at these trees.
set -euo pipefail

ENVS="${BENCH_ENVS:-$HOME/.bench-envs}"
LOCK="$(cd "$(dirname "$0")/.." && pwd)/baselines.lock"
PYTHON_VERSION="${BENCH_BASELINE_PYTHON:-3.12}"
TORCH_CU130="https://download.pytorch.org/whl/cu130"

command -v uv >/dev/null || python3 -m pip install --user --quiet uv
export PATH="$HOME/.local/bin:$PATH"

pinned() { grep -E "^$1==" "$LOCK" | head -1; }
pip_in() { uv pip install --python "$ENVS/$1/bin/python" "${@:2}"; }

link_cuda_tree() {  # name: make the pip CUDA tree linkable
  local tree="$ENVS/$1/lib/python$PYTHON_VERSION/site-packages/nvidia/cu13"
  [ -e "$tree/lib64" ] || ln -s lib "$tree/lib64"
  [ -e "$tree/lib/libcudart.so" ] \
    || ln -s "$(ls "$tree/lib" | grep -E '^libcudart\.so\.[0-9]+$' | head -1)" "$tree/lib/libcudart.so"
}

uv venv --python "$PYTHON_VERSION" --seed "$ENVS/vllm"
pip_in vllm "$(pinned vllm)"

uv venv --python "$PYTHON_VERSION" --seed "$ENVS/sglang"
pip_in sglang "$(pinned sglang)"
link_cuda_tree sglang

ldconfig -p | grep -q 'libmpi.so.40' || { echo "TensorRT-LLM needs libopenmpi3: apt install libopenmpi3" >&2; exit 1; }
uv venv --python "$PYTHON_VERSION" --seed "$ENVS/trtllm"
pip_in trtllm "$(pinned tensorrt_llm)" --extra-index-url https://pypi.nvidia.com
pip_in trtllm "torch==2.9.1+cu130" "torchvision==0.24.1+cu130" --index-url "$TORCH_CU130" \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
pip_in trtllm "nvidia-cuda-nvcc==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-crt==13.0.*" \
  "nvidia-cuda-cccl==13.0.*"
link_cuda_tree trtllm
