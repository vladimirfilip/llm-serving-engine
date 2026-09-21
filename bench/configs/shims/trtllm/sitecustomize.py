"""Loaded by the trtllm server and its workers through PYTHONPATH. The pynvml binding passes the
confidential-compute settings struct by value where NVML expects a pointer, so on a driver that
exposes the query TensorRT-LLM's startup probe segfaults. Reporting the query as unsupported
sends TensorRT-LLM down its own fallback path. No baseline source is modified."""

try:
    import pynvml

    def _unsupported(*_args, **_kwargs):
        raise pynvml.NVMLError_NotSupported

    pynvml.nvmlSystemGetConfComputeSettings = _unsupported
except ImportError:
    pass
