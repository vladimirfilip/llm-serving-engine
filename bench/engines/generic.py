"""The adapter for every engine. An engine YAML says how to launch it, which body fields it
needs, and where its stats and prompt-scoring endpoints are."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import yaml

from .. import env as bench_env
from ..config import BENCH_DIR, Config, EngineSpec, expand
from ..run import Run
from .base import CheckFailed, EngineAdapter, Launch, request_body
from .server_proc import ServerProcess, free_port

HTTP_TIMEOUT_S = 60
POLL_S = 0.5
SERIES = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)$")


def http_json(url: str, body: dict | None = None, timeout: float = HTTP_TIMEOUT_S) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def render(value, values: dict):
    """Fills `{name}` fields through nested YAML values. A string that is exactly one field
    keeps the field's own type, so a number stays a number."""
    if isinstance(value, str):
        whole = re.fullmatch(r"\{(\w+)\}", value)
        return values[whole.group(1)] if whole else expand(value, values)
    if isinstance(value, dict):
        return {k: render(v, values) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, values) for v in value]
    return value


def parse_prometheus(text: str) -> dict[str, float]:
    """Series values by metric name, summed over label sets."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if match := SERIES.match(line):
            totals[match.group(1)] = totals.get(match.group(1), 0.0) + float(match.group(2))
    return totals


class GenericAdapter(EngineAdapter):
    def __init__(self, spec: EngineSpec, cfg: Config, run: Run):
        self.spec, self.cfg, self.run = spec, cfg, run
        self.name = spec.name
        self.accepts_token_ids = spec.accepts_token_ids
        self.extra_body = spec.extra_body
        self.request_model = expand(spec.request_model, self._values(Launch(), 0))
        self.proc: ServerProcess | None = None
        self.port = 0
        self.launch_argv: list[str] = []

    def _values(self, launch: Launch, port: int) -> dict:
        model = self.cfg.model
        return {
            "python": expand(self.spec.python, {}) if self.spec.python else sys.executable,
            "model_path": self.cfg.model_path, "served_model_name": model["served_model_name"],
            "port": port, "max_model_len": model["max_model_len"],
            "max_num_seqs": model["max_num_seqs"], "gpu_mem_util": model["gpu_mem_util"],
            "dtype": model["dtype"],
            "token_budget": launch.token_budget, "run_dir": str(self.run.dir),
            "bench_dir": str(BENCH_DIR),
        }

    def launch(self, launch: Launch, phase: str, env_overrides: dict | None = None) -> None:
        if self.spec.uses_gpu and bench_env.gpu_available() and bench_env.compute_pids():
            raise CheckFailed("the GPU has a compute process running before launch")
        self.port = free_port()
        values = self._values(launch, self.port)
        for name, content in self.spec.files.items():
            (self.run.dir / name).write_text(yaml.safe_dump(render(content, values)))
        self.launch_argv = launch.wrapper + expand(self.spec.launch, values) + launch.args_add
        env = {k: str(rendered) for k, v in
               (self.spec.env | launch.env | (env_overrides or {})).items()
               if (rendered := render(v, values)) is not None}
        if self.spec.path_prepend:
            env["PATH"] = ":".join(
                [*expand(self.spec.path_prepend, values), os.environ.get("PATH", "")])
        self.proc = ServerProcess(self.launch_argv, env, self.run.log_path(self.name, phase),
                                  self.cfg.hardware["cpu_affinity"]["server"],
                                  wait_gpu_free=self.spec.uses_gpu)
        self.proc.start()

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def wait_ready(self, timeout_s: float = 900) -> float:
        deadline = time.perf_counter() + timeout_s
        health = self.base_url() + self.spec.health_path
        while True:
            if self.proc.exited():
                raise RuntimeError(f"{self.name} exited during startup:\n{self.proc.log_tail()}")
            if time.perf_counter() > deadline:
                raise TimeoutError(f"{self.name} not ready after {timeout_s} s")
            try:
                with urllib.request.urlopen(health, timeout=5) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(POLL_S)
        self.complete([1] + [2] * 7, 8)
        return time.perf_counter() - self.proc.started_at

    def complete(self, prompt_ids: list[int], max_tokens: int) -> dict:
        prompt = prompt_ids if self.accepts_token_ids else " ".join(map(str, prompt_ids[1:]))
        body = request_body(self.request_model, prompt, max_tokens, self.extra_body, stream=False)
        return http_json(self.base_url() + "/v1/completions", body, timeout=timeout_for(max_tokens))

    def complete_text(self, text: str, max_tokens: int) -> dict:
        body = request_body(self.request_model, text, max_tokens, self.extra_body, stream=False)
        return http_json(self.base_url() + "/v1/completions", body, timeout=timeout_for(max_tokens))

    def stats(self) -> dict | None:
        try:
            if self.spec.stats_api == "internal":
                return http_json(self.base_url() + "/internal/stats", timeout=5)
            if self.spec.stats_api == "prometheus" and self.spec.metrics:
                return self._prometheus_stats()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return None
        return None

    def _prometheus_stats(self) -> dict:
        with urllib.request.urlopen(self.base_url() + self.spec.metrics["path"], timeout=5) as r:
            series = parse_prometheus(r.read().decode())

        def first(key: str) -> float | None:
            return next((series[n] for n in self.spec.metrics.get(key, []) if n in series), None)

        usage = first("kv_usage")
        return {"running": first("running"), "waiting": first("waiting"),
                "preemptions_total": first("preemptions"), "memory_bytes": None,
                "kv": {"block_size": None, "blocks_total": None, "blocks_used": None,
                       "tokens_used": None, "usage": usage}}

    def score(self, token_ids: list[int]) -> list[float] | None:
        api, url = self.spec.score_api, self.base_url()
        if api == "internal":
            return http_json(url + "/internal/score", {"token_ids": token_ids})["logprobs"]
        if api == "vllm":
            body = request_body(self.request_model, token_ids, 1, {}, stream=False)
            body["prompt_logprobs"] = 0
            entries = http_json(url + "/v1/completions", body, 600)["choices"][0]["prompt_logprobs"]
            return [entry[str(t)]["logprob"]
                    for entry, t in zip(entries[1:], token_ids[1:], strict=True)]
        if api == "sglang":
            body = {"input_ids": token_ids,
                    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                    "return_logprob": True, "logprob_start_len": 0}
            out = http_json(url + "/generate", body, 600)
            return [entry[0] for entry in out["meta_info"]["input_token_logprobs"][1:]]
        return None

    def shutdown(self) -> None:
        if self.proc is not None:
            self.proc.stop()
            self.proc = None


def timeout_for(max_tokens: int) -> float:
    return HTTP_TIMEOUT_S + max_tokens * 0.5


def make_adapter(spec: EngineSpec, cfg: Config, run: Run) -> GenericAdapter:
    return GenericAdapter(spec, cfg, run)
