import numpy as np
import pandas as pd


def record(req_id: int, t_sched: float, ttft: float = 0.1, tpot: float = 0.02, n_tokens: int = 5,
           status: str = "ok", lag: float = 0.0, prompt_len: int = 100) -> dict:
    """One client record: tokens arrive `ttft` after the send, then every `tpot`."""
    t_send = t_sched + lag
    times = [t_send + ttft + tpot * i for i in range(n_tokens)]
    ok = status == "ok"
    return {
        "run_id": "r", "engine": "e", "phase": "sweep", "workload": "w", "rate_rps": 1.0,
        "repeat": 0, "req_id": req_id, "t_sched": t_sched, "t_send": t_send,
        "t_first": times[0] if ok else np.nan, "t_done": times[-1] if ok else np.nan,
        "prompt_len": prompt_len, "output_len_req": n_tokens,
        "prompt_tokens_usage": prompt_len, "completion_tokens_usage": n_tokens if ok else 0,
        "n_chunks": n_tokens if ok else 0, "token_times": times if ok else [],
        "status": status, "http_status": 200 if ok else 500, "error": "" if ok else "boom",
    }


def frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)
