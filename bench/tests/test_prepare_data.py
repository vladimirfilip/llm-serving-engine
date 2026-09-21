"""`prepare_data` end to end with the network and tokenizer faked."""

import json
import sys
import types

import numpy as np
import pytest

from bench.config import load_config
from bench.workloads.prepare import prepare_data


class FakeTokenizer:
    bos_token_id = 9

    @classmethod
    def from_pretrained(cls, _path):
        return cls()

    def __call__(self, texts, add_special_tokens=True):
        text_list = [texts] if isinstance(texts, str) else texts
        ids = [[self.bos_token_id] * add_special_tokens + [ord(c) for c in t] for t in text_list]
        return types.SimpleNamespace(input_ids=ids[0] if isinstance(texts, str) else ids)


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    conversations = [
        {"id": f"c{i}", "conversations": [{"from": "human", "value": "q" * (10 + i % 40)},
                                          {"from": "gpt", "value": "a" * (10 + i % 30)}]}
        for i in range(300)
    ]
    sharegpt = tmp_path / "sharegpt.json"
    sharegpt.write_text(json.dumps(conversations))
    wikitext = {"train": {"text": ["word " * 400] * 200}, "test": {"text": ["word " * 50] * 20}}
    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda *_a, **_k: wikitext))
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=lambda *_a, **_k: str(sharegpt)))
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(AutoTokenizer=FakeTokenizer))


def test_prepare_data_writes_every_dataset_file_and_the_bos_id(tmp_path, fake_hub):
    cfg = load_config()
    suite = cfg.suite | {
        "workloads": cfg.suite["workloads"] | {"sharegpt": cfg.suite["workloads"]["sharegpt"]
                                                | {"n_pool": 100}},
        "correctness": cfg.suite["correctness"] | {"n_prompts": 20, "n_long_prompts": 2,
                                                   "long_prompt_len": 100,
                                                   "prompt_len_range": [4, 1024]},
    }
    prepare_data(type(cfg)(cfg.hardware, cfg.model, suite), tmp_path)

    assert json.loads((tmp_path / "meta.json").read_text()) == {"bos_token_id": 9,
                                                                 "synthetic": False}
    pool_lines = (tmp_path / "sharegpt_pool.jsonl").read_text().splitlines()
    pool = [json.loads(line) for line in pool_lines]
    assert len(pool) == 100 and all(r["prompt_token_ids"][0] == 9 for r in pool)
    train = np.load(tmp_path / "wikitext2_train_ids.npy")
    assert train.dtype == np.int32 and 9 not in train.tolist()  # no BOS in the raw WikiText ids
    test = np.load(tmp_path / "wikitext2_test_ids.npy")
    assert test.dtype == np.int32 and 0 < len(test) < len(train)
    prompts = [json.loads(line) for line in
               (tmp_path / "correctness_prompts.jsonl").read_text().splitlines()]
    assert len(prompts) == 22 and [p["kind"] for p in prompts].count("long") == 2
