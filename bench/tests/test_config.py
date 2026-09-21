from bench.config import Config, apply_quick, expand, load_config


def test_quick_halves_the_sweep_grid_and_shortens_windows():
    full, quick = load_config(), load_config(quick=True)
    for name, w in full.suite["load_sweep"]["per_workload"].items():
        q = quick.suite["load_sweep"]["per_workload"][name]
        assert q["fractions"] == w["fractions"][::2]
        assert q["repeats"] == 1
    assert quick.suite["load_sweep"]["target_duration_s"] == 30
    assert quick.suite["memory"]["grid_batches"] == [1, 8, 64]
    assert quick.suite["correctness"]["gsm8k_limit"] == 100
    assert full.suite["correctness"]["gsm8k_limit"] is None


def test_quick_leaves_the_full_config_untouched():
    suite = load_config().suite
    apply_quick(suite)
    assert suite["load_sweep"]["target_duration_s"] == 120


def test_a_workload_disables_itself_when_the_model_length_is_too_short():
    cfg = load_config()
    assert "ctx32k" not in cfg.workloads
    long_model = Config(cfg.hardware, cfg.model | {"max_model_len": 32768}, cfg.suite)
    assert "ctx32k" in long_model.workloads


def test_workload_index_is_stable_across_disabled_workloads():
    cfg = load_config()
    assert cfg.workload_index("ctx32k") == 4


def test_expand_fills_fields_and_the_home_directory(monkeypatch):
    monkeypatch.setenv("HOME", "/home/x")
    out = expand(["{python}", "--port={port}", "~/envs/{name}/bin/py", "--cfg=~/opts.yaml"],
                 {"python": "/usr/bin/python3", "port": 8000, "name": "vllm"})
    assert out == ["/usr/bin/python3", "--port=8000", "/home/x/envs/vllm/bin/py",
                   "--cfg=/home/x/opts.yaml"]


def test_required_hardware_fields_are_named_when_unset():
    from bench.cli import missing_hardware_fields

    hw = load_config().hardware
    assert missing_hardware_fields(hw) == []
    assert missing_hardware_fields(hw | {"peak_mem_bw_gbs": None}) == ["peak_mem_bw_gbs"]


def test_shipped_configs_leave_no_required_value_unset():
    from bench.cli import missing_hardware_fields

    cfg = load_config()
    assert missing_hardware_fields(cfg.hardware) == []
    assert all(cfg.model[k] is not None for k in
               ("local_path", "dtype", "served_model_name", "max_model_len", "max_num_seqs",
                "gpu_mem_util"))


def test_the_run_snapshots_the_config_directory_it_was_loaded_from(tmp_path):
    import shutil

    from bench.config import CONFIG_DIR
    from bench.run import Run

    shutil.copytree(CONFIG_DIR, tmp_path / "cfg")
    (tmp_path / "cfg" / "suite.yaml").write_text("seed: 7\n")
    cfg = load_config(tmp_path / "cfg")
    run = Run.open(cfg, "r1", results_dir=tmp_path / "results")
    assert (run.dir / "config_snapshot" / "suite.yaml").read_text() == "seed: 7\n"


def test_baseline_versions_skip_files_that_are_not_engines_and_ask_each_environment(tmp_path):
    import shutil
    import sys

    import yaml

    from bench.config import CONFIG_DIR
    from bench.env import baseline_versions

    shutil.copytree(CONFIG_DIR, tmp_path / "cfg")
    engine = yaml.safe_load((tmp_path / "cfg" / "engines" / "mock.yaml").read_text())
    engine |= {"name": "fake", "python": sys.executable, "package": "pyyaml"}
    (tmp_path / "cfg" / "engines" / "fake.yaml").write_text(yaml.safe_dump(engine))
    versions = baseline_versions(load_config(tmp_path / "cfg"))
    assert versions["fake"] == yaml.__version__
    assert "ours_ablation" not in versions  # a steps file is not an engine
