import json

import pytest

from bench.publish import (
    END,
    START,
    MissingMarkers,
    publish,
    replace_block,
    unmet_conditions,
)
from bench.tests import fixture_run

README = f"# engine\n\nintro\n\n{START}\nOLD RESULTS\n{END}\n\nfooter\n"


@pytest.fixture(scope="module")
def good_run(tmp_path_factory):
    return fixture_run.write(tmp_path_factory.mktemp("results") / "20260921-120000-abc1234",
                             fixture_run.ALL_SUITES)


def write(tmp_path, name, **kwargs):
    return fixture_run.write(tmp_path / name, fixture_run.ALL_SUITES, **kwargs)


def test_a_clean_run_has_no_unmet_conditions(good_run):
    assert unmet_conditions(good_run) == []


@pytest.mark.parametrize(("change", "expected"), [
    ({"quick": True}, "--quick"),
    ({"locked": False}, "clocks were not locked"),
    ({"dirty": True}, "dirty"),
    ({"invalid": True}, "invalid or throttled"),
    ({"gates_ok": False}, "gate failed"),
])
def test_each_precondition_blocks_a_publish_on_its_own(tmp_path, change, expected):
    run = write(tmp_path, "r", **change)
    unmet = unmet_conditions(run)
    assert len(unmet) == 1 and expected in unmet[0]


def test_unevaluated_gates_and_unfinished_phases_block_a_publish(tmp_path):
    run = write(tmp_path, "r")
    summary = json.loads((run / "correctness" / "summary.json").read_text())
    summary["gates"]["ppl_vs_reference"]["passed"] = None
    (run / "correctness" / "summary.json").write_text(json.dumps(summary))
    (run / "ablation" / "DONE").unlink()
    unmet = unmet_conditions(run)
    assert any("not all four correctness gates" in u for u in unmet)
    assert any("ablation" in u for u in unmet)


def test_a_refused_publish_writes_nothing(tmp_path):
    run = write(tmp_path, "r", quick=True)
    published, readme = tmp_path / "published", tmp_path / "README.md"
    readme.write_text(README)
    assert publish(run, published=published, readme=readme, write_readme=True)
    assert not published.exists() and readme.read_text() == README


def test_a_publish_writes_only_under_published_unless_the_readme_is_asked_for(good_run, tmp_path):
    published, readme = tmp_path / "published", tmp_path / "README.md"
    readme.write_text(README)
    assert publish(good_run, published=published, readme=readme) == []
    assert readme.read_text() == README
    assert {p.name for p in published.iterdir()} == {
        "plots", "tables", "report.md", "run.json", "commands.json", "env.json",
        "readme_section.md"}
    assert all(p.suffix == ".png" for p in (published / "plots").iterdir())
    assert not list((published / "tables").glob("*.parquet"))


def test_no_published_file_contains_a_home_path_username_or_hostname(good_run, tmp_path):
    published = tmp_path / "published"
    publish(good_run, published=published, readme=tmp_path / "README.md")
    for path in published.rglob("*"):
        if path.is_file() and path.suffix != ".png":
            text = path.read_text()
            leaked = ("/home/" in text or "alice" in text or fixture_run.HOST in text)
            assert not leaked, path
    env = json.loads((published / "env.json").read_text())
    assert env["hostname"] == "<host>" and env["model"]["path"] == "Llama-3.2-3B-Instruct"
    launch = json.loads((published / "commands.json").read_text())["ours"]["launch"]
    assert launch[0] == "~/.bench-envs/ours/bin/python"
    assert launch[-1] == "--model=Llama-3.2-3B-Instruct"


def test_writing_the_readme_twice_gives_identical_bytes_and_keeps_the_rest(good_run, tmp_path):
    published, readme = tmp_path / "published", tmp_path / "README.md"
    readme.write_text(README)
    publish(good_run, write_readme=True, published=published, readme=readme)
    first = readme.read_bytes()
    publish(good_run, write_readme=True, published=published, readme=readme)
    assert readme.read_bytes() == first
    text = readme.read_text()
    assert text.startswith("# engine\n\nintro\n\n" + START) and text.endswith(END + "\n\nfooter\n")
    assert "OLD RESULTS" not in text and "## Benchmarks" in text
    assert "bench/published/plots/p01_throughput_latency_sharegpt.png" in text


def test_missing_markers_give_an_error_and_no_edit(good_run, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# no markers here\n")
    with pytest.raises(MissingMarkers):
        publish(good_run, write_readme=True, published=tmp_path / "published", readme=readme)
    assert readme.read_text() == "# no markers here\n"
    with pytest.raises(MissingMarkers):
        replace_block(f"{END} before {START}", "x")


def test_a_forced_publish_states_the_unmet_conditions_in_the_readme_block(tmp_path):
    run = write(tmp_path, "r", quick=True)
    published = tmp_path / "published"
    assert publish(run, force=True, published=published, readme=tmp_path / "README.md")
    assert "Published with --force. Unmet conditions: the run was --quick." in (
        published / "readme_section.md").read_text()
