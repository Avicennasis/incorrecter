import ast
import json
import py_compile
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from incorrecter import SYSTEM_PROMPT
from incorrecter.jsonl import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parent.parent


def test_modelfile_system_prompt_matches_package():
    match = re.search(r'^SYSTEM "(.*)"$', (ROOT / "Modelfile").read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None
    assert match.group(1) == SYSTEM_PROMPT


def test_train_unsloth_compiles(tmp_path):
    py_compile.compile(str(ROOT / "train" / "train_unsloth.py"), cfile=str(tmp_path / "t.pyc"), doraise=True)


def test_train_mlx_is_valid_bash():
    subprocess.run(["bash", "-n", str(ROOT / "train" / "train_mlx.sh")], check=True)


def test_train_mlx_stops_before_training_when_data_is_missing(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "train" / "train_mlx.sh")],
        env={"DATA": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "missing" in result.stderr


def test_train_unsloth_evaluates_on_the_valid_split():
    # The script hands data/valid.jsonl to SFTTrainer; without an eval strategy TRL never evaluates on it.
    tree = ast.parse((ROOT / "train" / "train_unsloth.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "SFTConfig"]
    assert len(calls) == 1
    kwargs = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kwargs.get("eval_strategy"), ast.Constant)
    assert kwargs["eval_strategy"].value in {"steps", "epoch"}
    if kwargs["eval_strategy"].value == "steps":
        assert "eval_steps" in kwargs


def test_train_mlx_fuses_to_full_precision():
    # Fusing into the 4-bit base re-quantizes the merged weights, and the LoRA delta is smaller than one 4-bit
    # step: measured on macbook4 2026-09-27, the 4-bit fused model behaved like the untuned base.
    script = (ROOT / "train" / "train_mlx.sh").read_text(encoding="utf-8")
    fuse_call = script[script.index("mlx_lm.fuse") :].split("\n\n")[0]
    assert "--dequantize" in fuse_call


def test_train_mlx_sets_learning_rate_and_max_seq_length_and_checks_lengths_first():
    script = (ROOT / "train" / "train_mlx.sh").read_text(encoding="utf-8")
    lora_call = script[script.index("mlx_lm.lora") :].split("\n\n")[0]
    assert '--learning-rate "$LEARNING_RATE"' in lora_call
    assert '--max-seq-length "$MAX_SEQ_LENGTH"' in lora_call
    assert 'LEARNING_RATE="${LEARNING_RATE:-1e-4}"' in script and 'ITERS="${ITERS:-2000}"' in script
    assert script.index("check_lengths.py") < script.index("mlx_lm.lora")


def test_mac_side_scripts_compile(tmp_path):
    for name in ("evaluate_mlx.py", "check_lengths.py"):
        py_compile.compile(str(ROOT / "train" / name), cfile=str(tmp_path / f"{name}.pyc"), doraise=True)


def _load_train_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), ROOT / "train" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_check_lengths():
    return _load_train_script("check_lengths.py")


def test_check_lengths_drops_long_rows_and_reports_ids(tmp_path):
    import json

    check_lengths = _load_check_lengths()
    rows = [
        {"id": f"r{i}", "messages": [{"role": "user", "content": "x" * (5000 if i == 3 else 10)}]} for i in range(200)
    ]
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    code = check_lengths.check(tmp_path, lambda m: len(m[0]["content"]), 2048, tmp_path / "over_length.txt")
    assert code == 0
    assert (tmp_path / "over_length.txt").read_text() == "train.jsonl r3\nvalid.jsonl r3\n"
    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 199


def test_check_lengths_aborts_above_one_percent(tmp_path):
    import json

    check_lengths = _load_check_lengths()
    rows = [{"messages": [{"role": "user", "content": "x" * (5000 if i < 5 else 10)}]} for i in range(100)]
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert check_lengths.check(tmp_path, lambda m: len(m[0]["content"]), 2048, tmp_path / "o.txt") == 1


class _DictByDefaultTokenizer:
    """transformers 5.x: apply_chat_template(tokenize=True) returns a BatchEncoding (len 2) unless return_dict=False."""

    def apply_chat_template(self, messages, tokenize=False, return_dict=True):
        ids = [0] * sum(len(m["content"]) for m in messages)
        return {"input_ids": ids, "attention_mask": ids} if return_dict else ids


def test_token_counter_counts_tokens_not_batch_encoding_keys():
    check_lengths = _load_check_lengths()
    count = check_lengths.token_counter(_DictByDefaultTokenizer())
    assert count([{"role": "user", "content": "x" * 3000}]) == 3000


def _content_length(messages):
    return len(messages[0]["content"])


def _write_splits(tmp_path, rows):
    import json

    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_check_lengths_abort_leaves_the_files_unchanged(tmp_path):
    check_lengths = _load_check_lengths()
    rows = [
        {"id": f"r{i}", "messages": [{"role": "user", "content": "x" * (5000 if i < 5 else 10)}]} for i in range(100)
    ]
    _write_splits(tmp_path, rows)
    before = {p.name: p.read_bytes() for p in sorted(tmp_path.glob("*.jsonl"))}
    count = _content_length
    assert check_lengths.check(tmp_path, count, 2048, tmp_path / "o.txt") == 1
    assert {p.name: p.read_bytes() for p in sorted(tmp_path.glob("*.jsonl"))} == before
    assert check_lengths.check(tmp_path, count, 2048, tmp_path / "o.txt") == 1  # a re-run cannot slip past the abort


def test_check_lengths_rerun_gives_the_same_report_and_rows(tmp_path):
    check_lengths = _load_check_lengths()
    rows = [
        {"id": f"r{i}", "messages": [{"role": "user", "content": "x" * (5000 if i == 3 else 10)}]} for i in range(200)
    ]
    _write_splits(tmp_path, rows)
    count = _content_length
    for _ in range(2):
        assert check_lengths.check(tmp_path, count, 2048, tmp_path / "over_length.txt") == 0
        assert (tmp_path / "over_length.txt").read_text() == "train.jsonl r3\nvalid.jsonl r3\n"
        assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 199


def _load_model_too_early(path):
    raise AssertionError("evaluate_mlx loaded the model before checking the held-out files")


def test_evaluate_mlx_checks_every_heldout_file_before_loading_the_model(tmp_path, monkeypatch, capsys):
    # mlx is not installed on dev; this stand-in fails the test if the model is loaded at all.
    fake = types.ModuleType("mlx_lm")
    fake.load = _load_model_too_early
    fake.generate = None
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    present = tmp_path / "heldout.jsonl"
    write_jsonl(present, [{"id": "work-0001", "writer": "glm", "license": "Z.ai terms", "text": "Hello there."}])
    missing = tmp_path / "private" / "heldout_human.jsonl"
    argv = ["--model", "m", "--label", "t", "--heldout", str(present), str(missing), "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit) as exit_info:
        _load_train_script("evaluate_mlx.py").main(argv)
    assert exit_info.value.code != 0
    err = capsys.readouterr().err
    assert str(missing) in err and str(present) not in err


class _LastTurnTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        return messages[-1]["content"]


def _load_echo_model(path):
    return None, _LastTurnTokenizer()


def _echo(model, tokenizer, prompt, max_tokens, verbose, sampler=None):
    return prompt


def test_evaluate_mlx_keeps_no_text_from_a_private_file_whatever_its_license(tmp_path, monkeypatch):
    # A file under data/private/ holds human text (spec S13), even if a row's license would not say so.
    fake = types.ModuleType("mlx_lm")
    fake.load, fake.generate = _load_echo_model, _echo
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    row = {"writer": "v1", "license": "MIT", "text": "Their report is attached."}
    public, private = tmp_path / "data" / "heldout.jsonl", tmp_path / "data" / "private" / "heldout_human.jsonl"
    write_jsonl(public, [row | {"id": "public-1"}])
    write_jsonl(private, [row | {"id": "private-1"}])
    # data/private/ as a symlink to storage elsewhere: resolving the path would lose the data/private part.
    write_jsonl(tmp_path / "store" / "heldout_human.jsonl", [row | {"id": "linked-1"}])
    (tmp_path / "linked" / "data").mkdir(parents=True)
    (tmp_path / "linked" / "data" / "private").symlink_to(tmp_path / "store", target_is_directory=True)
    linked = tmp_path / "linked" / "data" / "private" / "heldout_human.jsonl"
    out = tmp_path / "eval"
    _load_train_script("evaluate_mlx.py").main(
        ["--model", "m", "--label", "t", "--heldout", str(public), str(private), str(linked), "--out-dir", str(out)]
    )
    records = {r["id"]: r for r in read_jsonl(out / "t.jsonl")}
    assert records["public-1"]["input"] == row["text"] and not records["public-1"]["human"]
    for row_id in ("private-1", "linked-1"):
        assert "input" not in records[row_id] and "output" not in records[row_id]
        assert records[row_id]["human"]


def test_ci_ruff_job_checks_train():
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    (command,) = [line for line in workflow.splitlines() if "ruff check" in line]
    assert {"incorrecter", "tests", "train"} <= set(command.split())


def test_ci_ruff_is_pinned_to_the_pre_commit_version():
    # An unpinned ruff changed its default rules in 0.16 (I001 among them); CI must lint with the version
    # pre-commit uses, so a ruff release cannot break CI with no code change.
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    pin = re.search(r"pip install ruff==(\S+)", workflow)
    rev = re.search(r"astral-sh/ruff-pre-commit\s+rev: v(\S+)", config)
    assert rev is not None
    assert pin is not None and pin.group(1) == rev.group(1)


@pytest.fixture
def stub_mlx_lm(monkeypatch):
    """One reusable stand-in for mlx_lm + mlx.core: records the sampler make_sampler built, the
    mx.random.seed value, and the sampler= kwarg every generate call received."""
    seen = {"sampler": None, "min_p": None, "mx_seed": None, "gen": []}
    fake_mx_core = types.ModuleType("mlx.core")
    fake_mx_core.random = types.SimpleNamespace(seed=lambda s: seen.update(mx_seed=s))
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mx_core

    def make_sampler(temp, top_p=0.0, min_p=0.0):
        seen["sampler"] = (temp, top_p)
        seen["min_p"] = min_p
        return f"sampler({temp},{top_p})"

    fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")
    fake_sample_utils.make_sampler = make_sampler

    def generate(model, tokenizer, prompt, max_tokens, verbose, sampler=None):
        seen["gen"].append(sampler)
        return _echo(model, tokenizer, prompt, max_tokens, verbose)

    fake_lm = types.ModuleType("mlx_lm")
    fake_lm.load, fake_lm.generate = _load_echo_model, generate
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_lm)
    return seen


def _heldout_row(row_id="d1"):
    return {
        "id": row_id,
        "text": "Hello there team.",
        "category": "x",
        "length": "short",
        "writer": "glm",
        "source": "drafted",
        "license": "Z.ai terms",
        "requested_length": "short",
    }


def test_evaluate_mlx_sampling_flags_reach_the_sampler(tmp_path, stub_mlx_lm):
    # --temp/--top-p/--seed must reach make_sampler and mx.random.seed (the Stage 1 sweep's instrument).
    f = tmp_path / "heldout.jsonl"
    write_jsonl(f, [_heldout_row()])
    argv = [
        "--model",
        "m",
        "--label",
        "t",
        "--heldout",
        str(f),
        "--out-dir",
        str(tmp_path / "eval"),
        "--temp",
        "0.6",
        "--top-p",
        "0.9",
        "--seed",
        "7",
    ]
    _load_train_script("evaluate_mlx.py").main(argv)
    assert stub_mlx_lm["sampler"] == (0.6, 0.9)
    assert stub_mlx_lm["mx_seed"] == 7
    assert stub_mlx_lm["gen"] and all(s is not None for s in stub_mlx_lm["gen"])


def test_evaluate_mlx_defaults_stay_greedy(tmp_path, stub_mlx_lm):
    # The recorded Phase 2 rows are greedy: with no flags, no sampler is built and no seed is set.
    # Two files: the sweep passes both (nargs="+").
    f1 = tmp_path / "heldout.jsonl"
    f2 = tmp_path / "heldout2.jsonl"
    write_jsonl(f1, [_heldout_row()])
    write_jsonl(f2, [_heldout_row("d2")])
    argv = ["--model", "m", "--label", "t", "--heldout", str(f1), str(f2), "--out-dir", str(tmp_path / "eval")]
    _load_train_script("evaluate_mlx.py").main(argv)
    assert stub_mlx_lm["sampler"] is None and stub_mlx_lm["mx_seed"] is None
    assert stub_mlx_lm["gen"] and all(s is None for s in stub_mlx_lm["gen"])


def test_train_mlx_seed_reaches_lora():
    script = (ROOT / "train" / "train_mlx.sh").read_text(encoding="utf-8")
    assert 'SEED="${SEED:-0}"' in script
    lora_call = script[script.index("mlx_lm.lora") :].split("\n\n")[0]
    assert '--seed "$SEED"' in lora_call


def test_train_mlx_num_layers_is_conditional_and_setu_safe():
    # macbook4's /bin/bash is 3.2: expanding an empty array under set -u errors there, so the flag
    # must live inside the EXTRA_LORA conditional and the call must use the ${ARR[@]+...} idiom.
    script = (ROOT / "train" / "train_mlx.sh").read_text(encoding="utf-8")
    assert 'NUM_LAYERS="${NUM_LAYERS:-}"' in script
    assert 'EXTRA_LORA+=(--num-layers "$NUM_LAYERS")' in script
    lora_call = script[script.index("mlx_lm.lora") :].split("\n\n")[0]
    assert '"${EXTRA_LORA[@]+"${EXTRA_LORA[@]}"}"' in lora_call


def test_evaluate_mlx_generation_error_is_recorded_without_text(tmp_path, monkeypatch, capsys):
    # A failed generation must neither kill the run nor leak the prompt: the row becomes an error
    # record (type name only), the summary counts it, and the run exits non-zero (a sweep marks fail).
    calls = {"n": 0}

    def generate(model, tokenizer, prompt, max_tokens, verbose, sampler=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom-secret-prompt-text")
        return _echo(model, tokenizer, prompt, max_tokens, verbose)

    fake_lm = types.ModuleType("mlx_lm")
    fake_lm.load, fake_lm.generate = _load_echo_model, generate
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_lm)
    public = tmp_path / "heldout.jsonl"
    write_jsonl(public, [_heldout_row("d1"), _heldout_row("d2")])
    out = tmp_path / "eval"
    with pytest.raises(SystemExit) as exit_info:
        _load_train_script("evaluate_mlx.py").main(
            ["--model", "m", "--label", "t", "--heldout", str(public), "--out-dir", str(out)]
        )
    assert exit_info.value.code != 0
    records = {r["id"]: r for r in read_jsonl(out / "t.jsonl")}
    assert "changed" in records["d1"]
    assert records["d2"] == {"id": "d2", "error": "RuntimeError", "human": False}
    everything = (out / "t.jsonl").read_text() + capsys.readouterr().out
    assert "boom-secret-prompt-text" not in everything and "d2" in everything


def test_evaluate_mlx_meaning_records_without_text_for_private_rows(tmp_path, monkeypatch, stub_mlx_lm, capsys):
    # --meaning-judge: verdicts land in eval/t.meaning.jsonl (human rows: no text anywhere); a failing
    # judge call is an error record (type name only) and is counted, without failing the run.
    private = tmp_path / "data" / "private" / "heldout_human.jsonl"
    private.parent.mkdir(parents=True)
    write_jsonl(tmp_path / "heldout.jsonl", [_heldout_row("d1")])
    write_jsonl(
        private, [{"id": "h1", "text": "CANARY private words here.", "writer": "human", "license": "public-record"}]
    )

    def fake_verdict(clean, corrupted):
        if "CANARY" in corrupted:
            raise RuntimeError("judge-down CANARY")
        return True

    monkeypatch.setattr("incorrecter.meaning.build_verdict", lambda base, model, **kw: fake_verdict)
    _load_train_script("evaluate_mlx.py").main(
        [
            "--model",
            "m",
            "--label",
            "t",
            "--heldout",
            str(tmp_path / "heldout.jsonl"),
            str(private),
            "--out-dir",
            str(tmp_path / "eval"),
            "--meaning-judge",
            "http://127.0.0.1:18102/v1",
        ]
    )
    out = (tmp_path / "eval" / "t.meaning.jsonl").read_text()
    assert "CANARY" not in out and "judge-down" not in out
    records = {r["id"]: r for r in read_jsonl(tmp_path / "eval" / "t.meaning.jsonl")}
    assert records["d1"]["verdict"] is True and not records["d1"]["human"]
    assert records["h1"]["error"] == "RuntimeError" and records["h1"]["human"]  # type name only
    assert "verdict" not in records["h1"] and "input" not in records["h1"] and "output" not in records["h1"]
    captured = capsys.readouterr().out
    assert '"meaning_errors": 1' in captured


def test_memorize_runner_is_counts_only_on_a_private_path(tmp_path, stub_mlx_lm, capsys):
    # The runner reads human text under data/private/: its stdout is {"rows": N, "hits": M} and nothing
    # else. Lives in tests/test_training_files.py, which owns the mlx stub and the importlib helper.
    seeds = tmp_path / "data" / "private" / "seeds.jsonl"
    seeds.parent.mkdir(parents=True)
    write_jsonl(
        seeds,
        [
            {
                "id": "h1",
                "text": "furnace word " * 30,
                "writer": "human",
                "source": "sample-archive",
                "license": "public-record",
            }
        ],
    )
    _load_train_script("memorize_mlx.py").main(["--model", "m", "--seeds", str(seeds)])
    captured = capsys.readouterr().out
    assert set(json.loads(captured)) <= {"rows", "hits"}
    assert "furnace" not in captured


def test_evaluate_mlx_calibration_reports_each_suite_rate_and_the_verdict(tmp_path, monkeypatch, capsys):
    # "both" must not pool the suites: 3 yes / 1 no in aggregate is a pass or a fail depending on WHICH
    # pairs said yes. Per-pair records carry suite/index/verdict only — never text (private rows).
    write_jsonl(tmp_path / "heldout.jsonl", [_heldout_row("d1")])
    clean = [("a", "a~"), ("CANARY", "CANARY~")]
    mismatched = [("a", "CANARY~"), ("CANARY", "a~")]
    monkeypatch.setattr("incorrecter.meaning.calibration_pairs", lambda texts, rng: (clean, mismatched))
    answers = {("a", "a~"): True, ("CANARY", "CANARY~"): True, ("a", "CANARY~"): True, ("CANARY", "a~"): None}
    monkeypatch.setattr("incorrecter.meaning.build_verdict", lambda base, model, **kw: lambda c, k: answers[(c, k)])
    _load_train_script("evaluate_mlx.py").main(
        [
            "--model",
            "m",
            "--label",
            "cal",
            "--heldout",
            str(tmp_path / "heldout.jsonl"),
            "--out-dir",
            str(tmp_path / "eval"),
            "--meaning-judge",
            "http://127.0.0.1:18103/v1",
            "--calibrate-meaning",
            "both",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["clean"] == {"pairs": 2, "yes": 2, "no": 0, "unparsed": 0, "yes_rate": 1.0}
    assert report["mismatch"] == {"pairs": 2, "yes": 1, "no": 0, "unparsed": 1, "no_rate": 0.0}
    assert report["calibrated"] is False
    rows = read_jsonl(tmp_path / "eval" / "cal.calibration.jsonl")
    assert [(r["suite"], r["index"], r["verdict"]) for r in rows] == [
        ("clean", 0, True),
        ("clean", 1, True),
        ("mismatch", 0, True),
        ("mismatch", 1, None),
    ]
    assert "CANARY" not in (tmp_path / "eval" / "cal.calibration.jsonl").read_text()


def test_evaluate_mlx_meaning_max_tokens_reaches_the_judge(tmp_path, monkeypatch, capsys):
    # A thinking judge needs room to finish its answer; the default stays 8 (an obedient one-word judge).
    write_jsonl(tmp_path / "heldout.jsonl", [_heldout_row("d1")])
    seen = []

    def build(base, model, **kw):
        seen.append(kw.get("max_tokens"))
        return lambda c, k: True

    monkeypatch.setattr("incorrecter.meaning.build_verdict", build)
    argv = ["--model", "m", "--label", "cal", "--heldout", str(tmp_path / "heldout.jsonl")]
    argv += ["--out-dir", str(tmp_path / "eval"), "--meaning-judge", "http://127.0.0.1:18103/v1"]
    _load_train_script("evaluate_mlx.py").main([*argv, "--calibrate-meaning", "clean"])
    _load_train_script("evaluate_mlx.py").main([*argv, "--calibrate-meaning", "clean", "--meaning-max-tokens", "512"])
    assert seen == [8, 512]


def test_evaluate_mlx_min_p_reaches_the_sampler(tmp_path, stub_mlx_lm):
    # min-p drops tokens far below the top one: the copy stays exact while the typo positions, where the
    # model is genuinely unsure, keep their options. Off by default (0.0), so recorded runs are unchanged.
    f = tmp_path / "heldout.jsonl"
    write_jsonl(f, [_heldout_row()])
    argv = ["--model", "m", "--label", "t", "--heldout", str(f), "--out-dir", str(tmp_path / "eval"), "--temp", "1.0"]
    _load_train_script("evaluate_mlx.py").main(argv)
    assert stub_mlx_lm["min_p"] == 0.0
    _load_train_script("evaluate_mlx.py").main([*argv, "--min-p", "0.1"])
    assert stub_mlx_lm["min_p"] == 0.1
