import random
from pathlib import Path

import pytest

from incorrecter import SYSTEM_PROMPT, TASK_INSTRUCTION
from incorrecter.convert_mlx import MLX_BATCH_SIZE, group_key, main, split_rows, to_messages
from incorrecter.dataset import main as dataset_main
from incorrecter.jsonl import read_jsonl

SEEDS = Path(__file__).resolve().parent.parent / "data" / "seeds.txt"


def task(text: str, output: str) -> dict[str, str]:
    return {"instruction": TASK_INSTRUCTION, "input": text, "output": output}


def ident(question: str, answer: str) -> dict[str, str]:
    return {"instruction": question, "input": "", "output": answer}


def test_task_row_puts_input_in_user_turn():
    messages = to_messages(task("clean text", "cleen text"))["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "clean text"
    assert messages[2]["content"] == "cleen text"


def test_identity_row_puts_question_in_user_turn():
    assert to_messages(ident("Who are you?", "I am Incorrecter."))["messages"][1]["content"] == "Who are you?"


CANARY = "Canary text that may be human and must never reach an error message."


def test_row_without_input_or_instruction_raises():
    with pytest.raises(ValueError) as error:
        to_messages({"id": "bad-1", "instruction": "", "input": "", "output": CANARY})
    assert "bad-1" in str(error.value) and CANARY not in str(error.value)


def test_split_never_puts_a_source_on_both_sides():
    rows = [task(f"seed {i}", f"seed {i} v{j}") for i in range(10) for j in range(4)]
    rows += [ident(f"q{i}", f"a{j}") for i in range(20) for j in range(3)]
    train, valid = split_rows(rows, random.Random(0), 0.1)
    assert {group_key(r) for r in train}.isdisjoint({group_key(r) for r in valid})
    assert len(train) + len(valid) == len(rows)


def test_split_holds_out_both_kinds():
    rows = [task(f"seed {i}", "x") for i in range(4)] + [ident(f"q{i}", "a") for i in range(30)]
    _, valid = split_rows(rows, random.Random(0), 0.1)
    assert {group_key(r)[0] for r in valid} == {"task", "identity"}


def test_single_group_cannot_split():
    with pytest.raises(ValueError):
        split_rows([task("only", "onyl"), task("only", "olny")], random.Random(0))


@pytest.mark.parametrize("ratio", [0, 1, 1.5, -0.1])
def test_bad_ratio_raises(ratio):
    with pytest.raises(ValueError):
        split_rows([task("a", "b"), task("c", "d")], random.Random(0), ratio)


def test_cli_end_to_end_from_shipped_seeds(tmp_path, capsys):
    data = tmp_path / "incorrecter_data.jsonl"
    dataset_main(["--seeds", str(SEEDS), "-o", str(data)])
    main(["-i", str(data), "-o", str(tmp_path / "data")])
    train = read_jsonl(tmp_path / "data" / "train.jsonl")
    valid = read_jsonl(tmp_path / "data" / "valid.jsonl")
    assert all(len(row["messages"]) == 3 for row in train + valid)
    # mlx_lm.lora refuses a split smaller than its batch size, so the shipped seeds must clear it.
    assert len(valid) >= MLX_BATCH_SIZE
    assert "warning" not in capsys.readouterr().err


def test_identity_probe_groups_always_train():
    from incorrecter.identity import PROBE_QUESTIONS, identity_rows

    rows = identity_rows(random.Random(0), 110) * 5
    rows += [{"instruction": "x", "input": f"Seed {i}.", "output": f"seed {i}"} for i in range(40)]
    for seed in range(30):
        train, valid = split_rows(rows, random.Random(seed))
        assert not {r["instruction"] for r in valid} & set(PROBE_QUESTIONS)
        assert set(PROBE_QUESTIONS) <= {r["instruction"] for r in train}


def test_ids_ride_along_and_long_rows_are_dropped(tmp_path):
    from incorrecter.convert_mlx import MAX_ESTIMATED_TOKENS, estimated_tokens, main
    from incorrecter.jsonl import read_jsonl, write_jsonl

    long_text = "word " * 1500
    assert estimated_tokens({"input": long_text, "output": long_text}) > MAX_ESTIMATED_TOKENS
    rows = [
        {"id": f"work-{i:04d}#0", "instruction": "x", "input": f"Seed number {i}.", "output": f"seed number {i}"}
        for i in range(30)
    ]
    rows.append({"id": "long#0", "instruction": "x", "input": long_text, "output": long_text})
    src = tmp_path / "data.jsonl"
    write_jsonl(src, rows)
    main(["-i", str(src), "-o", str(tmp_path)])
    out = read_jsonl(tmp_path / "train.jsonl") + read_jsonl(tmp_path / "valid.jsonl")
    assert len(out) == 30 and "long#0" not in {r["id"] for r in out}
    assert all(set(r) == {"messages", "id"} for r in out)


def test_dropped_rows_are_reported_by_id_or_row_number(tmp_path, capsys):
    from incorrecter.jsonl import write_jsonl

    long_text = "word " * 1500
    rows = [
        {"id": f"s{i}#0", "instruction": "x", "input": f"Seed number {i}.", "output": f"seed {i}"} for i in range(30)
    ]
    rows.append({"id": "long#0", "instruction": "x", "input": long_text, "output": long_text})
    rows.append({"instruction": "x", "input": long_text + "Two.", "output": long_text})  # no id: row 32
    src = tmp_path / "data.jsonl"
    write_jsonl(src, rows)
    main(["-i", str(src), "-o", str(tmp_path)])
    err = capsys.readouterr().err
    assert "dropped 2 rows" in err and "long#0" in err and "row 32" in err
    assert "word word" not in err  # ids and row numbers only, never the text


def test_a_missing_input_file_is_a_usage_error(tmp_path, capsys):
    missing = tmp_path / "incorrecter_data.jsonl"
    with pytest.raises(SystemExit) as exit_info:
        main(["-i", str(missing), "-o", str(tmp_path)])
    assert exit_info.value.code == 2
    assert str(missing) in capsys.readouterr().err


@pytest.mark.parametrize(("rows", "extra"), [([], []), ([task("a", "b"), task("c", "d")], ["--valid-ratio", "1.5"])])
def test_input_that_cannot_split_is_a_usage_error(tmp_path, capsys, rows, extra):
    from incorrecter.jsonl import write_jsonl

    src = tmp_path / "data.jsonl"
    write_jsonl(src, rows)
    with pytest.raises(SystemExit) as exit_info:
        main(["-i", str(src), "-o", str(tmp_path / "out"), *extra])
    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("bad", "name"),
    [({"id": "bad-1", "instruction": "", "input": "", "output": CANARY}, "bad-1"), ({"output": CANARY}, "row 31")],
)
def test_a_row_with_neither_input_nor_instruction_is_a_usage_error(tmp_path, capsys, bad, name):
    from incorrecter.jsonl import write_jsonl

    src = tmp_path / "data.jsonl"
    write_jsonl(src, [task(f"Seed number {i}.", f"seed {i}") for i in range(30)] + [bad])
    with pytest.raises(SystemExit) as exit_info:
        main(["-i", str(src), "-o", str(tmp_path / "out")])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert name in err and CANARY not in err
    assert not (tmp_path / "out").exists()
