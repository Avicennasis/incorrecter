"""Rewrite every held-out text with a fused model and run the identity probes. Mac-only (needs mlx-lm).

    python train/evaluate_mlx.py --model ./incorrecter-fused --label v2

Greedy decoding. Writes eval/<label>.jsonl (human rows, and every row from data/private/, keep scores only, never
text, spec S13) and eval/<label>.identity.jsonl, and prints the summaries that go into docs/design-notes.md.
"""

import argparse
import json
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incorrecter import SYSTEM_PROMPT
from incorrecter.evaluate import PARAPHRASE_PROBES, identity_ok, rewrite_record, summarize
from incorrecter.identity import PROBE_QUESTIONS
from incorrecter.jsonl import read_jsonl, write_jsonl


def is_private(path: Path) -> bool:
    """A file under data/private/ holds human text whatever its rows' licenses say (spec S13). Both the path as given
    (data/private/ may be a symlink to storage elsewhere) and its resolved form (a link into data/private/) count."""
    return any(("data", "private") in pairwise(p.parts) for p in (path.absolute(), path.resolve()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fused Incorrecter model on the held-out set.")
    parser.add_argument("--model", required=True, help="fused model directory")
    parser.add_argument("--label", required=True, help="v1, v1b or v2")
    parser.add_argument(
        "--heldout",
        type=Path,
        nargs="+",
        default=[Path("data/heldout.jsonl"), Path("data/private/heldout_human.jsonl")],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval"))
    parser.add_argument("--temp", type=float, default=0.0, help="0 keeps greedy decoding (the recorded rows)")
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument(
        "--min-p", type=float, default=0.0, help="drop tokens below min-p x the top token's probability"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--meaning-judge", help="base URL of the local meaning judge")
    parser.add_argument("--meaning-model", default="/Users/Shared/ai-models/mlx/Qwen3.8-27B-8bit")
    parser.add_argument(
        "--calibrate-meaning",
        choices=["clean", "mismatch", "both"],
        help="run the calibration suites instead of a model evaluation",
    )
    parser.add_argument(
        "--meaning-max-tokens", type=int, default=8, help="judge reply budget; raise it for a thinking judge"
    )
    args = parser.parse_args(argv)
    # Before the model loads: a missing private file would otherwise surface only after every drafted row ran.
    missing = [str(p) for p in args.heldout if not p.exists()]
    if missing:
        parser.error(f"missing held-out file(s): {', '.join(missing)}; pass --heldout to evaluate on fewer files")

    if args.calibrate_meaning:
        import random as _random

        from incorrecter import meaning

        verdict = meaning.build_verdict(args.meaning_judge, args.meaning_model, max_tokens=args.meaning_max_tokens)
        texts = [r["text"] for p in args.heldout for r in read_jsonl(p)]
        clean, mismatched = meaning.calibration_pairs(texts, _random.Random(20260927))
        suites = {"clean": clean, "mismatch": mismatched}
        modes = list(suites) if args.calibrate_meaning == "both" else [args.calibrate_meaning]
        # Each suite reports on its own: pooled yes/no cannot tell a near-pass from a coin-flip.
        report: dict = {"mode": args.calibrate_meaning}
        records = []
        for mode in modes:
            results = [verdict(a, b2) for a, b2 in suites[mode]]
            records += [{"suite": mode, "index": i, "verdict": r} for i, r in enumerate(results)]
            yes = sum(1 for r in results if r is True)
            no = sum(1 for r in results if r is False)
            stats = {"pairs": len(results), "yes": yes, "no": no, "unparsed": len(results) - yes - no}
            if mode == "clean":
                stats["yes_rate"] = round(yes / len(results), 4) if results else 0.0
            else:
                stats["no_rate"] = round(no / len(results), 4) if results else 0.0
            report[mode] = stats
        if len(modes) == 2:
            report["calibrated"] = meaning.calibrated(report["clean"]["yes_rate"], report["mismatch"]["no_rate"])
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out_dir / f"{args.label}.calibration.jsonl", records)
        print(json.dumps(report))
        return

    from mlx_lm import generate, load

    model, tokenizer = load(args.model)
    sampler = None
    if args.temp > 0:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(args.seed)
        sampler = make_sampler(args.temp, top_p=args.top_p, min_p=args.min_p)

    def ask(messages: list[dict[str, str]], max_tokens: int) -> str:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False, sampler=sampler)

    system = {"role": "system", "content": SYSTEM_PROMPT}
    rewrites = []
    errors = 0
    meaning_records = []
    meaning_errors = 0
    meaning_verdict = None
    if args.meaning_judge:
        from incorrecter import meaning

        meaning_verdict = meaning.build_verdict(
            args.meaning_judge, args.meaning_model, max_tokens=args.meaning_max_tokens
        )
    for path in args.heldout:
        human = is_private(path)
        for row in read_jsonl(path):
            budget = int(len(row["text"]) / 3.5 * 1.5) + 32
            try:
                output = ask([system, {"role": "user", "content": row["text"]}], budget)
            except Exception as e:  # noqa: BLE001 — any failure here must become a counted error record
                # A traceback can carry the prompt, which may be human text: record the type name only.
                rewrites.append({"id": row.get("id"), "error": type(e).__name__, "human": human})
                errors += 1
                continue
            rewrites.append(rewrite_record(row, output, human=human))
            if meaning_verdict is not None:
                try:
                    value = meaning_verdict(row["text"], output)
                    meaning_records.append({"id": row.get("id"), "verdict": value, "human": human})
                except Exception as e:  # noqa: BLE001 — a judge failure is a counted record, never a crash
                    meaning_records.append({"id": row.get("id"), "error": type(e).__name__, "human": human})
                    meaning_errors += 1
    identity = []
    for question in (*PROBE_QUESTIONS, *PARAPHRASE_PROBES):
        for with_system in (True, False):
            messages = [system] if with_system else []
            answer = ask([*messages, {"role": "user", "content": question}], 96)
            identity.append(
                {
                    "question": question,
                    "system_prompt": with_system,
                    "answer": answer,
                    "ok": identity_ok(answer),
                    "trained_probe": question in PROBE_QUESTIONS,
                }
            )
    write_jsonl(args.out_dir / f"{args.label}.jsonl", rewrites)
    write_jsonl(args.out_dir / f"{args.label}.identity.jsonl", identity)
    if meaning_verdict is not None:
        write_jsonl(args.out_dir / f"{args.label}.meaning.jsonl", meaning_records)
    report = {
        "label": args.label,
        "all": summarize([r for r in rewrites if "error" not in r]),
        "drafted": summarize([r for r in rewrites if "error" not in r and not r["human"]]),
        "human": summarize([r for r in rewrites if "error" not in r and r["human"]]),
        "errors": errors,
        "trained_probes_passed_with_system_prompt": sum(
            r["ok"] for r in identity if r["system_prompt"] and r["trained_probe"]
        ),
        "identity": [{k: r[k] for k in ("question", "system_prompt", "ok")} for r in identity],
    }
    if meaning_verdict is not None:
        report["meaning_errors"] = meaning_errors
    print(json.dumps(report, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
