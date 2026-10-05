# Design notes

Measurements behind Incorrecter. Every number on this page is a measurement from
the runs below.

## The task: rewrite, not generation

The model receives clean text and must return the same text with a few word-level
human errors. Anything beyond that — answering, continuing, rewriting — is failure.
This is why the training pairs are verbatim copies with injected errors, and why the
evaluation scores line counts and sign-offs rather than fluency.

## Base model

Qwen2.5-0.5B-Instruct: Apache 2.0, tiny, easy to steer with LoRA. A 4-bit fusion
erases this small a LoRA delta (the rounding swallows it), so training fuses to fp16.

## Training data

1,337 clean seeds (644 drafted by GLM/Qwen/Gemini/Claude, 693 human-written from
OpenAssistant/oasst2 and google/civil_comments), 600 real-error pairs from
grammarly/coedit reversed clean → erroneous, and identity rows. Every seed passes a
deterministic clean gate (wrapper, known-error, length, spelling, truncation and
sign-off rules) plus a local-model cleanliness judge.

## The recipe

mlx-lm LoRA, rank 8, learning rate 1e-4, 2,000 iterations, batch 8, fused to fp16
(`--dequantize`; a 4-bit fusion erases this small a delta). Greedy decoding is
intentionally timid on this task. Recommended sampling: **temperature 1.2, top_p 0.9,
no repetition penalty** (see "Meaning" below for why).

## Evaluation

58 held-out texts (26 published in the seeds dataset), temperature 1.2 + top_p 0.9,
6 draws (2 training seeds × 3 sampling seeds), mlx-lm:

| Metric | Result | Target |
|---|---|---|
| texts changed | 0.830 [0.76–0.90] | ≥ 0.80 |
| of changed, 1–3 word edits | 0.882 [0.81–0.92] | ≥ 0.70 |
| line count kept | 0.971 [0.97–0.98] | ≥ 0.95 |
| sign-off kept | 0.958 [0.91–1.00] | ≥ 0.95 |
| meaning kept | 0.957 [0.93–0.98] | ≥ 0.95 |
| identity probes (with system prompt) | 3/3 on every model | 3/3 |

The published checkpoint on its own (3 draws): 0.874 changed, 0.901 with 1–3 edits,
0.971 lines, 0.958 sign-off, 0.954 meaning.

Identity: the model answers as Incorrecter, created by Léon, with or without a
system prompt.

## Meaning

**The judge.** Llama-3.3-70B-Instruct-4bit (local) answers whether each output still
says what the input said, ignoring surface errors (`incorrecter/meaning.py`). Before
scoring any model it was calibrated on two sets:

- 58/58 noise-engine corruptions judged "same meaning" (target ≥ 0.98)
- 58/58 mismatched pairs (a text against another text's corruption) judged
  "different" (target ≥ 0.95)

A 27B judge passed by one pair (57/58). Earlier "coin-flip" readings were a harness bug:
the calibration printed pooled counts for both sets, which cannot separate a coin-flip
from a near-pass. It now reports each set's rate.

**The edit contract.** `incorrecter/contract.py` names every edit the noise engine never
makes:

- a real word swapped for a different real word outside the word lists (my → your)
- a word garbled past a two-letter typo (credit → cambic)
- words added or dropped

On 600 judged outputs it caught 40 of the 41 meaning changes, and 361 of the 362
in-contract outputs were judged meaning-kept. So it works as a free stand-in for the
judge. The training generator stays inside the contract 98% of the time.

**The fix was sampling, not training.** At temperature 0.9 with no cap, meaning was
0.914, below target, and about 40% of outputs left the contract. Every token was
sampled, including the tokens the model should copy verbatim, so stray word swaps, garbles and
dropped words slipped into the copy.

A top_p 0.9 cap cuts that tail and keeps the typo positions, where the model is
genuinely unsure. Raising the temperature to 1.2 keeps "changed" above target.

Retraining without the off-contract real-error pairs did not reliably help: the spread
between training seeds was larger than the effect.

**Sampling sweep** (published checkpoint, 3 draws each):

| Setting | changed | 1–3 edits | lines | sign-off | off-contract |
|---|---|---|---|---|---|
| t 0.9 (old recipe) | 0.828 | 0.864 | 0.977 | 0.938 | 0.383 |
| t 0.9, top_p 0.9 | 0.759 | 0.947 | 0.989 | 0.948 | 0.192 |
| t 1.0, top_p 0.9 | 0.816 | 0.937 | 0.977 | 0.938 | 0.200 |
| **t 1.2, top_p 0.9** | **0.874** | **0.901** | **0.971** | **0.958** | **0.208** |

**Other runtimes** (same checkpoint, 40 held-out texts, 3 draws):

| Runtime | Settings | changed | 1–3 edits | lines | sign-off | meaning |
|---|---|---|---|---|---|---|
| transformers (CPU) | t 1.2, top_p 0.9, no repetition penalty | 0.79 | 0.81 | 0.96 | 0.96 | 0.94 |
| transformers (CPU) | old config: t 0.9, repetition penalty 1.05 (1 draw) | 1.00 | 0.53 | 0.93 | 0.88 | — |
| ollama Q8_0 | t 1.0, top_p 0.9, top_k 40, repeat_penalty 1.0 | 0.93 | 0.85 | 0.98 | 0.97 | 0.94 |
| ollama Q8_0 | old: t 0.9 + ollama defaults (repeat_penalty 1.1) | 0.98 | 0.62 | 0.98 | 0.97 | — |

A repetition penalty is the wrong tool here: it penalizes every token already present,
including the prompt. That pushes the model off the verbatim copy that is the whole job.

## GGUF and ollama

F16 → Q8_0 → Q4_K_M via llama.cpp's converter; conversion verified tensor-by-tensor
against the safetensors (218/218 within 0.001). Served through ollama at t = 0.9 on
40 held-out texts: Q8_0 passes the gate and ships; Q4_K_M misses the 1–3-edit target
(0.600). A 4-bit MLX re-quantize also fails, on lines (0.925) and sign-off (0.885).
The GGUF repo's `params` file pins the sampling above (see "Meaning"). Serving note:
a hand-built llama.cpp CPU binary degraded this model to input-copying while
ollama's bundled runtime ran the same file correctly — prefer ollama.

## Open items

- A near-miss calibration set for the judge: one meaning-changing edit (a dropped
  "not", a changed number). This would bound how often the judge misses a subtle
  change.
- A pipe-through CLI and a base-model comparison are future work.
