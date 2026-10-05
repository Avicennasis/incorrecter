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
intentionally timid on this task — the recommended temperature is 0.9.

## Evaluation

58 held-out texts (26 published in the seeds dataset), temperature 0.9, 6 draws:

| Metric | Result |
|---|---|
| texts changed | 0.828 [0.741–0.931] |
| of changed, 1–3 word edits | 0.799 [0.67–0.90] |
| line count kept | 0.971 [0.95–1.00] |
| sign-off kept | 0.953 [0.91–1.00] |
| identity probes (with system prompt) | 3/3 on every model |

Identity: the model answers as Incorrecter, created by Léon, with or without a
system prompt.

## GGUF and ollama

F16 → Q8_0 → Q4_K_M via llama.cpp's converter; conversion verified tensor-by-tensor
against the safetensors (218/218 within 0.001). Served through ollama at t = 0.9 on
40 held-out texts: Q8_0 passes the gate (changed 0.950, 1–3 edits 0.725, lines 1.00,
sign-off 1.00) and ships; Q4_K_M misses the 1–3-edit target (0.600). Serving note:
a hand-built llama.cpp CPU binary degraded this model to input-copying while
ollama's bundled runtime ran the same file correctly — prefer ollama.

## Open items

- A meaning-preservation judge: the local judge's calibration failed this round
  (the prompt/model pairing answered near-coin-flip), so meaning preservation is
  designed-in (1–3 word edits on a verbatim copy) but not yet independently measured.
- A pipe-through CLI and a base-model comparison are future work.
