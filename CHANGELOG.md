# Changelog

All notable changes to `incorrecter` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-10-05

### Added
- `incorrecter/contract.py`: `off_contract()` names every edit the noise engine never
  makes:
  - a real word swapped for a different real word outside the word lists
  - a word garbled past a two-letter typo
  - words added or dropped

  It is a free stand-in for the meaning judge: it caught 40 of 41 judged meaning
  changes, and 361 of 362 in-contract outputs were judged meaning-kept.
- `train/evaluate_mlx.py`: `--min-p` and `--meaning-max-tokens`.

### Changed
- Recommended sampling is now temperature 1.2, top_p 0.9, no repetition penalty, up
  from temperature 0.9. At these settings the model keeps meaning on 0.957 of held-out
  texts and meets every target. The HF `generation_config.json`, the GGUF repo's ollama
  `params` and the `Modelfile` carry the new settings.

### Fixed
- `--calibrate-meaning both` reports each suite's rate and the `calibrated()`
  verdict. It used to print pooled yes/no counts, which cannot tell a coin-flip from a
  near-pass. It also writes per-pair verdicts (no text).
- `meaning.parse_verdict` reads through markdown, an `Answer:` label and a finished
  think block. It no longer reads "nothing" as "no".

## [0.4.0] - 2026-10-04

### Added
- First public release: the span-based noise engine (eggcorns, homophones, fat-finger
  slips, lowercasing, a doubled space, a dropped final period), the clean gate, the
  dataset builder and mlx-lm converter, the mlx-lm training and evaluation scripts, and
  the meaning and memorization checks.
- Weights: [Avicennasis/incorrecter](https://huggingface.co/Avicennasis/incorrecter)
  (MLX fp16) and [Avicennasis/incorrecter-GGUF](https://huggingface.co/Avicennasis/incorrecter-GGUF)
  (Q8_0). Training seeds:
  [Avicennasis/incorrecter-seeds](https://huggingface.co/datasets/Avicennasis/incorrecter-seeds).
