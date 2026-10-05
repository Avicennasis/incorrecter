# Changelog

All notable changes to `incorrecter` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
