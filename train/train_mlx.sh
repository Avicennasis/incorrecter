#!/usr/bin/env bash
# LoRA-train Incorrecter with mlx-lm on Apple Silicon, then fuse the adapter into standalone weights.
# Written against mlx-lm 0.31.3; first run on macbook4 on 2026-09-27 (ROADMAP.md, item 1).
# Override any setting through the environment, e.g. LEARNING_RATE=1e-5 ITERS=200 train/train_mlx.sh
set -euo pipefail

MODEL="${MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
DATA="${DATA:-./data}"
ITERS="${ITERS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# Item 1 ran mlx-lm's default of 1e-5; an earlier Unsloth recipe used 2e-4.
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
# Training PRNG seed (mlx-lm's default is 0). Stage 3 trains each arm with SEED=0 and SEED=1
# (full/seed 0 may reuse v2 — see the public-release plan).
SEED="${SEED:-0}"
# LoRA target layers. Unset keeps mlx-lm's default; NUM_LAYERS=-1 targets every layer (a Stage 1b probe).
NUM_LAYERS="${NUM_LAYERS:-}"
EXTRA_LORA=()
if [[ -n "$NUM_LAYERS" ]]; then
  EXTRA_LORA+=(--num-layers "$NUM_LAYERS")
fi
ADAPTERS="${ADAPTERS:-./incorrecter-adapters}"
FUSED="${FUSED:-./incorrecter-fused}"

for split in train valid; do
  if [[ ! -s "$DATA/$split.jsonl" ]]; then
    echo "missing $DATA/$split.jsonl; run: python -m incorrecter.dataset && python -m incorrecter.convert_mlx" >&2
    exit 1
  fi
done

# mlx-lm truncates rows past --max-seq-length from the tail, where the sign-off is: drop them first, counted with
# the real tokenizer. Aborts if more than 1% are over (convert_mlx's estimate would be wrong).
python3 "$(dirname "$0")/check_lengths.py" --model "$MODEL" --data "$DATA" --max-tokens "$MAX_SEQ_LENGTH" \
  --report "$DATA/over_length.txt"

# --mask-prompt: loss on the assistant turn only, so the model learns to corrupt rather than to echo.
mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --batch-size "$BATCH_SIZE" \
  --iters "$ITERS" \
  --learning-rate "$LEARNING_RATE" \
  --seed "$SEED" \
  "${EXTRA_LORA[@]+"${EXTRA_LORA[@]}"}" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --mask-prompt \
  --adapter-path "$ADAPTERS"

# --dequantize: fusing back into a 4-bit base re-quantizes the merged weights, and the LoRA delta is smaller
# than one 4-bit step, so the fine-tune rounds away. Measured on macbook4 (M4 Max) 2026-09-27: the 4-bit fused
# model answered "I am Qwen" and left text untouched; the fp16 fused model (~1 GB) kept the fine-tune.
mlx_lm.fuse \
  --model "$MODEL" \
  --adapter-path "$ADAPTERS" \
  --save-path "$FUSED" \
  --dequantize

echo "fused model written to $FUSED"
