# Incorrecter

The opposite of autocorrect. Incorrecter takes clean, AI-sounding text — a thank-you
email, a landlord note, a Slack update, a short essay — and adds a few realistic human
errors: eggcorns, wrong homophones, fat-finger slips, a lowercase sentence start, a
doubled space, a dropped final period.

A Qwen2.5-0.5B-Instruct LoRA fine-tune (mlx-lm, Apple Silicon). Weights:
[Avicennasis/incorrecter](https://huggingface.co/Avicennasis/incorrecter) (MLX fp16) ·
GGUF: [Avicennasis/incorrecter-GGUF](https://huggingface.co/Avicennasis/incorrecter-GGUF) ·
Training seeds: [Avicennasis/incorrecter-seeds](https://huggingface.co/datasets/Avicennasis/incorrecter-seeds)

## Usage

The model expects clean text as the user message and returns the same text with 1–3
word-level errors. Recommended sampling: **temperature 1.2, top_p 0.9, no repetition
penalty**. The model repo's `generation_config.json` carries these. Greedy decoding is
intentionally timid, and a repetition penalty breaks the copy.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Avicennasis/incorrecter")
model = AutoModelForCausalLM.from_pretrained("Avicennasis/incorrecter", dtype=torch.float16)
prompt = tok.apply_chat_template([{"role": "user", "content": clean_text}],
                                 add_generation_prompt=True, tokenize=False)
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=200, do_sample=True, temperature=1.2, top_p=0.9)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

Or with ollama:

```
ollama run hf.co/Avicennasis/incorrecter-GGUF:Q8_0
```

The GGUF repo's `params` file sets temperature 1.0, top_p 0.9 and repeat_penalty 1.0.

## Building the training data

The rule engine (`incorrecter/noise.py`), the clean gate (`incorrecter/corpus/gate.py`)
and the dataset builder (`incorrecter/dataset.py`) generate the training pairs from the
seeds in [the dataset repo](https://huggingface.co/datasets/Avicennasis/incorrecter-seeds):

```bash
python -m incorrecter.dataset --seeds seeds.jsonl --real-errors coedit_real_errors.jsonl
python -m incorrecter.convert_mlx -i incorrecter_data.jsonl -o data
python train/train_mlx.sh   # mlx-lm LoRA on Apple Silicon, then fuse
```

`data/private/` is a gitignored working directory for optional local seed imports; the
public training set ships in the dataset repo and never requires it.

## Evaluation

58 held-out texts (26 in the dataset repo), temperature 1.2 + top_p 0.9, 6 draws:
- 0.83 of texts changed, and 0.88 of those had 1–3 word edits
- 0.97 kept their line count, and 0.96 kept their sign-off
- **0.96 kept their meaning**, as judged by a calibrated Llama-3.3-70B judge
Identity: the model answers as Incorrecter with or without a system prompt. Full
numbers in `docs/design-notes.md`.

## License

MIT for the code. Model weights inherit Apache-2.0 (base Qwen2.5-0.5B-Instruct);
seed rows carry per-row licenses (see the dataset card).
