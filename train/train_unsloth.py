"""LoRA-train Incorrecter with Unsloth on an NVIDIA GPU and export a q4_k_m GGUF for Ollama.

Written against unsloth 2026.9.11, which pins trl>=0.18.2,<=0.24.0 (TRL 1.x will not install
alongside it). Not yet run: it needs an NVIDIA GPU (item 1 ran the MLX path). Reads the same data/train.jsonl and
data/valid.jsonl that train/train_mlx.sh uses; produce them with incorrecter.dataset and
incorrecter.convert_mlx.
"""

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 1024


def main() -> None:
    # unsloth must be imported before trl and transformers so its patches apply.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files={"train": "data/train.jsonl", "valid": "data/valid.jsonl"})
    bf16 = is_bfloat16_supported()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        args=SFTConfig(
            max_length=MAX_LENGTH,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=10,
            max_steps=120,
            learning_rate=2e-4,
            bf16=bf16,
            fp16=not bf16,
            logging_steps=10,
            # Evaluate on data/valid.jsonl; TRL's default eval_strategy is "no", which ignores eval_dataset.
            eval_strategy="steps",
            eval_steps=20,
            output_dir="incorrecter_lora",
            report_to="none",
        ),
    )
    # Loss on the assistant turn only (Qwen2.5 ChatML markers), so the model learns to corrupt, not to echo.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()
    model.save_pretrained_gguf("incorrecter_gguf", tokenizer, quantization_method="q4_k_m")


if __name__ == "__main__":
    main()
