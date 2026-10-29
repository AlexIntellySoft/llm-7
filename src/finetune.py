"""Дообучение LoRA-адаптера (4-bit QLoRA + PEFT + trl.SFTTrainer) и генерация на тесте.

Этим модулем пользуется Colab-ноутбук notebooks/train_colab.ipynb. Локально его можно
прогнать на крошечной модели без квантования (tools/smoke_test.py), чтобы проверить код
до запуска на GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def prompt_messages(row: dict) -> list[dict]:
    return [{"role": "user", "content": f"{row['instruction']}:\n{row['input']}"}]


def to_dataset(rows: list[dict]) -> Dataset:
    # Формат prompt/completion: SFTTrainer считает loss только по ответу (ECQL),
    # а не по русскому запросу, который модели и так дан
    return Dataset.from_list([
        {"prompt": prompt_messages(r), "completion": [{"role": "assistant", "content": r["output"]}]}
        for r in rows
    ])


def load_base(model_id: str = BASE_MODEL, quantize: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # T4 не умеет bf16
        )
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tokenizer


def train_adapter(
    train_rows: list[dict],
    eval_rows: list[dict],
    r: int,
    lora_alpha: int,
    out_dir: str,
    model_id: str = BASE_MODEL,
    quantize: bool = True,
    epochs: float = 3,
    max_steps: int = -1,
) -> list[dict]:
    """Обучает адаптер с рангом r, сохраняет его в out_dir и возвращает историю loss."""
    model, tokenizer = load_base(model_id, quantize)
    if quantize:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # С 4-bit загрузкой веса LoRA получают dtype модели из её конфига (у Qwen3 — bfloat16),
    # а fp16-обучение на T4 (GradScaler) обновляет только fp32-веса
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # эффективный батч 8
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=4,
        save_strategy="no",
        max_length=256,
        fp16=quantize,
        gradient_checkpointing=quantize,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=to_dataset(train_rows),
        eval_dataset=to_dataset(eval_rows),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)  # сохраняются только веса LoRA (adapter_model.safetensors)
    history = trainer.state.log_history
    Path(out_dir, "log_history.json").write_text(json.dumps(history, indent=1))
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return history


def attach_adapters(base_model, adapters: dict[str, str]) -> PeftModel:
    """Подключает обученные адаптеры к базовой модели; активный выбирается set_adapter."""
    names = list(adapters)
    model = PeftModel.from_pretrained(base_model, adapters[names[0]], adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(adapters[name], adapter_name=name)
    model.eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, rows: list[dict], batch_size: int = 8, max_new_tokens: int = 128) -> list[str]:
    tokenizer.padding_side = "left"
    outputs = []
    for i in range(0, len(rows), batch_size):
        texts = [
            tokenizer.apply_chat_template(prompt_messages(r), tokenize=False, add_generation_prompt=True)
            for r in rows[i:i + batch_size]
        ]
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        for seq in out[:, batch["input_ids"].shape[1]:]:
            text = tokenizer.decode(seq, skip_special_tokens=True).strip()
            outputs.append(text.splitlines()[0].strip() if text else "")
    return outputs


def loss_points(history: list[dict]) -> tuple[list, list, list, list]:
    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    return [s for s, _ in train], [v for _, v in train], [s for s, _ in evals], [v for _, v in evals]
