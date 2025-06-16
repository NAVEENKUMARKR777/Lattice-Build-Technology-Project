import argparse
import glob
import json
from pathlib import Path
from typing import List
import sys
import math

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

DEFAULT_MODEL = "microsoft/phi-3-mini-4k-instruct"
CHUNK_DEFAULT = 256


def load_jsonl_to_dataset(files: List[str]):
    """Load JSON docs -> HF Dataset with `text` field"""
    records = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            docs = json.load(f)
        for d in docs:
            records.append({"text": f"{d['title']}\n\n{d['summary']}"})
    return Dataset.from_list(records)


def tokenize_and_chunk(examples, tokenizer, chunk_size=256):
    tokens = tokenizer(
        examples["text"],
        return_attention_mask=False,
        truncation=False,
    )["input_ids"]

    # Flatten list of lists -> single long list then chunk
    concatenated = []
    for ids in tokens:
        concatenated.extend(ids + [tokenizer.eos_token_id])

    # Split into chunks
    result = {
        "input_ids": [concatenated[i : i + chunk_size] for i in range(0, len(concatenated), chunk_size)]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Phi-3 on quantum physics corpus")
    parser.add_argument("--data_dir", type=str, default="data/papers", help="Folder containing JSON dumps from download_papers.py")
    parser.add_argument("--output_dir", type=str, default="models/phi3_quant", help="Where to save LoRA adapters")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Base Phi-3 model name or path")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    json_files = glob.glob(str(Path(args.data_dir) / "*.json"))
    assert json_files, f"No JSON files found in {args.data_dir}. Run download_papers.py first."

    print(f"[+] Loading {len(json_files)} files…")
    dataset = load_jsonl_to_dataset(json_files)

    print(f"Dataset size: {len(dataset)} docs")

    print("[+] Loading base model…")

    # For Windows (no GPU bitsandbytes) load full model in FP16 directly onto GPU.
    print("[+] Moving model to GPU (FP16)…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    ).to("cuda")

    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required when using gradient checkpointing

    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    )

    model = get_peft_model(model, lora_cfg)
    print(model)

    # Tokenise + chunk dataset lazily
    tokenised = dataset.map(
        lambda batch: tokenize_and_chunk(batch, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        gradient_accumulation_steps=max(4, math.ceil(16 / args.batch)),  # ensure effective batch 16
        logging_steps=25,
        save_strategy="epoch",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenised,
        args=training_args,
        data_collator=data_collator,
    )

    print("[+] Starting training…")
    trainer.train()

    print("[+] Saving LoRA adapters →", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main() 