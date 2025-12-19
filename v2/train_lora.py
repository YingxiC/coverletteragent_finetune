\
#!/usr/bin/env python3
"""
train_lora.py (v4-fixed2)

Fixes:
- Works across Transformers versions: eval_strategy vs evaluation_strategy
- Works across TRL versions: processing_class vs tokenizer; max_length vs max_seq_length
- Explicitly applies PEFT LoRA (get_peft_model) to guarantee trainable params
- Enables input grads for gradient checkpointing to avoid "requires_grad" issues
- Heartbeat prints so it never looks stuck

Supports:
A) --inputs <jsonl...> + --val_ratio
B) --train_inputs ... --val_inputs ...

JSONL line keys:
  resume: str
  job_description: str
  output: str
Optional:
  instruction: str
"""

from __future__ import annotations

import argparse
import inspect
import platform
import time
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer

from peft import LoraConfig, get_peft_model

# prepare_model_for_kbit_training exists in peft>=0.5; import if available
try:
    from peft import prepare_model_for_kbit_training  # type: ignore
except Exception:  # pragma: no cover
    prepare_model_for_kbit_training = None


DEFAULT_SYSTEM_RULES = (
    "You write a tailored cover letter using ONLY the resume content. "
    "Do NOT invent degrees, employers, titles, dates, or metrics. "
    "If something is missing, write more generally and truthfully."
)

DEFAULT_USER_TASK = (
    "Write a professional cover letter (250–400 words) tailored to the job. "
    "Use specific evidence from the resume. Do not fabricate details."
)


class HeartbeatCallback(TrainerCallback):
    """Print a heartbeat every N seconds so training never looks stuck."""
    def __init__(self, every_secs: int = 30):
        self.every_secs = max(5, int(every_secs))
        self.last_t = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"[heartbeat] training started | every {self.every_secs}s", flush=True)
        self.last_t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if now - self.last_t >= self.every_secs:
            last_loss = None
            for item in reversed(state.log_history):
                if "loss" in item:
                    last_loss = item["loss"]
                    break
            if last_loss is not None:
                print(f"[heartbeat] step={state.global_step} epoch={state.epoch:.4f} last_loss={last_loss:.4f}", flush=True)
            else:
                print(f"[heartbeat] step={state.global_step} epoch={state.epoch:.4f}", flush=True)
            self.last_t = now


def _print_env():
    print("\n==== Environment ====", flush=True)
    print(f"Platform: {platform.platform()}", flush=True)
    print(f"Python: {platform.python_version()}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}", flush=True)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}", flush=True)
        try:
            print(f"BF16 supported: {torch.cuda.is_bf16_supported()}", flush=True)
        except Exception:
            pass
    print("=====================\n", flush=True)


def _load_jsonl_files(paths: List[str]):
    return load_dataset("json", data_files=paths, split="train")


def _ensure_fields(example: Dict[str, Any]) -> Dict[str, Any]:
    example["instruction"] = (example.get("instruction") or "").strip()
    example["resume"] = (example.get("resume") or "").strip()
    example["job_description"] = (example.get("job_description") or "").strip()
    example["output"] = (example.get("output") or "").strip()
    return example


def _build_messages(example: Dict[str, Any], system_rules: str):
    instruction = example.get("instruction", "").strip()
    resume = example["resume"]
    jd = example["job_description"]
    out = example["output"]

    sys = system_rules
    if instruction:
        sys = f"{system_rules}\nAdditional instruction: {instruction}"

    user = f"RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{jd}\n\n{DEFAULT_USER_TASK}"
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
        {"role": "assistant", "content": out},
    ]


def _format_for_sft(tokenizer, system_rules: str):
    def _fn(example: Dict[str, Any]) -> Dict[str, Any]:
        example = _ensure_fields(example)
        if not (example["resume"] and example["job_description"] and example["output"]):
            return {"text": ""}

        messages = _build_messages(example, system_rules=system_rules)

        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = (
                f"SYSTEM: {messages[0]['content']}\n"
                f"USER:\n{messages[1]['content']}\n"
                f"ASSISTANT:\n{messages[2]['content']}"
            )
        return {"text": text}

    return _fn


def _count_trainable_params(model) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(1, total)
    print("==== Parameters ====", flush=True)
    print(f"Total params: {total:,}", flush=True)
    print(f"Trainable params: {trainable:,} ({pct:.4f}%)", flush=True)
    print("====================\n", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, required=True)

    # Option A
    ap.add_argument("--inputs", type=str, nargs="*", default=None)
    ap.add_argument("--val_ratio", type=float, default=0.2)

    # Option B
    ap.add_argument("--train_inputs", type=str, nargs="*", default=None)
    ap.add_argument("--val_inputs", type=str, nargs="*", default=None)

    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=200)

    ap.add_argument("--heartbeat_secs", type=int, default=15)
    ap.add_argument("--system_rules", type=str, default=DEFAULT_SYSTEM_RULES)

    return ap.parse_args()


def main():
    args = parse_args()
    _print_env()

    use_explicit_split = (
        args.train_inputs and args.val_inputs and len(args.train_inputs) > 0 and len(args.val_inputs) > 0
    )
    if not use_explicit_split and not args.inputs:
        raise SystemExit("Need either --inputs ... OR --train_inputs ... --val_inputs ...")

    print("==== Config ====", flush=True)
    print(f"Base model: {args.model_name}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Max seq len: {args.max_seq_len}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Grad accum: {args.grad_accum}", flush=True)
    print(f"LR: {args.lr}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}", flush=True)
    print(f"Heartbeat every: {args.heartbeat_secs}s", flush=True)
    print("==============\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_explicit_split:
        print("Loading explicit train/val files...", flush=True)
        train_ds = _load_jsonl_files(args.train_inputs).map(_ensure_fields)
        val_ds = _load_jsonl_files(args.val_inputs).map(_ensure_fields)
    else:
        print("Loading inputs and splitting by val_ratio...", flush=True)
        full = _load_jsonl_files(args.inputs).map(_ensure_fields)
        full = full.filter(lambda x: len(x["resume"]) > 0 and len(x["job_description"]) > 0 and len(x["output"]) > 0)
        split = full.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
        train_ds, val_ds = split["train"], split["test"]

    print("==== Dataset ====", flush=True)
    print(f"Train size: {len(train_ds)}", flush=True)
    print(f"Val size:   {len(val_ds)}", flush=True)
    print("=================\n", flush=True)

    fmt_fn = _format_for_sft(tokenizer, system_rules=args.system_rules)
    train_ds = train_ds.map(fmt_fn, remove_columns=train_ds.column_names).filter(lambda x: len(x["text"]) > 0)
    val_ds = val_ds.map(fmt_fn, remove_columns=val_ds.column_names).filter(lambda x: len(x["text"]) > 0)

    # Detect bitsandbytes availability (Windows often fails)
    bnb_ok = True
    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        bnb_ok = False
        print(f"[warn] bitsandbytes not available ({e}); disabling 4-bit + 8-bit optimizer.", flush=True)

    if bnb_ok:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
        )
    else:
        quant_cfg = None

    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16),
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Gradient checkpointing: safe enable
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Important: for gradient checkpointing + PEFT, ensure inputs require grads
    try:
        model.enable_input_require_grads()
    except Exception:
        try:
            emb = model.get_input_embeddings()
            emb.weight.requires_grad_(True)
        except Exception:
            pass

    # Prepare for k-bit training if available and using 4-bit
    if prepare_model_for_kbit_training is not None and bnb_ok:
        try:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        except TypeError:
            # older signature
            model = prepare_model_for_kbit_training(model)

    # Apply LoRA explicitly (guarantees trainable params)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    print(f"LoRA target modules: {target_modules}", flush=True)
    model = get_peft_model(model, lora_config)

    # Verify trainable params > 0
    _count_trainable_params(model)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # TrainingArguments API compatibility
    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        disable_tqdm=False,
        log_level="info",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        optim=("paged_adamw_8bit" if bnb_ok else "adamw_torch"),
        report_to="none",
        seed=args.seed,
    )
    sig_ta = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig_ta.parameters:
        ta_kwargs["eval_strategy"] = "steps"
    else:
        ta_kwargs["evaluation_strategy"] = "steps"

    training_args = TrainingArguments(**ta_kwargs)

    # SFTTrainer API compatibility
    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters
    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in params:
        trainer_kwargs["dataset_text_field"] = "text"

    if "max_seq_length" in params:
        trainer_kwargs["max_seq_length"] = args.max_seq_len
    elif "max_length" in params:
        trainer_kwargs["max_length"] = args.max_seq_len

    if "packing" in params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.add_callback(HeartbeatCallback(args.heartbeat_secs))

    steps_per_epoch = (len(train_ds) // (args.batch_size * max(1, torch.cuda.device_count()) * args.grad_accum)) + 1
    print(f"Estimated steps/epoch: {steps_per_epoch}", flush=True)
    print(f"Estimated total steps: {steps_per_epoch * args.epochs}", flush=True)

    print("Starting training...", flush=True)
    trainer.train()

    print("Saving LoRA adapter + tokenizer...", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n✅ Done.", flush=True)
    print(f"Adapter saved to: {args.output_dir}", flush=True)
    print("You should see: adapter_model.safetensors + adapter_config.json", flush=True)


if __name__ == "__main__":
    main()
