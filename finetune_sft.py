import os
import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- Load Configuration ---
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()

# --- Configuration ---
# Model and Dataset
model_name = config["model"]["name"]
dataset_name = config["model"]["dataset_name"]
new_model = config["model"]["new_model_name"]

# QLoRA parameters
lora_r = config["lora"]["r"]
lora_alpha = config["lora"]["alpha"]
lora_dropout = config["lora"]["dropout"]

# bitsandbytes parameters
use_4bit = config["bitsandbytes"]["use_4bit"]
bnb_4bit_compute_dtype = config["bitsandbytes"]["compute_dtype"]
bnb_4bit_quant_type = config["bitsandbytes"]["quant_type"]
use_nested_quant = config["bitsandbytes"]["use_nested_quant"]

# TrainingArguments parameters
output_dir = config["training"]["output_dir"]
num_train_epochs = config["training"]["num_train_epochs"]
fp16 = config["training"]["fp16"]
bf16 = config["training"]["bf16"]
per_device_train_batch_size = config["training"]["per_device_train_batch_size"]
per_device_eval_batch_size = config["training"]["per_device_eval_batch_size"]
gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
gradient_checkpointing = config["training"]["gradient_checkpointing"]
max_grad_norm = config["training"]["max_grad_norm"]
learning_rate = float(config["training"]["learning_rate"])
weight_decay = config["training"]["weight_decay"]
optim = config["training"]["optim"]
lr_scheduler_type = config["training"]["lr_scheduler_type"]
max_steps = config["training"]["max_steps"]
warmup_ratio = config["training"]["warmup_ratio"]
group_by_length = config["training"]["group_by_length"]
save_steps = config["training"]["save_steps"]
logging_steps = config["training"]["logging_steps"]

# SFT parameters
max_seq_length = config["sft"]["max_seq_length"]
packing = config["sft"]["packing"]
device_map = config["model"]["device_map"]

# --- Implementation ---

def formatting_prompts_func(example):
    output_texts = []
    
    # Check if the input is a batch (list) or a single example (string/value)
    if isinstance(example['Cover Letter'], list):
        for i in range(len(example['Cover Letter'])):
            text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a cover letter for the position of {example['Job Title'][i]} at {example['Hiring Company'][i]}.

### Input:
Applicant: {example['Applicant Name'][i]}
Skills: {example['Skillsets'][i]}
Experience: {example['Current Working Experience'][i]}
Qualifications: {example['Qualifications'][i]}

### Response:
{example['Cover Letter'][i]}"""
            output_texts.append(text)
    else:
        # Single example case
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a cover letter for the position of {example['Job Title']} at {example['Hiring Company']}.

### Input:
Applicant: {example['Applicant Name']}
Skills: {example['Skillsets']}
Experience: {example['Current Working Experience']}
Qualifications: {example['Qualifications']}

### Response:
{example['Cover Letter']}"""
        return text
        
    return output_texts

def main():
    # 1. Load Dataset
    # print(f"Loading dataset: {dataset_name}")
    # dataset = load_dataset(dataset_name, split="train")
    # 1. Load Dataset (local train/val json/jsonl)
    # train_file = config["data"]["train_file"]
    # val_file = config["data"]["val_file"]
    # dataset_format = config["data"].get("dataset_format", "json")

    # print(f"Loading local datasets:\n  train: {train_file}\n  val:   {val_file}")

    # ds = load_dataset(
    #     dataset_format,
    #     data_files={"train": train_file, "validation": val_file},
    # )

    # train_dataset = ds["train"]
    # eval_dataset = ds["validation"]

    # 1. Load Dataset (local train/val jsonl under ./data)
    base_dir = os.path.dirname(os.path.abspath(__file__))  # folder where finetune_sft.py lives

    train_file = os.path.join(base_dir, "data", "train.jsonl")
    val_file   = os.path.join(base_dir, "data", "val.jsonl")  # rename your val file to this

    dataset_format = "json"

    print(f"Loading local datasets:\n  train: {train_file}\n  val:   {val_file}")

    ds = load_dataset(
        dataset_format,
        data_files={"train": train_file, "validation": val_file},
    )

    train_dataset = ds["train"]
    eval_dataset  = ds["validation"]



    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # 3. Load Base Model with Quantization
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

    print(f"Loading model: {model_name}")
    
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # --- Verification ---
    print("="*50)
    print(f"Model is loaded on device: {model.device}")
    print("="*50)

    # 4. Load LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Set Training Arguments
    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard",
        packing=packing,
    )

    # 6. Initialize SFTTrainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_arguments,
        formatting_func=formatting_prompts_func,
    )

    # 7. Train
    print("Starting training...")
    trainer.train()

    # 8. Save Model
    print(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
