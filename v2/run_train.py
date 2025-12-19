\
"""
run_train.py
One-click training launcher (no long terminal commands).

Open this file in VS Code and click ▶ Run.
"""

import sys

# ===== EDIT THESE DEFAULTS IF NEEDED =====
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
USE_AUTO_SPLIT = True
INPUT_FILE = "data/all_data.jsonl"
VAL_RATIO = "0.2"

# If USE_AUTO_SPLIT = False:
TRAIN_FILE = "data/all_data_train_80.jsonl"
VAL_FILE = "data/all_data_val_20.jsonl"

OUTPUT_DIR = "outputs/all_data_lora"
MAX_SEQ_LEN = "2048"
EPOCHS = "2"
BATCH_SIZE = "1"
GRAD_ACCUM = "16"
LR = "2e-4"
HEARTBEAT_SECS = "15"
# ========================================

def main():
    argv = [
        "train_lora.py",
        "--model_name", MODEL_NAME,
        "--output_dir", OUTPUT_DIR,
        "--max_seq_len", MAX_SEQ_LEN,
        "--epochs", EPOCHS,
        "--batch_size", BATCH_SIZE,
        "--grad_accum", GRAD_ACCUM,
        "--lr", LR,
        "--heartbeat_secs", HEARTBEAT_SECS,
    ]
    if USE_AUTO_SPLIT:
        argv += ["--inputs", INPUT_FILE, "--val_ratio", VAL_RATIO]
    else:
        argv += ["--train_inputs", TRAIN_FILE, "--val_inputs", VAL_FILE]

    sys.argv = argv
    import train_lora
    train_lora.main()

if __name__ == "__main__":
    main()
