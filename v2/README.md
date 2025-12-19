# coverletter_demo_v4_fixed

## Install
pip install -r requirements.txt
pip install -r requirements_local.txt

## One-click training (VS Code)
Open `run_train.py` and click ▶ Run.

## One-click demo (VS Code)
Open `run_demo.py` and click ▶ Run.

## Where is the trained model?
The LoRA adapter is saved to `outputs/all_data_lora/` by default.


## Fix2 notes
- If you previously saw `loss does not require grad`, this version explicitly applies `get_peft_model(...)` and enables input grads.
- If you are on Windows + Python 3.13 Store Python, training may still be unstable; WSL2 + Python 3.12 is recommended.
