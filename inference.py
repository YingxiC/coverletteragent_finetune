import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# 1. Configuration
base_model_name = "Qwen/Qwen3-0.6B-Base"
adapter_path = "./results"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

# 2. Load Base Model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=None,
    trust_remote_code=True
)

# 3. Load Tokenizer
print("Loading tokenizer...")
# Load tokenizer from the fine-tuned checkpoint to ensure we use the same special tokens
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 4. Load Adapter (Fine-tuned LoRA weights)
print("Loading adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.to(device)

# 5. Prepare Input
# We use the same format as used in training
job_title = "Software Engineer"
hiring_company = "TechCorp Inc."
applicant_name = "John Doe"
skills = "Python, JavaScript, React, AWS, Docker"
experience = "3 years as a Full Stack Developer at StartupXYZ"
qualifications = "B.S. in Computer Science"

prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a cover letter for the position of {job_title} at {hiring_company}.

### Input:
Applicant: {applicant_name}
Skills: {skills}
Experience: {experience}
Qualifications: {qualifications}

### Response:
"""

# 6. Generate
print("Generating response...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)
streamer = TextStreamer(tokenizer, skip_prompt=True)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

