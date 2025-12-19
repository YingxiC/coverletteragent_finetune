# llm_utils.py (fixed)
from __future__ import annotations

import os
import re
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


SYSTEM_RULES = (
    "You write a tailored cover letter using ONLY the resume content. "
    "Do NOT invent degrees, employers, titles, dates, or metrics. "
    "If something is missing, write more generally and truthfully."
)

DEFAULT_TASK = (
    "Write a professional cover letter (250–400 words) tailored to the job. "
    "Use specific evidence from the resume. Do not fabricate details. "
    "Output ONLY the cover letter text."
)


# ---------- Streamlit cache if available ----------
def _cache_resource(fn):
    try:
        import streamlit as st  # type: ignore
        return st.cache_resource(fn)
    except Exception:
        from functools import lru_cache
        return lru_cache(maxsize=2)(fn)


# ---------- Cleaning / postprocess ----------
def _clean_input_text(s: str) -> str:
    s = (s or "").strip()

    # Remove wrapper quotes if user pasted a quoted JSON string
    s = re.sub(r'^\s*"(.*)"\s*$', r"\1", s, flags=re.DOTALL)
    s = s.replace('\\"', '"')

    # Remove common accidental JSON prefixes that appeared in your outputs
    s = s.replace('resume":"', "")
    s = s.replace('job_description":"', "")
    s = s.replace('output":"', "")

    # Strip leftover JSON punctuation
    s = s.replace('",', "")
    s = s.replace('"}', "")
    return s.strip()


def _postprocess_generated(text: str) -> str:
    """
    Trim model drift if it starts generating additional chat turns/tasks.
    We do this AFTER decoding (no stopping criteria that might match the prompt).
    """
    if not text:
        return text

    # Cut off if it starts a new "turn" or other tasks
    cut_patterns = [
        r"\nHuman:\s", r"\nUSER:\s", r"\nSYSTEM:\s",
        r"\n\[USER\]", r"\n\[SYSTEM\]", r"\n\[ASSISTANT\]",
        r"<\|im_start\|>user", r"<\|im_start\|>system", r"<\|im_start\|>assistant",
    ]
    for pat in cut_patterns:
        m = re.search(pat, text)
        if m:
            text = text[: m.start()].strip()

    # Remove stray bracket tokens if the model leaks them
    text = text.replace("[/ASSISTANCE]", "").replace("[/ASSISTANT]", "").strip()

    return text.strip()


def _build_messages(resume: str, jd: str, tone: str, length_words: int) -> List[Dict[str, str]]:
    sys = f"{SYSTEM_RULES}\nTone: {tone}\nTarget length: {length_words} words."
    user = f"RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{jd}\n\n{DEFAULT_TASK}"
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def _encode_chat(tokenizer, messages: List[Dict[str, str]]):
    """
    For Qwen / chat models, prefer apply_chat_template with tokenize=True.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    # Fallback
    prompt = (
        f"SYSTEM:\n{messages[0]['content']}\n\n"
        f"USER:\n{messages[1]['content']}\n\nASSISTANT:\n"
    )
    return tokenizer(prompt, return_tensors="pt")["input_ids"]


# ---------- Local model loading ----------
@_cache_resource
def load_local_model_cached(
    base_model: str,
    lora_dir: Optional[str] = None,
    prefer_4bit: bool = True,
) -> Tuple[object, object]:
    """
    Loads base model and optionally attaches a LoRA adapter directory.
    Uses Streamlit cache so it loads once.
    """

    use_cuda = torch.cuda.is_available()

    quant_cfg = None
    if use_cuda and prefer_4bit:
        try:
            import bitsandbytes  # noqa: F401
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        except Exception:
            quant_cfg = None  # fall back to normal load

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if use_cuda else None,
        quantization_config=quant_cfg,
        torch_dtype=(torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16),
        trust_remote_code=True,
    )

    # Attach LoRA adapter if present
    if lora_dir and os.path.isdir(lora_dir) and PeftModel is not None:
        model = PeftModel.from_pretrained(model, lora_dir)

    model.eval()
    return tokenizer, model


# ---------- Local generation ----------
@torch.inference_mode()
def generate_cover_letter_local(
    tokenizer,
    model,
    resume: str,
    job_description: str,
    tone: str = "Professional",
    length_words: int = 320,
    temperature: float = 0.7,
    max_new_tokens: int = 350,
) -> str:
    resume = _clean_input_text(resume)
    job_description = _clean_input_text(job_description)

    messages = _build_messages(resume, job_description, tone, length_words)
    input_ids = _encode_chat(tokenizer, messages).to(model.device)

    # Generate (NO stopping criteria — we post-trim instead)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        do_sample=(float(temperature) > 0),
        temperature=max(1e-5, float(temperature)) if float(temperature) > 0 else 1.0,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.05,
    )

    # ✅ decode only new tokens
    gen_ids = out[0][input_ids.shape[-1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ✅ trim drift / extra tasks
    return _postprocess_generated(text)


# ---------- OpenAI (optional path) ----------
def generate_cover_letter_openai(
    api_key: str,
    model: str,
    resume: str,
    job_description: str,
    tone: str = "Professional",
    length_words: int = 320,
    temperature: float = 0.7,
) -> str:
    resume = _clean_input_text(resume)
    job_description = _clean_input_text(job_description)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    messages = _build_messages(resume, job_description, tone, length_words)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
    )
    return resp.choices[0].message.content.strip()
