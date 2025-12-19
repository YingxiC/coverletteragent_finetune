\
import os
import streamlit as st

from llm_utils import generate_cover_letter_openai, generate_cover_letter_local, load_local_model_cached
from utils_io import extract_text_from_upload

st.set_page_config(page_title="Cover Letter Generator (v2)", layout="wide")
st.title("Resume → Cover Letter (Demo v2)")

with st.sidebar:
    st.header("Backend")
    backend = st.selectbox("Choose backend", ["Local LoRA (GPU)", "OpenAI API"], index=0)
    tone = st.selectbox("Tone", ["Professional", "Concise", "Startup", "Academic"], index=0)
    length_words = st.slider("Target length (words)", 200, 450, 320)
    temperature = st.slider("Temperature", 0.0, 1.2, 0.7, 0.1)

    if backend == "Local LoRA (GPU)":
        st.subheader("Local model settings")
        base_model = st.text_input("Base model", value=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
        lora_dir = st.text_input("LoRA adapter dir", value=os.getenv("LORA_DIR", "outputs/all_data_lora"))
        max_new_tokens = st.slider("Max new tokens", 200, 800, 450, 10)
        if st.button("Load/Reload local model"):
            st.cache_resource.clear()
            st.success("Cache cleared. Model will reload on next generation.")
    else:
        st.subheader("OpenAI settings")
        model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_upload = st.file_uploader("Upload resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    resume_text_manual = st.text_area("Or paste resume text", height=220)

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste job description", height=300)

if st.button("Generate Cover Letter"):
    resume_text = ""
    if resume_text_manual.strip():
        resume_text = resume_text_manual.strip()
    elif resume_upload is not None:
        resume_text = extract_text_from_upload(resume_upload)

    if not resume_text.strip():
        st.error("Please upload or paste a resume.")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    with st.spinner("Generating..."):
        if backend == "OpenAI API":
            if not api_key:
                st.error("Missing OPENAI_API_KEY.")
                st.stop()
            letter = generate_cover_letter_openai(
                api_key=api_key,
                model=model,
                resume=resume_text,
                job_description=jd_text,
                tone=tone,
                length_words=length_words,
                temperature=temperature,
            )
        else:
            tok, mdl = load_local_model_cached(base_model, lora_dir)
            letter = generate_cover_letter_local(
                tokenizer=tok,
                model=mdl,
                resume=resume_text,
                job_description=jd_text,
                tone=tone,
                length_words=length_words,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

    st.subheader("Cover Letter")
    st.write(letter)
    st.download_button("Download TXT", data=letter.encode("utf-8"), file_name="cover_letter.txt", mime="text/plain")
