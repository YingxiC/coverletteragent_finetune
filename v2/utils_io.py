\
from __future__ import annotations
from io import BytesIO
import pdfplumber
from docx import Document

def extract_text_from_upload(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.read()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()

    if name.endswith(".pdf"):
        out = []
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        return "\n".join(out).strip()

    if name.endswith(".docx"):
        doc = Document(BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs]).strip()

    raise ValueError("Unsupported file type. Use PDF/DOCX/TXT.")
