"""Text from a file a tester attaches to the LLM bot's prompt (an SOP, a policy PDF, an FAQ sheet).

``extract(name, data) -> {"name", "kind", "chars", "text", "truncated", "note"}``. Plain text formats are decoded;
PDF pages are read with pypdf (pdfplumber as a fallback); Word files with python-docx (paragraphs and tables);
spreadsheets with openpyxl (one line per row). Anything else is refused with the formats that work. The text is
capped (``MAX_CHARS``) so a large manual cannot crowd out the procedure in the prompt; the cap is reported.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

MAX_BYTES = 8 * 1024 * 1024
MAX_CHARS = 120_000
TEXT = {".txt", ".md", ".markdown", ".json", ".yaml", ".yml", ".csv", ".tsv", ".html", ".htm", ".xml", ".rst", ".log"}
FORMATS = "text (.txt .md .json .yaml .csv .html), PDF, Word (.docx) or Excel (.xlsx)"


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]
        if any(x.strip() for x in pages):
            return "\n\n".join(f"[page {i + 1}]\n{x.strip()}" for i, x in enumerate(pages) if x.strip())
    except ImportError:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(f"[page {i + 1}]\n{x.strip()}" for i, x in enumerate(pages) if x.strip())
    except ImportError as e:
        raise ValueError("reading PDF needs pypdf or pdfplumber (pip install pypdf)") from e


def _docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as e:
        raise ValueError("reading Word files needs python-docx (pip install python-docx)") from e
    d = docx.Document(io.BytesIO(data))
    out = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            out.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(out)


def _xlsx(data: bytes) -> str:
    try:
        import openpyxl
    except ImportError as e:
        raise ValueError("reading Excel files needs openpyxl (pip install openpyxl)") from e
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"[sheet {ws.title}]")
        for row in ws.iter_rows(values_only=True):
            if any(c is not None and str(c).strip() for c in row):
                out.append(" | ".join("" if c is None else str(c) for c in row))
    return "\n".join(out)


def extract(name: str, data: bytes) -> dict[str, Any]:
    name = Path(name or "attachment").name
    if len(data) > MAX_BYTES:
        raise ValueError(f"{name} is {len(data) / 1e6:.1f} MB; attach a file under {MAX_BYTES // (1024 * 1024)} MB")
    ext = Path(name).suffix.lower()
    if ext in TEXT:
        text, kind = _decode(data), ext.lstrip(".")
        if ext == ".json":                          # pretty, so the model reads keys and values
            try:
                text = json.dumps(json.loads(text), indent=1, ensure_ascii=False)
            except ValueError:
                pass
        elif ext in (".csv", ".tsv"):
            rows = list(csv.reader(io.StringIO(text), delimiter="\t" if ext == ".tsv" else ","))
            text = "\n".join(" | ".join(r) for r in rows)
    elif ext == ".pdf":
        text, kind = _pdf(data), "pdf"
    elif ext == ".docx":
        text, kind = _docx(data), "docx"
    elif ext in (".xlsx", ".xlsm"):
        text, kind = _xlsx(data), "xlsx"
    else:
        raise ValueError(f"{name}: this format cannot be read as text; attach {FORMATS}")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise ValueError(f"{name} has no readable text (a scanned PDF needs OCR first)")
    cut = len(text) > MAX_CHARS
    return {"name": name, "kind": kind, "chars": min(len(text), MAX_CHARS), "text": text[:MAX_CHARS],
            "truncated": cut, "note": f"only the first {MAX_CHARS:,} of {len(text):,} characters are used" if cut else ""}
