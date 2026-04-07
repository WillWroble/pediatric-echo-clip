"""Extract structured + text fields from echo report PDFs.

Handles both legacy format (pre-2010) and modern format.
Outputs: study_id, demographics, and all text sections.

Usage:
    python extract_v2.py $SLURM_ARRAY_TASK_ID
"""

import pdfplumber
import pandas as pd
import re
import sys
from pathlib import Path

BASE = Path("/lab-share/Cardio-Mayourian-e2/Public")
MANIFEST = BASE / "Echo_Clip/manifest.txt"
OUTPUT_DIR = BASE / "Echo_Clip/echo_reports_chunks_v3"
CHUNK_SIZE = 1000

# All known section headers (longest first avoids partial matches)
SECTION_NAMES = [
    "patient history codes",
    "anatomical diagnoses",
    "report recipients",
    "study findings",
    "report summary",
    "authentication",
    "measurements",
    "study information",
    "billing",
    "history",
    "summary",
    "graphs",
]


def clean_section(text):
    """Strip trailing blank lines only."""
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def parse_demographics(header):
    """Extract structured fields from the report header."""

    def first_match(pattern, text, group=1):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(group).strip() if m else ""

    return {
        "mrn": first_match(r"MRN:[ \t]*(\d+)", header) or first_match(r"Record\s*#:[ \t]*(\d+)", header),
        "dob": first_match(r"(?:Born|DOB)\s*:[ \t]*(.+?)(?=\s{2,}|\n|Age)", header),
        "age": first_match(r"Age:[ \t]*(\S+(?:\s+\S+)?)", header),
        "gender": first_match(r"Gender:[ \t]*(\w+)", header),
        "weight_kg": first_match(r"Weight:[ \t]*([\d.]+)\s*kg", header),
        "height_cm": first_match(r"Height:[ \t]*([\d.]+)\s*cm", header),
        "bsa": first_match(r"BSA:[ \t]*([\d.]+)", header),
        "bmi": first_match(r"BMI:[ \t]*([\d.]+)", header),
    }


def split_sections(text):
    """Split full report text into (header, {section_name: content})."""
    # Collapse horizontal rules into newlines
    text = re.sub(r"_{10,}", "\n", text)

    # Build regex: match section name at line start, optional "(continued)"
    sorted_names = sorted(SECTION_NAMES, key=len, reverse=True)
    alt = "|".join(re.escape(s) for s in sorted_names)
    pattern = rf"(?:^|\n)[ \t]*({alt})[ \t]*(?:\(continued\))?[ \t]*\n"

    parts = re.split(pattern, text, flags=re.IGNORECASE)

    header = parts[0] if parts else ""
    sections = {}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].strip().lower()
        content = parts[i + 1].strip()
        # Merge "(continued)" sections
        sections[key] = sections.get(key, "") + ("\n" if key in sections else "") + content

    return header, sections


def extract_subfields(history_text):
    """Pull cardiac_history and reason_for_exam from the History section."""
    cardiac = ""
    reason = ""

    m = re.search(r"Cardiac history:[ \t]*(.+?)(?=\nReason for|\Z)", history_text, re.DOTALL | re.IGNORECASE)
    if m:
        cardiac = m.group(1).strip()

    m = re.search(r"Reason for (?:exam|test):[ \t]*(.+?)(?=\n[A-Z]|\n\n|\Z)", history_text, re.DOTALL | re.IGNORECASE)
    if m:
        reason = m.group(1).strip()

    return cardiac, reason


def process_pdf(pdf_path):
    try:
        raw = extract_text(pdf_path)
        header, sections = split_sections(raw)
        # Demographics may be in the header (old format) or "study information" section (new format)
        demo_text = header + "\n" + sections.get("study information", "")
        demos = parse_demographics(demo_text)

        # Cardiac history / reason: try History section first, fall back to header
        cardiac_history, reason = "", ""
        if "history" in sections:
            cardiac_history, reason = extract_subfields(sections["history"])
        if not cardiac_history:
            m = re.search(r"Diagnosis:[ \t]*(.+)", header, re.IGNORECASE)
            if m:
                cardiac_history = m.group(1).strip()
        if not reason:
            m = re.search(r"Reason for (?:test|exam):[ \t]*(.+)", header, re.IGNORECASE)
            if m:
                reason = m.group(1).strip()

        # Normalize section name variants and clean
        summary = clean_section(sections.get("summary", "") or sections.get("report summary", ""))
        findings = clean_section(sections.get("study findings", "") or sections.get("anatomical diagnoses", ""))
        measurements = clean_section(sections.get("measurements", ""))

        return {
            "file": pdf_path.name,
            "study_id": pdf_path.stem.split("_")[0],
            **demos,
            "cardiac_history": cardiac_history,
            "reason_for_exam": reason,
            "summary": summary,
            "study_findings": findings,
            "measurements": measurements,
            "history": clean_section(sections.get("history", "")),
        }

    except Exception as e:
        empty = {
            "file": pdf_path.name, "study_id": "",
            "dob": "", "age": "", "gender": "", "weight_kg": "",
            "height_cm": "", "bsa": "", "bmi": "",
            "cardiac_history": "", "reason_for_exam": "",
            "summary": "", "study_findings": "", "measurements": "",
            "history": "", "error": str(e),
        }
        return empty


def main():
    task_id = int(sys.argv[1])
    paths = MANIFEST.read_text().strip().splitlines()
    start = task_id * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, len(paths))
    chunk = [BASE / p.strip() for p in paths[start:end]]

    if not chunk:
        print(f"Task {task_id}: no files in range [{start}:{end}]")
        return

    results = [process_pdf(p) for p in chunk]
    df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"chunk_{task_id:04d}.csv", index=False)
    print(f"Task {task_id}: processed {len(results)} files")


if __name__ == "__main__":
    main()
