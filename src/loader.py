import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdfplumber
import re


def _table_to_markdown(table):
    """Convert a pdfplumber table (list of lists) to a markdown string."""
    if not table or not table[0]:
        return ""

    cleaned = [
        [cell if cell is not None else "" for cell in row]
        for row in table
    ]

    header = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]

    return "\n".join([header, separator] + rows)


def _extract_metadata_from_spec(pdf_metadata):
    """
    Extract title and author from standard PDF spec metadata fields.
    These are populated by most PDF exporters (LaTeX, Word, Adobe).
    Returns a dict with None values for any fields not found.
    """
    metadata = {
        "title": None,
        "authors": None,
        "institution": None,  # not a PDF spec field — always None from this path
    }

    if not pdf_metadata:
        return metadata

    # PDF spec field names vary slightly by exporter — check common variants
    title = pdf_metadata.get("Title") or pdf_metadata.get("title")
    author = pdf_metadata.get("Author") or pdf_metadata.get("author")

    if title and title.strip():
        metadata["title"] = title.strip()
    if author and author.strip():
        metadata["authors"] = author.strip()

    return metadata


def _extract_metadata_from_text(first_page_text):
    """
    Heuristic fallback: extract title, authors, institution from
    first page text. Works reasonably well for academic PDFs.
    Not guaranteed for all document types.
    """
    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

    metadata = {
        "title": None,
        "authors": None,
        "institution": None,
    }

    for line in lines:
        if len(line) > 20:
            metadata["title"] = line
            break

    for i, line in enumerate(lines):
        if metadata["title"] and line == metadata["title"]:
            continue
        if re.search(r'\d', line) and len(line) > 10 and "Abstract" not in line:
            metadata["authors"] = re.sub(r'\d', '', line).strip(", ")
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if "Abstract" not in next_line and len(next_line) > 5:
                    metadata["institution"] = next_line
            break

    return metadata


def _merge_metadata(spec, heuristic):
    """
    Merge spec and heuristic metadata.
    Spec fields take priority; heuristic fills in any gaps.
    """
    return {
        "title": spec["title"] or heuristic["title"],
        "authors": spec["authors"] or heuristic["authors"],
        "institution": spec["institution"] or heuristic["institution"],
    }


def load_pdf(path):
    pages = []
    metadata = {}

    with pdfplumber.open(path) as pdf:
        # --- Metadata extraction ---
        spec_metadata = _extract_metadata_from_spec(pdf.metadata)

        # Only run heuristic if spec is incomplete
        if not spec_metadata["title"] or not spec_metadata["authors"]:
            first_page_text = pdf.pages[0].extract_text() or ""
            heuristic_metadata = _extract_metadata_from_text(first_page_text)
            metadata = _merge_metadata(spec_metadata, heuristic_metadata)
        else:
            metadata = spec_metadata

        # --- Page extraction ---
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            # Extract tables separately — do NOT append to page text
            page_tables = page.extract_tables()
            for table in page_tables:
                if table and len(table) >= 2:
                    tables.append((i + 1, table))

            if text.strip():
                pages.append({
                    "page": i + 1,
                    "text": text,
                    "source": path
                })

    return pages, tables, metadata