def table_to_sentences(table, page, source):
    """
    Convert a pdfplumber table (list of lists) into a natural language
    string where each data row becomes a sentence of header=value pairs.

    Works on any table regardless of column names or document type.
    Assumes row 0 is the header row.
    """
    if not table or len(table) < 2:
        return None

    # Use column index as fallback if a header cell is empty
    headers = [
        cell.strip() if cell and cell.strip() else f"col{i}"
        for i, cell in enumerate(table[0])
    ]

    sentences = []
    for i, row in enumerate(table[1:], 1):
        # Skip rows that are entirely empty
        if not any(cell for cell in row):
            continue

        pairs = ", ".join(
            f"{h}={v.strip()}"
            for h, v in zip(headers, row)
            if v and v.strip()
        )

        if pairs:
            sentences.append(f"Table on page {page}, row {i}: {pairs}.")

    if not sentences:
        return None

    return "\n".join(sentences)


def process_tables(tables, source):
    """
    Convert all tables extracted from a PDF into chunk dicts
    ready for embedding and storage in Weaviate.

    Each table becomes one chunk containing all its row sentences.
    """
    chunks = []
    for page_num, table in tables:
        text = table_to_sentences(table, page_num, source)
        if text:
            chunks.append({
                "text": text,
                "page": page_num,
                "source": source
            })
    return chunks