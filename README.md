# PDF Q&A with RAG

A local RAG (Retrieval-Augmented Generation) pipeline for question-answering over PDF documents. Built as a hands-on project for learning RAG architecture from the ground up.

## How it works

1. **Ingest** - PDF is parsed, chunked, embedded, and stored in Weaviate
2. **Query** - Your question is embedded and matched against stored chunks via hybrid search (BM25 + vector)
3. **Answer** - Retrieved chunks are passed to an LLM with your question to generate a grounded answer

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.10 |
| Vector DB | Weaviate (local via Docker) |
| LLM (local) | Ollama - `llama3.2:3b` |
| LLM (cloud) | Groq - `llama-3.3-70b-versatile` |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| PDF Parsing | pdfplumber |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |

## Setup

**Prerequisites:** Docker, Ollama, Python 3.10+

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Start Weaviate
docker start <your-weaviate-container>

# 3. Pull the local LLM
ollama pull llama3.2

# 4. (Optional) Add Groq API key for cloud LLM
echo "GROQ_API_KEY=your_key_here" > .env
```

## Usage

```bash
# Ingest a PDF
python ingest.py assets/your_document.pdf

# Single question (local, private)
python query.py "your question here"

# Single question via Groq (faster, better quality)
python query.py "your question here" --no-confidential

# Interactive chat
python chat.py

# Interactive chat with Groq
python chat.py --no-confidential

# Show retrieval diagnostics
python query.py "your question" --verbose

# Inspect the vector database
python inspect_db.py
```

## Key Features

- **Fully local by default** - PDF, embeddings, and LLM inference all stay on your machine
- **Hybrid search** - combines BM25 keyword matching with vector similarity for better retrieval
- **Conversation memory** - chat mode retains the last 3 turns of context
- **Table extraction** - tables are converted to natural language chunks and embedded alongside text
- **Metadata injection** - document title and authors are injected into every prompt
- **Cloud fallback** - swap to Groq at runtime with `--no-confidential`, no config changes needed

---

## How the project evolved

This wasn't built in one shot. It grew incrementally, with each phase fixing a real problem discovered in the previous one.

### Phase 1-2 - Getting a basic pipeline working
The first version was minimal: load a PDF with `pypdf`, split it into chunks, embed with `all-MiniLM-L6-v2`, store in Weaviate, retrieve the top chunks, and send them to a local Ollama model. No tuning, no frills - just enough to get an answer out.

### Phase 3–4 - Tuning retrieval quality
Early answers were inconsistent. After testing, the weak point was retrieval: chunks were too small, overlap was too low, and not enough context was being passed to the LLM. Chunk size was bumped from 512 → 768 tokens, overlap from 64 → 128, and TOP_K from 8 → 10. Re-ingesting with these settings reduced total chunk count from 132 down to 90 (larger, less fragmented chunks) and improved answer quality noticeably.

### Phase 5 - Testing and guardrails
Systematic testing revealed a recurring pattern: retrieval was the weak link, not the model. When the right chunk was surfaced, even the small 3b model answered correctly. When the wrong chunks were retrieved, no LLM could save it. A "I don't know" instruction was added to the prompt to stop the model from hallucinating when context was insufficient. A certainty threshold was introduced to filter low-confidence retrievals (later replaced by hybrid search).

### Phase 6 - Conversation memory
The pipeline started as single-shot only - each question was independent. A multi-turn `chat.py` loop was added, with the last 3 Q&A pairs injected into every prompt. A fixed history window was chosen over more complex approaches (token budgets, summarisation) to keep it predictable within the 4096-token context limit. Testing confirmed the model could correctly handle follow-up questions like "can you elaborate?" using prior context.

### Phase 6.5 - Hybrid search (BM25 + vector)
A specific failure exposed the limits of pure vector search: follow-up questions like "what is LS and DS?" had almost no semantic overlap with document content, so the vector search returned near-useless chunks. Switching to Weaviate's hybrid search (combining BM25 keyword matching with vector similarity via Reciprocal Rank Fusion) fixed this class of failure entirely. The top score for "what is LS and DS?" jumped from fallback territory to 0.90. No re-ingestion was required - BM25 indexes are built automatically on stored text.

### Phase 7 - CLI and source attribution
Small but useful: answers now print the source page numbers they drew from (`Sources: Page 1, Page 3, Page 8`), and both `query.py` and `chat.py` gained a `--verbose` flag that shows retrieved chunk scores and counts for debugging retrieval quality without changing the default output.

### Phase 8 - Groq integration and the confidential flag
The local Ollama model (`llama3.2:3b`) was benchmarked head-to-head against Groq's `llama-3.3-70b-versatile`. Groq won clearly on synthesis, multi-step reasoning, and counterfactual questions. Ollama was competitive on direct single-chunk lookups. Rather than switching permanently, a `--no-confidential` CLI flag was added: default behaviour stays fully local (nothing leaves the machine), and Groq is opt-in for non-sensitive documents. Since both use OpenAI-compatible APIs, the code change was a single branch in `llm.py`.

### Phase 9 - Metadata extraction and table handling
Two persistent failures from Phase 8 were fixed here. First, questions about authorship ("who conducted this study?") failed for both models because that information lived in PDF metadata, not in the text chunks. The loader was extended to extract title and authors from standard PDF spec fields (with a heuristic fallback for academic papers), save them to a JSON file at ingest time, and inject them as a fixed prefix above retrieved context in every prompt. This fixed authorship questions completely. Second, table content was being mixed into page text, degrading chunk quality. Tables are now extracted separately by pdfplumber and converted to natural language sentences ("Table on page 6, row 2: Model=Faster R-CNN, AP=57.2, ...") before embedding. This approach is fully generic - no hardcoded column names - so it works across any document.

---

## Possible Future Improvements

- **Better table extraction** - `camelot` (lattice mode) to handle PDFs where pdfplumber garbles table text due to character spacing encoding issues
- **Query rewriting** - use the LLM to rewrite vague follow-up questions into retrieval-friendly form before embedding
- **Re-ranking** - pass hybrid search results through a cross-encoder re-ranker for better final ordering
- **Parent-child chunking** - retrieve small chunks for precision but inject surrounding context into the prompt
- **Multi-PDF support** - scope retrieval per document or across documents
- **Streaming responses** - stream tokens to the terminal as generated instead of waiting for the full response
- **Evaluation harness** - fixed Q&A pairs scored automatically so changes can be measured, not just eyeballed