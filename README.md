# Deep Research Agent

An autonomous multi-agent system that writes academic **state-of-the-art reviews**. It searches arXiv, reads the full PDFs, critiques its own draft, and loops back to fill any gaps it finds.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-8B5CF6?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036)](https://groq.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Most LLM research tools are linear pipelines (query, retrieve, summarize). They fail when the first retrieval is poor, they usually read only abstracts, and they cannot correct themselves.

Deep Research Agent is a cyclic graph instead:

1. It downloads and reads the **full text** of each paper, not just the abstract.
2. A **Critic** evaluates the draft synthesis and lists knowledge gaps.
3. Each gap becomes a targeted arXiv query, and the loop runs again (up to 2 refinement iterations).
4. A **Curator** keeps exactly the number of papers you asked for.
5. A **persistent memory** of papers already shown prevents duplicates across sessions.

```text
START ─▶ Research ─▶ Read (full PDF) ─▶ Synthesize ─▶ Critic
            ▲                                          │
            └──── gaps found and iteration < 2 ────────┤
                                                       │
                      no gaps, or max iterations       ▼
                                  END ◀─────────────  Curate

Persistent memory filters previously seen papers out of every Research step.
```

## Features

- **Agentic loop:** the Critic finds gaps, the Researcher targets them, and the synthesis improves.
- **Full-text reading:** PyMuPDF extracts methodology, results and limitations from the PDF, so the output is not limited to what the abstract says.
- **Structured analysis:** each paper is parsed into a fixed JSON schema (objective, methodology, framework, datasets, metrics, results, limitations, and more).
- **Curation:** if refinement loops fetch extra papers, only the best _N_ make it into the report.
- **Persistent memory:** seen arXiv IDs are stored in a local JSON file and survive restarts.
- **Live web UI:** FastAPI streams progress over Server-Sent Events, with a paper count (1–20) and dark-mode support.
- **PDF export:** the report is rendered server-side into a paginated A4 PDF with `xhtml2pdf`, with no LaTeX or wkhtmltopdf needed.

## Architecture

| Component       | Role                                                             | Output                       |
| :-------------- | :--------------------------------------------------------------- | :--------------------------- |
| **Researcher**  | Queries arXiv, reformulates queries from gaps, filters by memory | Raw paper metadata           |
| **Reader**      | Downloads the PDF and extracts structured data                   | Paper plus JSON summary      |
| **Synthesizer** | Drafts the state of the art from all analyzed papers             | Markdown draft               |
| **Critic**      | Evaluates the draft and identifies gaps                          | Verdict and targeted queries |
| **Curator**     | Ranks all analyzed papers and assembles the final report         | Final Markdown               |

| Layer         | Technology                                                                                            |
| :------------ | :---------------------------------------------------------------------------------------------------- |
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` with conditional edges            |
| LLM           | [Groq](https://groq.com/) via `langchain_groq` (model set in the code; default `openai/gpt-oss-120b`) |
| Data source   | [arXiv API](https://info.arxiv.org/help/api/index.html) (HTTP and XML)                                |
| PDF           | `PyMuPDF` for extraction, `xhtml2pdf` for export                                                      |
| Backend       | [FastAPI](https://fastapi.tiangolo.com/) with SSE                                                     |
| Frontend      | Vanilla HTML/CSS/JS, `marked.js`, `DOMPurify`                                                         |

## Project structure

```text
PprSearcher/
├── .env                      # GROQ_API_KEY
├── requirements.txt
├── README.md
├── app/
│   ├── graph.py              # LangGraph orchestrator (synthesis and curation nodes)
│   ├── server.py             # FastAPI backend, SSE streaming, PDF endpoint
│   └── static/index.html     # Web UI
├── agents/
│   ├── researcher.py         # arXiv search and query reformulation
│   ├── reader.py             # PDF download and structured extraction
│   └── critic.py             # Synthesis evaluation and gap detection
├── tools/
│   ├── arxiv_tool.py         # arXiv API wrapper
│   ├── pdf_reader.py         # PyMuPDF full-text extractor
│   └── pdf_generator.py      # Markdown to PDF renderer
└── memory/
    ├── state.py              # ResearchState TypedDict
    └── paper_memory.py       # Persistent JSON store of seen papers
```

## Getting started

**Prerequisites:** Python 3.10+ and a free [Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/Shi667/PprSearcher.git
cd PprSearcher

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file at the project root:

```env
GROQ_API_KEY=your_key_here
```

Start the server from the project root:

```bash
python app/server.py
```

Open http://localhost:8000, enter a topic, choose how many papers you want, and watch the agent work. When it finishes you can copy the Markdown or download a PDF.

## Design decisions

**Why LangGraph instead of a chain?** A chain cannot revisit an earlier step. Conditional edges let the Critic send execution back to the Researcher whenever it finds gaps.

**Why read full PDFs?** Abstracts rarely contain the evaluation metrics, datasets or honest limitations. The Reader extracts the first 15 pages and truncates the text to about 12,000 characters to stay within context and rate limits.

**How are Groq's token-per-minute limits handled?** LLM calls have a hard `max_tokens` cap, and curation asks the model only for a short JSON list of indices. The final report is assembled with Python string formatting, which costs no LLM tokens.

**Why export PDFs on the server?** Client-side generators often break tables and pagination. Rendering on the backend gives consistent output.

**How does live progress work?** Graph nodes `print()` status lines. While the graph runs in a worker thread, stdout is redirected into a queue, and the SSE endpoint forwards each line to the browser.

## Roadmap

- [ ] Vector database (Chroma or Qdrant) for semantic search over previously analyzed papers
- [ ] Parallel gap research with LangGraph's `Send` API
- [ ] Citation graph visualization
- [ ] Multi-language reports

## License

MIT. See [LICENSE](LICENSE).
