# 🔬 Deep Research Agent

> An autonomous multi-agent system that generates comprehensive, academic **State-of-the-Art** reviews by searching arXiv, analyzing papers, and self-correcting through a critic loop.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-8B5CF6?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciLz4=)](https://groq.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Overview

Most LLM-powered research tools are **linear pipelines**: query → retrieve → summarize. They fail when the initial retrieval is poor and have no way to self-correct.

**Deep Research Agent** is different. It's a true **agentic system** where a Critic agent evaluates the synthesis, identifies gaps, and sends the Researcher back to find targeted papers — looping until the output meets academic standards.

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌──────────┐    ┌────────┐    ┌────────────┐             │
│   │ Research │───▶│  Read  │───▶│ Synthesize │             │
│   └──────────┘    └────────┘    └────────────┘             │
│        ▲                                │                   │
│        │          ┌──────────┐          │                   │
│        └──────────│  Critic  │◀─────────┘                   │
│   (if gaps found) └──────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Features

- 🤖 **Agentic loop** — Critic identifies gaps → Researcher targets them → Synthesis improves
- 📚 **arXiv integration** — Queries the official API and parses Atom XML
- 🧠 **Structured analysis** — Each paper is parsed into 10 academic fields (objective, method, results, limitations, etc.)
- 🔄 **Self-correction** — Up to 2 refinement iterations based on critique
- 🌐 **Live web UI** — FastAPI backend with Server-Sent Events streaming progress in real-time
- 📄 **PDF export** — Download the final synthesis as a formatted PDF
- 🎨 **Minimalist design** — Clean, academic aesthetic with elegant typography

---

## 🏗️ Architecture

### Agent Roles

| Agent           | Role                           | Input               | Output                                      |
| --------------- | ------------------------------ | ------------------- | ------------------------------------------- |
| **Researcher**  | Queries arXiv, can reformulate | Topic or gap query  | Raw papers (title, authors, abstract, link) |
| **Reader**      | Extracts structured summary    | Raw paper dict      | Paper + 10-field JSON summary               |
| **Synthesizer** | Writes state-of-the-art        | All analyzed papers | Markdown review                             |
| **Critic**      | Evaluates & identifies gaps    | Synthesis           | Verdict + list of gap queries               |

### Tech Stack

| Layer         | Technology                                                                                                        |
| ------------- | ----------------------------------------------------------------------------------------------------------------- |
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) — StateGraph with conditional edges                        |
| LLM           | [Groq](https://groq.com/) — `openai/gpt-oss-120b` via LangChain                                                   |
| Data source   | [arXiv API](https://info.arxiv.org/help/api/index.html) — HTTP + XML parsing                                      |
| Backend       | [FastAPI](https://fastapi.tiangolo.com/) + Server-Sent Events                                                     |
| Frontend      | Vanilla HTML/CSS/JS + [marked.js](https://marked.js.org/) + [html2pdf.js](https://ekoopgorjafeh.org/html2pdf.js/) |

---

## 📁 Project Structure

```
PprSearcher/
├── .env                        # GROQ_API_KEY
├── requirements.txt
├── README.md
│
├── app/
│   ├── __init__.py
│   ├── graph.py                # LangGraph orchestrator (the brain)
│   ├── server.py               # FastAPI backend + SSE streaming
│   └── static/
│       └── index.html          # Web UI
│
├── agents/
│   ├── researcher.py           # Tool-calling agent with arXiv
│   ├── reader.py               # Structured summary extraction
│   └── critic.py               # Synthesis evaluation + gap detection
│
├── tools/
│   └── arxiv_tool.py           # arXiv API wrapper (XML parser)
│
└── memory/
    └── state.py                # ResearchState TypedDict
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys) (free tier available)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PprSearcher.git
cd PprSearcher

# Create and activate a virtual environment (or use conda)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API key

Create a `.env` file at the project root:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
```

### 4. Launch the web interface

```bash
python app/server.py
```

Open your browser at **http://localhost:8000**, enter a research topic, and watch the agent work.

### 5. (Optional) Run the graph from the terminal

```bash
python app/graph.py
```

---

## 🎬 Example Output

Given the topic _"quantum machine learning benchmarking"_, the agent produces a multi-section academic review with:

- Thematic analysis with tables
- Citations to real arXiv papers
- Methodological standards
- Open challenges and future directions
- Full references section

The Critic typically triggers **1–2 refinement loops**, each targeting a specific gap (e.g., _"missing cross-platform benchmark suites"_), resulting in a richer, more comprehensive synthesis.

---

## 💡 Key Design Decisions

### Why LangGraph over a simple chain?

A linear chain can't self-correct. LangGraph's conditional edges let the Critic route execution back to Research when gaps are found — the defining feature of an **agentic** system.

### Why structured JSON from the Reader?

Raw abstracts are noisy. By forcing the Reader to output a 10-field JSON schema, the Synthesizer receives clean, comparable data — which dramatically improves the quality of the final review.

### Why stream progress via SSE?

The agent takes 30–90 seconds per run. Without live feedback, the user has no idea what's happening. SSE lets the UI show each step (🔍 Research → 📖 Read → 📝 Synthesize → ⚖️ Critique) as it happens.

---

## 🎤 Interview Talking Points

If you're using this project in interviews, here's what to emphasize:

1. **Agentic behavior** — The Critic loop is what separates this from a simple RAG pipeline. It demonstrates _autonomous decision-making_.
2. **State management** — `ResearchState` (a `TypedDict`) is the shared memory that flows between nodes. Each node reads and updates it.
3. **Tool calling** — The Researcher uses LangChain's `@tool` decorator + `bind_tools()` to let the LLM decide _when_ to call arXiv.
4. **Robust parsing** — The Reader has a fallback for malformed JSON, preventing graph crashes.
5. **Real-time UX** — SSE streaming + stdout redirection shows how to bridge Python backends with reactive frontends.

---

## 🔮 Future Improvements

- [ ] Full-text PDF parsing (download and analyze the paper, not just the abstract)
- [ ] Persistent memory with ChromaDB for cross-session knowledge
- [ ] Multi-agent parallel research (search multiple gaps concurrently)
- [ ] Citation graph visualization
- [ ] Export to LaTeX / BibTeX
- [ ] Authentication + saved research history

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built with 🧠 LangGraph · ⚡ Groq · 📚 arXiv</strong>
</p>
