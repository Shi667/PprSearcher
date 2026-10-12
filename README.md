# 🔬 Deep Research Agent

> An autonomous, multi-agent research system that generates comprehensive, academic **State-of-the-Art** reviews by downloading full PDFs from arXiv, analyzing them, and self-correcting through a critic loop.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-8B5CF6?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036)](https://groq.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Overview

Most LLM-powered research tools are **linear pipelines**: query → retrieve → summarize. They fail when the initial retrieval is poor, they only read abstracts, and they have no way to self-correct.

**Deep Research Agent** is different. It is a true **agentic system** that:

1. Downloads and reads the **full text** of research papers (not just abstracts).
2. Evaluates its own synthesis via a **Critic agent**.
3. Identifies knowledge gaps and **loops back** to fetch targeted papers.
4. **Curates** the final output to guarantee exactly the number of high-quality papers you requested.
5. Remembers what it has shown you via **persistent memory**, ensuring no duplicate recommendations across sessions.
