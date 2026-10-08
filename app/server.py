"""
Deep Research Agent - web backend (FastAPI + Server-Sent Events).

Run from the project root:   python app/server.py
Then open:                   http://localhost:8000

How progress streaming works: the LangGraph nodes `print()` their status
messages. While the graph runs in a worker thread, stdout is redirected to a
writer that pushes each printed line onto a queue; the SSE response drains that
queue and forwards every line to the browser as an event.
"""
import contextlib
import io
import json
import queue
import re
import sys
import threading
from pathlib import Path

# Make `app`, `agents`, `tools`, `memory` importable however the server is started.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")  # GROQ_API_KEY

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import graph as graph_module

STATIC_DIR = Path(__file__).resolve().parent / "static"
api = FastAPI(title="Deep Research Agent")

# Only one run at a time: redirecting stdout is process-wide.
run_lock = threading.Lock()


def get_compiled_graph():
    """Find the compiled LangGraph object in app/graph.py (adjust names if yours differs)."""
    for name in ("app", "graph", "research_graph", "workflow"):
        obj = getattr(graph_module, name, None)
        if obj is not None and hasattr(obj, "invoke"):
            return obj
    for name in ("build_graph", "create_graph", "get_graph"):
        factory = getattr(graph_module, name, None)
        if callable(factory):
            return factory()
    raise RuntimeError(
        "No compiled graph found in app/graph.py. Expose it as `graph`, `app` "
        "or `research_graph`, or edit get_compiled_graph() in server.py."
    )


class ResearchRequest(BaseModel):
    topic: str


class QueueWriter(io.TextIOBase):
    """File-like object: every complete line written is pushed onto a queue."""

    def __init__(self, q: queue.Queue):
        self.q, self._buf = q, ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.q.put(("log", line.strip()))
        return len(text)

    def flush(self) -> None:
        pass


def classify(line: str) -> str:
    """Map a printed log line to a UI stage."""
    low = line.lower()
    if "critic" in low or "gap" in low or "⚖" in line:
        return "critique"
    if "synth" in low or "📝" in line:
        return "synthesize"
    if "read" in low or "analy" in low or "📖" in line:
        return "read"
    if "research" in low or "search" in low or "🔍" in line:
        return "research"
    return "info"


def run_graph(topic: str, q: queue.Queue) -> None:
    """Worker thread: run the agent with stdout captured, then post the result."""
    try:
        with run_lock, contextlib.redirect_stdout(QueueWriter(q)):
            state = {
                "topic": topic, "raw_papers": [], "analyzed_papers": [],
                "synthesis": "", "critique": "", "gaps": [], "iteration": 0,
            }
            result = get_compiled_graph().invoke(state)
        q.put(("result", result.get("synthesis", "")))
    except Exception as exc:  # surface any crash to the UI
        q.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        q.put(("end", None))


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@api.post("/api/research")
def research(req: ResearchRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required.")

    q: queue.Queue = queue.Queue()
    threading.Thread(target=run_graph, args=(topic, q), daemon=True).start()

    def event_stream():
        while True:
            try:
                kind, data = q.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"  # keeps proxies from closing the stream
                continue
            if kind == "log":
                message = re.sub(r"^\W*\[Graph\]\s*", "", data)  # drop "🔍 [Graph]" prefix
                yield sse({"type": "progress", "stage": classify(data), "message": message})
            elif kind == "result":
                yield sse({"type": "result", "synthesis": data})
            elif kind == "error":
                yield sse({"type": "error", "message": data})
            elif kind == "end":
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=8000)
