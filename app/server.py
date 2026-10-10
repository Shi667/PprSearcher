import contextlib
import io
import json
import queue
import re
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


from tools.pdf_generator import generate_pdf_from_markdown
from app import graph as graph_module


import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

STATIC_DIR = Path(__file__).resolve().parent / "static"
api = FastAPI(title="Deep Research Agent")

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
    num_papers: int = 5  


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


def run_graph(topic: str, num_papers: int, q: queue.Queue) -> None:
    try:
        with run_lock, contextlib.redirect_stdout(QueueWriter(q)):
            state = {
                "topic": topic,
                "num_papers": num_papers,  # NEW
                "raw_papers": [],
                "analyzed_papers": [],
                "synthesis": "",
                "critique": "",
                "gaps": [],
                "iteration": 0,
            }
            result = get_compiled_graph().invoke(state)
            q.put(("result", result.get("synthesis", "")))
    except Exception as exc:
        q.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        q.put(("end", None))


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@api.post("/api/research")
def research(req: ResearchRequest):
    topic = req.topic.strip()
    num_papers = req.num_papers
    
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required.")
    if num_papers < 1 or num_papers > 20:
        raise HTTPException(status_code=400, detail="num_papers must be between 1 and 20")
    
    q: queue.Queue = queue.Queue()
    threading.Thread(target=run_graph, args=(topic, num_papers, q), daemon=True).start()
    
    def event_stream():
        while True:
            try:
                kind, data = q.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            
            if kind == "log":
                message = re.sub(r"^\W*\[Graph\]\s*", "", data)
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


@api.post("/api/download-pdf")
def download_pdf(req: dict):
    synthesis = req.get("synthesis", "")
    topic = req.get("topic", "research_report")
    
    if not synthesis:
        raise HTTPException(status_code=400, detail="No synthesis provided")
    
    # Create temporary PDF path
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf_path = tmp.name
    
    # Use the new WeasyPrint generator
    success = generate_pdf_from_markdown(synthesis, pdf_path)
    
    if not success:
        import os
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(status_code=500, detail="PDF generation failed.")
    
    safe_topic = "".join(e for e in topic if e.isalnum() or e == " ").strip().replace(" ", "_")[:30]
    
    from fastapi.responses import FileResponse
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"research_report_{safe_topic}.pdf",
        headers={"Content-Disposition": f"attachment; filename=research_report_{safe_topic}.pdf"}
    )

@api.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


api.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(api, host="127.0.0.1", port=8000)
