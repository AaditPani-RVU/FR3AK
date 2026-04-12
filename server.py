from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

# ensure the FR3AK package root is on sys.path when run from anywhere
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from pipeline.analyzer import ConversationAnalyzer
from pipeline.insights import analyze_insights
from pipeline.visualizer import build_visualization_data
from utils.parser import parse_conversation


app = FastAPI(title="FR3AK", docs_url=None, redoc_url=None)

_analyzer: ConversationAnalyzer | None = None


def get_analyzer() -> ConversationAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ConversationAnalyzer()
    return _analyzer


def _normalize_json_records(payload: Any) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            payload = payload["records"]
        elif isinstance(payload.get("messages"), list):
            payload = payload["messages"]
        else:
            payload = []
    if not isinstance(payload, list):
        return records
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        speaker = str(item.get("speaker") or item.get("speaker_id") or "").strip()
        text = str(item.get("cleaned_message") or item.get("message") or item.get("text") or "").strip()
        timestamp = item.get("timestamp", None)
        if not speaker or not text:
            continue
        records.append({"speaker": speaker, "cleaned_message": text, "timestamp": timestamp})
    return records


def load_data(content: bytes, filename: str) -> Tuple[List[Dict[str, Any]], str]:
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        raw_text = content.decode("utf-8", errors="ignore")
        parsed = parse_conversation(raw_text)
        records: List[Dict[str, Any]] = []
        for record in parsed.records:
            speaker = str(getattr(record, "speaker", "") or getattr(record, "speaker_id", "")).strip()
            text = str(getattr(record, "cleaned_message", "") or "").strip()
            ts_obj = getattr(record, "timestamp", None)
            ts = getattr(ts_obj, "raw", None) if ts_obj else None
            if not speaker or not text:
                continue
            records.append({"speaker": speaker, "cleaned_message": text, "timestamp": ts})
        return records, "txt"

    if ext == ".json":
        payload = json.loads(content.decode("utf-8", errors="ignore"))
        return _normalize_json_records(payload), "json"

    return [], ext


# ── static files ────────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = static_dir / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── API ──────────────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()
    records, source_type = load_data(content, file.filename or "upload.txt")
    if not records:
        raise HTTPException(status_code=422, detail="No valid records found in uploaded file.")

    analyzer = get_analyzer()
    analysis_output = analyzer.analyze(records)
    insights_output = analyze_insights(analysis_output)
    viz_data = build_visualization_data(analysis_output, insights_output)

    return JSONResponse({
        "records": records,
        "analysis": analysis_output,
        "insights": insights_output,
        "viz_data": viz_data,
        "source_type": source_type,
        "filename": file.filename or "",
    })


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
