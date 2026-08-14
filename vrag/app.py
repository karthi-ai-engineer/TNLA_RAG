"""P7 — FastAPI server: player + chat + evidence cards + graph.

Design rule for the live demo: RETRIEVAL never touches the network — index,
segments, graph and frames are all read from disk artifacts.  Only the final
answer-generation call goes to Gemini (and even that is disk-cached for any
question asked before).

Run:
    uvicorn vrag.app:app --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vrag import config
from vrag.answer import answer_question
from vrag.cache import read_json
from vrag.gemini import GeminiClient
from vrag.logging_setup import setup_logging
from vrag.retrieve import HybridIndex
from vrag.roster import ROSTER_DIR

log = logging.getLogger(__name__)

STATE: dict = {"indexes": {}, "client": None}


def list_ready_videos() -> list[str]:
    """Video ids that have every artifact the UI needs."""
    out = []
    if not config.ARTIFACT_DIR.exists():
        return out
    for d in sorted(config.ARTIFACT_DIR.iterdir()):
        if all((d / f).exists() for f in
               ("segments.json", "index_meta.json", "vectors.npy", "graph.json")):
            out.append(d.name)
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    STATE["client"] = GeminiClient()
    for vid in list_ready_videos():
        try:
            STATE["indexes"][vid] = HybridIndex(vid)
            log.info("app: loaded index for %s", vid)
        except Exception as exc:                          # noqa: BLE001
            log.error("app: could not load %s: %s", vid, exc)
    if not STATE["indexes"]:
        log.error("app: NO ready videos — run the pipeline first (P1..P5)")
    yield
    STATE["client"].close()


app = FastAPI(title="TNLA Video RAG", lifespan=lifespan)

app.mount("/roster", StaticFiles(directory=ROSTER_DIR), name="roster")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class AskBody(BaseModel):
    question: str
    video_id: str


@app.get("/api/videos")
def videos():
    out = []
    for vid, idx in STATE["indexes"].items():
        seg_art = read_json(config.artifact_dir(vid) / "segments.json")
        proxy = config.PROXY_DIR / f"{vid}.mp4"
        out.append({
            "video_id": vid,
            "duration": seg_art["duration"],
            "segment_count": seg_art["segment_count"],
            "has_proxy": proxy.exists(),
        })
    return out


@app.get("/api/segments/{video_id}")
def segments(video_id: str):
    p = config.artifact_dir(video_id) / "segments.json"
    if not p.exists():
        raise HTTPException(404, f"no segments for {video_id}")
    return read_json(p)


@app.get("/api/turns/{video_id}")
def turns(video_id: str):
    p = config.artifact_dir(video_id) / "turns.json"
    if not p.exists():
        raise HTTPException(404, f"no turns for {video_id}")
    return read_json(p)


@app.get("/api/graph/{video_id}")
def graph(video_id: str):
    p = config.artifact_dir(video_id) / "graph.json"
    if not p.exists():
        raise HTTPException(404, f"no graph for {video_id}")
    return read_json(p)


@app.post("/api/ask")
def ask(body: AskBody):
    idx = STATE["indexes"].get(body.video_id)
    if idx is None:
        raise HTTPException(404, f"video {body.video_id} not indexed")
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "empty question")
    try:
        return answer_question(STATE["client"], body.video_id, q, index=idx)
    except Exception as exc:                              # noqa: BLE001
        log.exception("ask failed")
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")


@app.get("/video/{video_id}")
def video(video_id: str):
    """Serve the H.264 proxy (browser-seekable); FileResponse handles Range."""
    proxy = config.PROXY_DIR / f"{video_id}.mp4"
    if not proxy.exists():
        raise HTTPException(404, f"no proxy for {video_id} — transcode it into data/proxy/")
    return FileResponse(proxy, media_type="video/mp4")


@app.get("/frame/{video_id}/{name}")
def frame(video_id: str, name: str):
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad frame name")
    p = config.artifact_dir(video_id) / "frames" / name
    if not p.exists():
        raise HTTPException(404, "no such frame")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/")
def index_page():
    html = (config.ROOT / "web" / "index.html")
    if not html.exists():
        raise HTTPException(500, "web/index.html missing")
    return HTMLResponse(html.read_text(encoding="utf-8"))
