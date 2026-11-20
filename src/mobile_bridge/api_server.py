"""
Lightweight FastAPI Mobile Bridge Server for PocketLens Mobile Visual RAG.
Serves the responsive Mobile PWA (mobile_web/index.html) + ONNX INT8 / SQLite mobile bundle
so smartphones on the same LAN or desktop browsers can query interactively.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from src.models.encoders import ENCODER_SPECS, load_encoder
from src.rag.pipelines import UnitRecord, VisualRAGWorkspace
from src.utils.win_asyncio import install_windows_proactor_guard, resolve_playable_video

BASE_DIR = Path(__file__).resolve().parents[2]
BUNDLE_DIR = BASE_DIR / "exports" / "mobile_bundle"
ONNX_DIR = BASE_DIR / "exports" / "onnx"
MOBILE_WEB = BASE_DIR / "mobile_web" / "index.html"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    install_windows_proactor_guard()
    _load_workspace()
    yield


app = FastAPI(title="PocketLens Mobile Visual RAG API", lifespan=lifespan)
app.mount("/data", StaticFiles(directory=str(BASE_DIR / "data")), name="data")

_STATE: dict[str, Any] = {}


def _load_workspace() -> None:
    if _STATE.get("ready"):
        return
    meta = json.loads((BUNDLE_DIR / "bundle_meta.json").read_text(encoding="utf-8"))
    conn = sqlite3.connect(str(BUNDLE_DIR / "gallery_index.sqlite"))
    rows = conn.execute("SELECT * FROM units ORDER BY idx").fetchall()
    conn.close()

    units: list[UnitRecord] = []
    vis_list, cap_list = [], []
    for r in rows:
        units.append(
            UnitRecord(
                unit_id=r[1],
                media_id=r[2],
                media_type=r[3],
                title=r[4],
                theme=r[5],
                caption=r[6],
                searchable_text=r[7],
                thumb_path=r[8],
                media_path=r[9],
                author=r[10],
                license_short=r[11],
                license_url=r[12],
                source_page_url=r[13],
                source_platform=r[14],
                timestamp_sec=r[15],
                window_start_sec=r[16],
                window_end_sec=r[17],
                duration_sec=r[18],
            )
        )
        vq = np.frombuffer(r[20], dtype=np.int8).astype(np.float32) * float(r[19])
        cq = np.frombuffer(r[22], dtype=np.int8).astype(np.float32) * float(r[21])
        vis_list.append(vq)
        cap_list.append(cq)

    weights = np.array(meta["rlaif_weights"], dtype=np.float32) if meta.get("rlaif_weights") else None
    enc = load_encoder(meta["model_key"], onnx_dir=ONNX_DIR)
    ws = VisualRAGWorkspace(
        units=units,
        visual_embs=np.vstack(vis_list),
        caption_embs=np.vstack(cap_list),
        search_algo="S1_sq8_int8",
        rlaif_reward_weights=weights,
    )
    _STATE.update({"ready": True, "meta": meta, "encoder": enc, "workspace": ws})


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(MOBILE_WEB.read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="#2563EB" stroke-width="2.2"><circle cx="11" cy="11" r="8"/>'
        '<path d="m21 21-4.3-4.3"/></svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/gallery")
def gallery(media_type: str = Query("all")) -> dict[str, Any]:
    _load_workspace()
    ws: VisualRAGWorkspace = _STATE["workspace"]
    seen: set[str] = set()
    items = []
    for u in ws.units:
        if u.media_id in seen:
            continue
        if media_type != "all" and u.media_type != media_type:
            continue
        seen.add(u.media_id)
        play_rel = resolve_playable_video(u.media_path, BASE_DIR)[0] if u.media_type == "video" else u.media_path
        items.append({
            "media_id": u.media_id,
            "media_type": u.media_type,
            "title": u.title,
            "caption": u.caption,
            "thumb_url": f"/{u.thumb_path}",
            "media_url": f"/{play_rel}",
            "duration_sec": u.duration_sec,
            "author": u.author,
            "license_short": u.license_short,
            "license_url": u.license_url,
            "source_page_url": u.source_page_url,
            "source_platform": u.source_platform,
            "score": 1.0,
        })
    return {"items": items}


@app.get("/api/search")
def search(
    q: str = Query(...),
    media_type: str = Query("all"),
    rag: str = Query("R4_adaptive_rlaif_rag"),
    top_k: int = Query(12),
) -> dict[str, Any]:
    _load_workspace()
    ws: VisualRAGWorkspace = _STATE["workspace"]
    enc = _STATE["encoder"]
    explicit = None if media_type == "all" else media_type
    out = ws.retrieve(
        raw_query=q,
        encoder=enc,
        prep_strategy="P3_prompt_template",
        rag_mode=rag,
        top_k=top_k,
        explicit_media_filter=explicit,
    )
    for r in out["results"]:
        r["thumb_url"] = f"/{r['thumb_path']}"
        play_rel = resolve_playable_video(r["media_path"], BASE_DIR)[0] if r["media_type"] == "video" else r["media_path"]
        r["media_url"] = f"/{play_rel}"
    top_titles = ", ".join(r["title"][:28] for r in out["results"][:3])
    out["rag_summary"] = (
        f"질의 '{out['preprocessing']['core_text']}'에 대해 상위 {len(out['results'])}건을 찾았습니다 "
        f"(대표 결과: {top_titles})."
        if out["results"]
        else "조건에 맞는 사진/동영상을 찾지 못했습니다."
    )
    return out


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    _load_workspace()
    meta = _STATE["meta"]
    ws: VisualRAGWorkspace = _STATE["workspace"]
    manifest = json.loads((BASE_DIR / "data" / "manifest_verified.json").read_text(encoding="utf-8"))
    spec = ENCODER_SPECS[meta["model_key"]]
    prefix = spec.base_key or spec.key
    model_files = list(ONNX_DIR.glob(f"{prefix}_*int8.onnx"))
    model_mb = round(sum(p.stat().st_size for p in model_files) / (1024**2), 2)
    return {
        "model_key": meta["model_key"],
        "bundle_mb": meta["sqlite_mb"],
        "model_mb": model_mb,
        "n_images": sum(1 for m in manifest if m["media_type"] == "image"),
        "n_videos": sum(1 for m in manifest if m["media_type"] == "video"),
        "n_keyframes": sum(1 for u in ws.units if u.media_type == "video"),
    }
