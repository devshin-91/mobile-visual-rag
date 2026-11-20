"""
Packages an On-Device Mobile Index Bundle (SQLite metadata + INT8 quantized visual & caption vectors)
that can be loaded directly by mobile clients (Android/iOS/PWA) without full FP32 embeddings.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from src.rag.pipelines import UnitRecord

BASE_DIR = Path(__file__).resolve().parents[2]
BUNDLE_DIR = BASE_DIR / "exports" / "mobile_bundle"


def _quantize_sq8(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    normed = (vecs / np.maximum(norms, 1e-12)).astype(np.float32)
    scales = np.maximum(np.max(np.abs(normed), axis=1, keepdims=True) / 127.0, 1e-8).astype(np.float16)
    q = np.clip(np.round(normed / scales.astype(np.float32)), -127, 127).astype(np.int8)
    return q, scales[:, 0]


def export_mobile_bundle(
    units: list[UnitRecord],
    visual_embs: np.ndarray,
    caption_embs: np.ndarray,
    model_key: str,
    rlaif_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    db_path = BUNDLE_DIR / "gallery_index.sqlite"
    if db_path.exists():
        db_path.unlink()

    vis_q, vis_s = _quantize_sq8(visual_embs)
    cap_q, cap_s = _quantize_sq8(caption_embs)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE units (
            idx INTEGER PRIMARY KEY,
            unit_id TEXT,
            media_id TEXT,
            media_type TEXT,
            title TEXT,
            theme TEXT,
            caption TEXT,
            searchable_text TEXT,
            thumb_path TEXT,
            media_path TEXT,
            author TEXT,
            license_short TEXT,
            license_url TEXT,
            source_page_url TEXT,
            source_platform TEXT,
            timestamp_sec REAL,
            window_start_sec REAL,
            window_end_sec REAL,
            duration_sec REAL,
            vis_scale REAL,
            vis_vec BLOB,
            cap_scale REAL,
            cap_vec BLOB
        )
        """
    )
    for i, u in enumerate(units):
        cur.execute(
            "INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                i,
                u.unit_id,
                u.media_id,
                u.media_type,
                u.title,
                u.theme,
                u.caption,
                u.searchable_text,
                u.thumb_path,
                u.media_path,
                u.author,
                u.license_short,
                u.license_url,
                u.source_page_url,
                u.source_platform,
                u.timestamp_sec,
                u.window_start_sec,
                u.window_end_sec,
                u.duration_sec,
                float(vis_s[i]),
                vis_q[i].tobytes(),
                float(cap_s[i]),
                cap_q[i].tobytes(),
            ),
        )
    conn.commit()
    conn.close()

    meta = {
        "model_key": model_key,
        "num_units": len(units),
        "dim": int(visual_embs.shape[1]),
        "sqlite_mb": round(db_path.stat().st_size / (1024**2), 4),
        "rlaif_weights": rlaif_weights.tolist() if rlaif_weights is not None else None,
    }
    (BUNDLE_DIR / "bundle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
