"""
Adaptive Video Keyframe Extraction + Foreground Activity Peak Detection for Mobile Video RAG.
Combines:
1. Foreground Activity Peak Detection (L1 deviation from temporal median background) so static-camera
   clips capture brief passing objects (cyclists, cars, people, products).
2. HSV Histogram Bhattacharyya Scene-Change Detection for camera cuts.
3. Uniform Coverage Fallback across video duration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
KEYFRAME_DIR = BASE_DIR / "data" / "keyframes"


def extract_video_keyframes(
    video_path: Path,
    media_id: str,
    max_keyframes: int = 8,
    min_spacing_sec: float = 1.8,
    scene_threshold: float = 0.28,
) -> list[dict[str, Any]]:
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0

    # Sample candidate frames roughly every ~0.5 sec
    step = max(1, int(round(fps * 0.5)))
    candidates: list[tuple[float, np.ndarray, float, np.ndarray]] = []
    prev_hist: np.ndarray | None = None

    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = round(idx / fps, 2) if fps > 0 else 0.0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        delta = 1.0 if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        tiny = cv2.resize(frame, (160, 90)).astype(np.float32)
        candidates.append((t_sec, frame, delta, tiny))
        prev_hist = hist
        idx += step
        if total_frames > 0 and idx >= total_frames:
            break
    cap.release()

    if not candidates:
        return []

    # Compute foreground activity deviation from temporal median background
    median_bg = np.median(np.stack([c[3] for c in candidates]), axis=0)
    fg_devs = [float(np.mean(np.abs(c[3] - median_bg))) for c in candidates]

    selected_indices: dict[int, str] = {}

    # 1. Pick top foreground activity peaks (up to 4) spaced by min_spacing_sec
    ranked_fg = sorted(range(len(candidates)), key=lambda i: -fg_devs[i])
    for i in ranked_fg:
        t_sec = candidates[i][0]
        if all(abs(t_sec - candidates[j][0]) >= min_spacing_sec for j in selected_indices):
            selected_indices[i] = "foreground_activity"
        if len(selected_indices) >= 4:
            break

    # 2. Add strong scene-change cuts (up to 2 more)
    for i, (t_sec, _, delta, _) in enumerate(candidates):
        if len(selected_indices) >= max_keyframes - 2:
            break
        if delta >= scene_threshold and all(abs(t_sec - candidates[j][0]) >= min_spacing_sec for j in selected_indices):
            selected_indices[i] = "scene_change"

    # 3. Uniform coverage fallback up to max_keyframes
    target_indices = np.linspace(0, len(candidates) - 1, min(max_keyframes, len(candidates)), dtype=int)
    for ti in target_indices:
        if len(selected_indices) >= max_keyframes:
            break
        i = int(ti)
        t_sec = candidates[i][0]
        if all(abs(t_sec - candidates[j][0]) >= 1.2 for j in selected_indices):
            selected_indices[i] = "uniform_sample"

    selected = [(candidates[i][0], candidates[i][1], method) for i, method in selected_indices.items()]
    selected.sort(key=lambda x: x[0])

    # Remove stale keyframes for this media_id before saving fresh ones
    for old in KEYFRAME_DIR.glob(f"{media_id}_f*.jpg"):
        old.unlink()

    results: list[dict[str, Any]] = []
    for k_idx, (t_sec, bgr, method) in enumerate(selected):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((768, 768))
        out_path = KEYFRAME_DIR / f"{media_id}_f{k_idx:02d}.jpg"
        pil_img.save(out_path, format="JPEG", quality=88)

        t_start = max(0.0, round(t_sec - 1.5, 2))
        t_end = min(round(duration, 2), round(t_sec + 1.5, 2)) if duration > 0 else round(t_sec + 1.5, 2)
        results.append({
            "frame_id": f"{media_id}_f{k_idx:02d}",
            "media_id": media_id,
            "frame_index": k_idx,
            "timestamp_sec": t_sec,
            "window_start_sec": t_start,
            "window_end_sec": t_end,
            "extraction_method": method,
            "frame_path": str(out_path.relative_to(BASE_DIR)).replace("\\", "/"),
        })
    return results


def build_all_video_keyframes(verified_manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(verified_manifest_path.read_text(encoding="utf-8"))
    all_frames: list[dict[str, Any]] = []
    for item in manifest:
        if item["media_type"] == "video":
            vpath = BASE_DIR / item["local_path"]
            frames = extract_video_keyframes(vpath, item["media_id"])
            all_frames.extend(frames)
            print(f"Extracted {len(frames)} keyframes for {item['media_id']}")
    out_json = BASE_DIR / "data" / "video_keyframes.json"
    out_json.write_text(json.dumps(all_frames, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_frames


if __name__ == "__main__":
    build_all_video_keyframes(BASE_DIR / "data" / "manifest_verified.json")
