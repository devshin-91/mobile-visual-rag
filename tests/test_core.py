"""Unit tests for query preprocessing, mobile vector search algorithms, and RLAIF parsing."""
from __future__ import annotations

import numpy as np

from src.eval.rlaif_judge import _parse_json_score, train_bradley_terry_reward_reranker
from src.indexers.search_algorithms import build_index
from src.preprocessing.query_processor import preprocess_query


def test_query_preprocessing_strategies() -> None:
    q = "지난 주말에 찍은 해변에서 뛰어노는 강아지 동영상 좀 찾아줘"
    p0 = preprocess_query(q, "P0_raw", model_supports_multilingual=False)
    assert p0.media_type_filter is None
    assert p0.primary_text == q

    p1 = preprocess_query(q, "P1_intent_clean", model_supports_multilingual=True)
    assert p1.media_type_filter == "video"
    assert p1.extracted_temporal_hint == "지난주" or "주말" in q
    assert "찾아줘" not in p1.primary_text

    p3_en = preprocess_query(q, "P3_prompt_template", model_supports_multilingual=False)
    assert p3_en.media_type_filter == "video"
    assert "dog" in p3_en.primary_text.lower()
    assert "beach" in p3_en.primary_text.lower()

    p4 = preprocess_query(q, "P4_multi_aspect_expansion", model_supports_multilingual=True)
    assert len(p4.expanded_variants) >= 2


def test_mobile_search_algorithms() -> None:
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(64, 128)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    texts = [f"sample caption item_{i}" for i in range(64)]
    q = vecs[13].copy()

    for algo in ["S0_flat_fp32", "S1_sq8_int8", "S2_bq_hamming", "S3_bq_rescore", "S4_hnsw", "S5_ivfpq", "S6_hybrid_bm25_rrf"]:
        idx = build_index(algo, vecs, texts)
        out = idx.search(q, query_text="sample caption item_13", top_k=5)
        assert len(out.indices) == 5
        assert out.indices[0] == 13 or (algo in {"S2_bq_hamming", "S5_ivfpq"} and 13 in out.indices)
        assert idx.memory_bytes() > 0


def test_rlaif_parse_and_bradley_terry() -> None:
    parsed = _parse_json_score('Result: {"entity": 5, "attribute": 4.5, "media_fit": 5, "overall": 4.8, "reason": "exact"}')
    assert parsed["entity"] == 5.0 and parsed["overall"] == 4.8

    pairs = [
        (np.array([0.9, 0.8, 0.5, 0.4, 0.5, 0.8, 1.0], dtype=np.float32),
         np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0], dtype=np.float32))
        for _ in range(6)
    ]
    w = train_bradley_terry_reward_reranker(pairs)
    assert w.shape == (7,)
    assert np.all(w >= 0)


def test_windows_proactor_guard_and_playable_video() -> None:
    from pathlib import Path
    from src.utils.win_asyncio import _finish_proactor_cleanup, install_windows_proactor_guard, resolve_playable_video

    install_windows_proactor_guard()

    class _DummySock:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _DummyServer:
        def __init__(self) -> None:
            self.detached = False

        def _detach(self) -> None:
            self.detached = True

    class _DummyTransport:
        def __init__(self) -> None:
            self._sock = _DummySock()
            self._server = _DummyServer()
            self._called_connection_lost = False

    t = _DummyTransport()
    sock_ref = t._sock
    srv_ref = t._server
    _finish_proactor_cleanup(t)
    assert sock_ref.closed is True
    assert srv_ref.detached is True
    assert t._sock is None and t._server is None and t._called_connection_lost is True

    base_dir = Path(__file__).resolve().parents[1]
    rel_mp4, mime_mp4 = resolve_playable_video("data/raw/videos/vid_cat_puddle.ogv", base_dir)
    assert rel_mp4 == "data/raw/videos/vid_cat_puddle_web.mp4"
    assert mime_mp4 == "video/mp4"
    rel_webm, mime_webm = resolve_playable_video("data/raw/videos/vid_cat_salto.webm", base_dir)
    assert rel_webm == "data/raw/videos/vid_cat_salto.webm"
    assert mime_webm == "video/webm"
