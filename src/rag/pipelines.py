"""
Five Applicable Visual RAG Architectures for Mobile Photo & Video Retrieval:
- R0_direct_dense      : Pure visual embedding retrieval (no captioning, no reranking)
- R1_caption_hybrid    : Caption-augmented Dual-Index + BM25 Reciprocal Rank Fusion
- R2_two_stage_rerank  : Coarse quantized retrieval (BQ/INT8) -> Exact Visual + Caption Cross-Signal Rerank
- R3_video_temporal    : Frame-level retrieval with Top-M temporal window pooling -> [t_start, t_end]
- R4_adaptive_rlaif_rag: Query-decomposed media filter + Hybrid RRF + RLAIF-trained Reward Reranker
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.indexers.search_algorithms import build_index
from src.preprocessing.query_processor import ProcessedQuery, preprocess_query


def _l2_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


@dataclass
class UnitRecord:
    unit_id: str          # image media_id or keyframe frame_id
    media_id: str         # parent media_id
    media_type: str       # "image" | "video"
    title: str
    theme: str
    caption: str
    searchable_text: str
    thumb_path: str
    media_path: str
    author: str
    license_short: str
    license_url: str
    source_page_url: str
    source_platform: str
    timestamp_sec: float | None = None
    window_start_sec: float | None = None
    window_end_sec: float | None = None
    duration_sec: float = 0.0


class VisualRAGWorkspace:
    """
    Holds indexed units (images + video keyframes), precomputed visual & caption embeddings,
    searchable indices, and optional RLAIF-learned linear/logistic reward weights.
    """

    def __init__(
        self,
        units: list[UnitRecord],
        visual_embs: np.ndarray,
        caption_embs: np.ndarray,
        search_algo: str = "S0_flat_fp32",
        rlaif_reward_weights: np.ndarray | None = None,
    ) -> None:
        self.units = units
        self.visual_embs = _l2_norm(visual_embs.astype(np.float32))
        self.caption_embs = _l2_norm(caption_embs.astype(np.float32))
        self.search_algo = search_algo
        self.rlaif_reward_weights = rlaif_reward_weights

        texts = [u.searchable_text for u in units]
        self.vis_index = build_index(search_algo, self.visual_embs, texts=texts)
        self.cap_index = build_index(search_algo, self.caption_embs, texts=texts)
        self.hybrid_index = build_index("S6_hybrid_bm25_rrf", self.visual_embs, texts=texts)

    def _encode_query_variants(self, encoder: Any, pq: ProcessedQuery) -> np.ndarray:
        variants = pq.expanded_variants or [pq.primary_text]
        vecs = encoder.encode_texts(variants)
        centroid = np.mean(vecs, axis=0, keepdims=True)
        return _l2_norm(centroid)[0]

    def _aggregate_to_media(
        self,
        unit_scores: list[tuple[int, float]],
        rag_mode: str,
        media_filter: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[tuple[int, float]]] = {}
        for u_idx, score in unit_scores:
            u = self.units[u_idx]
            if media_filter and u.media_type != media_filter:
                continue
            grouped.setdefault(u.media_id, []).append((u_idx, float(score)))

        merged: list[dict[str, Any]] = []
        for media_id, items in grouped.items():
            items.sort(key=lambda x: -x[1])
            best_idx, best_score = items[0]
            best_u = self.units[best_idx]
            if rag_mode in {"R3_video_temporal", "R4_adaptive_rlaif_rag"} and best_u.media_type == "video":
                # Top-2 temporal consistency pooling for video clips
                top_m = [s for _, s in items[:2]]
                agg_score = 0.75 * top_m[0] + 0.25 * (sum(top_m) / len(top_m))
            else:
                agg_score = best_score
            merged.append({
                "media_id": media_id,
                "unit_id": best_u.unit_id,
                "media_type": best_u.media_type,
                "title": best_u.title,
                "theme": best_u.theme,
                "score": float(agg_score),
                "visual_sim": float(best_score),
                "caption": best_u.caption,
                "thumb_path": best_u.thumb_path,
                "media_path": best_u.media_path,
                "timestamp_sec": best_u.timestamp_sec,
                "window_start_sec": best_u.window_start_sec,
                "window_end_sec": best_u.window_end_sec,
                "duration_sec": best_u.duration_sec,
                "author": best_u.author,
                "license_short": best_u.license_short,
                "license_url": best_u.license_url,
                "source_page_url": best_u.source_page_url,
                "source_platform": best_u.source_platform,
                "unit_index": best_idx,
            })
        merged.sort(key=lambda x: -x["score"])
        return merged[:top_k]

    def retrieve(
        self,
        raw_query: str,
        encoder: Any,
        prep_strategy: str = "P3_prompt_template",
        rag_mode: str = "R4_adaptive_rlaif_rag",
        top_k: int = 5,
        explicit_media_filter: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        multilingual = getattr(encoder.spec, "multilingual", False)
        pq = preprocess_query(raw_query, prep_strategy, model_supports_multilingual=multilingual)
        q_vec = self._encode_query_variants(encoder, pq)
        q_lex = pq.bilingual_english_text + " " + pq.primary_text

        n_units = len(self.units)
        cand_pool = min(n_units, max(top_k * 8, 40))

        # Apply query-decomposed media filter only in adaptive/clean modes or if user explicitly filtered
        active_filter = explicit_media_filter
        if active_filter is None and rag_mode == "R4_adaptive_rlaif_rag" and prep_strategy != "P0_raw":
            active_filter = pq.media_type_filter

        if rag_mode == "R0_direct_dense":
            out = self.vis_index.search(q_vec, query_text=q_lex, top_k=cand_pool)
            unit_scores = list(zip(out.indices, out.scores))

        elif rag_mode == "R1_caption_hybrid":
            v_out = self.vis_index.search(q_vec, query_text=q_lex, top_k=cand_pool)
            c_out = self.cap_index.search(q_vec, query_text=q_lex, top_k=cand_pool)
            h_out = self.hybrid_index.search(q_vec, query_text=q_lex, top_k=cand_pool)
            rrf: dict[int, float] = {}
            for r, idx in enumerate(v_out.indices):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + r + 1)
            for r, idx in enumerate(c_out.indices):
                rrf[idx] = rrf.get(idx, 0.0) + 0.7 / (60 + r + 1)
            for r, idx in enumerate(h_out.indices):
                rrf[idx] = rrf.get(idx, 0.0) + 0.8 / (60 + r + 1)
            unit_scores = sorted(rrf.items(), key=lambda x: -x[1])

        elif rag_mode == "R2_two_stage_rerank":
            # Stage 1: Fast search via workspace index; Stage 2: exact visual + caption linear blend
            coarse = self.vis_index.search(q_vec, query_text=q_lex, top_k=cand_pool)
            reranked = []
            for idx in coarse.indices:
                v_exact = float(self.visual_embs[idx] @ q_vec)
                c_exact = float(self.caption_embs[idx] @ q_vec)
                reranked.append((idx, 0.68 * v_exact + 0.32 * c_exact))
            unit_scores = sorted(reranked, key=lambda x: -x[1])

        elif rag_mode == "R3_video_temporal":
            v_out = self.vis_index.search(q_vec, query_text=q_lex, top_k=n_units)
            unit_scores = list(zip(v_out.indices, v_out.scores))

        elif rag_mode == "R4_adaptive_rlaif_rag":
            v_out = self.vis_index.search(q_vec, query_text=q_lex, top_k=cand_pool)
            c_out = self.cap_index.search(q_vec, query_text=q_lex, top_k=cand_pool)
            h_out = self.hybrid_index.search(q_vec, query_text=q_lex, top_k=cand_pool)

            v_rank = {idx: r for r, idx in enumerate(v_out.indices)}
            c_rank = {idx: r for r, idx in enumerate(c_out.indices)}
            h_rank = {idx: r for r, idx in enumerate(h_out.indices)}
            all_cands = set(v_out.indices) | set(c_out.indices) | set(h_out.indices)

            scored = []
            kw_set = set(w.lower() for w in pq.keywords)
            for idx in all_cands:
                u = self.units[idx]
                v_sim = float(self.visual_embs[idx] @ q_vec)
                c_sim = float(self.caption_embs[idx] @ q_vec)
                rrf_v = 1.0 / (60 + v_rank.get(idx, cand_pool) + 1)
                rrf_c = 1.0 / (60 + c_rank.get(idx, cand_pool) + 1)
                rrf_h = 1.0 / (60 + h_rank.get(idx, cand_pool) + 1)
                doc_tokens = set(u.searchable_text.lower().split())
                kw_overlap = len(kw_set & doc_tokens) / max(len(kw_set), 1)
                media_bonus = 1.0 if (pq.media_type_filter and u.media_type == pq.media_type_filter) else 0.0

                feats = np.array([v_sim, c_sim, rrf_v * 30.0, rrf_c * 30.0, rrf_h * 30.0, kw_overlap, media_bonus], dtype=np.float32)
                if self.rlaif_reward_weights is not None:
                    score = float(feats @ self.rlaif_reward_weights)
                else:
                    score = float(0.55 * v_sim + 0.25 * c_sim + 0.12 * (rrf_h * 30.0) + 0.08 * kw_overlap + 0.10 * media_bonus)
                scored.append((idx, score))
            unit_scores = sorted(scored, key=lambda x: -x[1])
        else:
            raise ValueError(f"Unknown RAG mode: {rag_mode}")

        results = self._aggregate_to_media(unit_scores, rag_mode=rag_mode, media_filter=active_filter, top_k=top_k)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "query": raw_query,
            "preprocessing": {
                "strategy": pq.strategy,
                "core_text": pq.primary_text,
                "bilingual_text": pq.bilingual_english_text,
                "media_type_filter": active_filter,
                "temporal_hint": pq.extracted_temporal_hint,
                "keywords": pq.keywords,
                "expanded_variants": pq.expanded_variants,
            },
            "rag_mode": rag_mode,
            "search_algo": self.search_algo,
            "latency_ms": round(total_ms, 2),
            "index_ram_kb": round(self.vis_index.memory_bytes() / 1024.0, 2),
            "results": results,
        }
