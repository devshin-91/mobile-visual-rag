"""
Full Multi-Axis Visual RAG Benchmark & Closed-Loop RLAIF Evaluation Suite.

Runs real experiments across:
1. Models & Quantization: TinyCLIP-8M (FP32 & ONNX INT8), TinyCLIP-40M, CLIP-B/32 (FP32 & ONNX INT8), SigLIP2-Base (FP32 & ONNX INT8)
2. Natural Language Query Preprocessing: P0_raw, P1_intent_clean, P2_keywords, P3_prompt_template, P4_multi_aspect_expansion
3. Search Algorithms: S0_flat_fp32, S1_sq8_int8, S2_bq_hamming, S3_bq_rescore, S4_hnsw, S5_ivfpq, S6_hybrid_bm25_rrf (+ 10k scale benchmark)
4. Applicable Visual RAG Strategies: R0_direct_dense, R1_caption_hybrid, R2_two_stage_rerank, R3_video_temporal, R4_adaptive_rlaif_rag
5. RLAIF Evaluation (Qwen2-VL-2B-Instruct Constitutional Judge + AB/BA Debiased Pairwise + Bradley-Terry Reward Reranker on Train/Test split)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.eval.rlaif_judge import QwenVLMConstitutionalJudge, train_bradley_terry_reward_reranker
from src.indexers.search_algorithms import benchmark_10k_mobile_scaling
from src.mobile_bridge.mobile_bundle import export_mobile_bundle
from src.models.encoders import ENCODER_SPECS, load_encoder, profile_encoder_latency
from src.preprocessing.query_processor import preprocess_query
from src.rag.pipelines import UnitRecord, VisualRAGWorkspace
from src.rag.vlm_captioner import SmolVLM2Captioner

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EMB_CACHE_DIR = ARTIFACTS_DIR / "emb_cache"
ONNX_DIR = BASE_DIR / "exports" / "onnx"


def load_units_from_manifest() -> list[UnitRecord]:
    manifest = json.loads((DATA_DIR / "manifest_verified.json").read_text(encoding="utf-8"))
    keyframes = json.loads((DATA_DIR / "video_keyframes.json").read_text(encoding="utf-8"))
    captions = json.loads((ARTIFACTS_DIR / "captions_cache.json").read_text(encoding="utf-8"))

    by_id = {m["media_id"]: m for m in manifest}
    units: list[UnitRecord] = []

    for m in manifest:
        if m["media_type"] == "image":
            cap = captions.get(m["media_id"], m.get("description", ""))
            searchable = f"{m['theme']} {m['canonical_title']} {m.get('description', '')} {cap}"
            units.append(
                UnitRecord(
                    unit_id=m["media_id"],
                    media_id=m["media_id"],
                    media_type="image",
                    title=m["canonical_title"],
                    theme=m["theme"],
                    caption=cap,
                    searchable_text=searchable,
                    thumb_path=m["local_path"],
                    media_path=m["local_path"],
                    author=m["author"],
                    license_short=m["license_short"],
                    license_url=m["license_url"],
                    source_page_url=m["source_page_url"],
                    source_platform=m["source_platform"],
                    duration_sec=0.0,
                )
            )

    for kf in keyframes:
        m = by_id[kf["media_id"]]
        cap = captions.get(kf["frame_id"], m.get("description", ""))
        searchable = f"video {m['theme']} {m['canonical_title']} {m.get('description', '')} {cap}"
        units.append(
            UnitRecord(
                unit_id=kf["frame_id"],
                media_id=m["media_id"],
                media_type="video",
                title=m["canonical_title"],
                theme=m["theme"],
                caption=cap,
                searchable_text=searchable,
                thumb_path=kf["frame_path"],
                media_path=m["local_path"],
                author=m["author"],
                license_short=m["license_short"],
                license_url=m["license_url"],
                source_page_url=m["source_page_url"],
                source_platform=m["source_platform"],
                timestamp_sec=kf["timestamp_sec"],
                window_start_sec=kf["window_start_sec"],
                window_end_sec=kf["window_end_sec"],
                duration_sec=m.get("duration_sec", 0.0),
            )
        )
    return units


def compute_ir_metrics(retrieved_media_ids: list[str], relevant_ids: list[str]) -> dict[str, float]:
    rel_set = set(relevant_ids)
    if not rel_set:
        return {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0}

    r1 = 1.0 if retrieved_media_ids[:1] and retrieved_media_ids[0] in rel_set else 0.0
    r3 = len(set(retrieved_media_ids[:3]) & rel_set) / len(rel_set)
    r5 = len(set(retrieved_media_ids[:5]) & rel_set) / len(rel_set)

    mrr = 0.0
    for rank, mid in enumerate(retrieved_media_ids, start=1):
        if mid in rel_set:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    for rank, mid in enumerate(retrieved_media_ids[:5], start=1):
        if mid in rel_set:
            dcg += 1.0 / math.log2(rank + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel_set), 5) + 1))
    ndcg5 = dcg / idcg if idcg > 0 else 0.0
    return {
        "recall_at_1": r1,
        "recall_at_3": r3,
        "recall_at_5": r5,
        "mrr": mrr,
        "ndcg_at_5": ndcg5,
    }


def cache_embeddings_for_model(model_key: str, units: list[UnitRecord]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = EMB_CACHE_DIR / f"{model_key}.npz"
    prof_file = EMB_CACHE_DIR / f"{model_key}_profile.json"
    if cache_file.exists() and prof_file.exists():
        npz = np.load(cache_file)
        prof = json.loads(prof_file.read_text(encoding="utf-8"))
        return npz["visual"], npz["caption"], prof

    print(f"Encoding {len(units)} units with {model_key}...")
    enc = load_encoder(model_key, onnx_dir=ONNX_DIR)
    images = []
    for u in units:
        with Image.open(BASE_DIR / u.thumb_path) as im:
            images.append(im.convert("RGB").copy())
    captions = [u.caption for u in units]

    t0 = time.perf_counter()
    vis_embs = enc.encode_images(images)
    cap_embs = enc.encode_texts(captions)
    index_sec = time.perf_counter() - t0

    lat = profile_encoder_latency(enc, images[0], "a cute dog running on the beach")
    enc.unload()

    prof = {
        "model_key": model_key,
        "hf_id": ENCODER_SPECS[model_key].hf_id,
        "license": ENCODER_SPECS[model_key].license_name,
        "multilingual": ENCODER_SPECS[model_key].multilingual,
        "is_onnx": ENCODER_SPECS[model_key].is_onnx,
        "embedding_dim": int(vis_embs.shape[1]),
        "corpus_index_sec": round(index_sec, 2),
        **lat,
    }
    np.savez_compressed(cache_file, visual=vis_embs, caption=cap_embs)
    prof_file.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    return vis_embs, cap_embs, prof


def evaluate_config(
    ws: VisualRAGWorkspace,
    encoder: Any,
    queries: list[dict[str, Any]],
    prep_strategy: str,
    rag_mode: str,
) -> dict[str, Any]:
    metrics_list = []
    latencies = []
    per_type: dict[str, list[dict[str, float]]] = {}

    for q in queries:
        out = ws.retrieve(
            raw_query=q["query"],
            encoder=encoder,
            prep_strategy=prep_strategy,
            rag_mode=rag_mode,
            top_k=5,
        )
        mids = [r["media_id"] for r in out["results"]]
        m = compute_ir_metrics(mids, q["relevant_media_ids"])
        metrics_list.append(m)
        latencies.append(out["latency_ms"])
        per_type.setdefault(q.get("category", "general"), []).append(m)

    agg = {
        k: round(float(np.mean([x[k] for x in metrics_list])), 4)
        for k in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5"]
    }
    agg["search_latency_ms"] = round(float(np.mean(latencies)), 3)
    agg["index_ram_kb"] = round(ws.vis_index.memory_bytes() / 1024.0, 2)
    agg["by_category"] = {
        cat: {
            "recall_at_1": round(float(np.mean([x["recall_at_1"] for x in vals])), 4),
            "mrr": round(float(np.mean([x["mrr"] for x in vals])), 4),
            "ndcg_at_5": round(float(np.mean([x["ndcg_at_5"] for x in vals])), 4),
        }
        for cat, vals in per_type.items()
    }
    return agg


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    units = load_units_from_manifest()
    eval_queries = json.loads((DATA_DIR / "eval_queries.json").read_text(encoding="utf-8"))
    # Filter queries that actually have positive ground truth in corpus
    pos_queries = [q for q in eval_queries if len(q.get("relevant_media_ids", [])) > 0]
    train_queries = [q for q in pos_queries if q.get("split") == "train"]
    test_queries = [q for q in pos_queries if q.get("split") == "test"]
    print(f"Loaded {len(units)} visual units, {len(pos_queries)} positive queries ({len(train_queries)} train / {len(test_queries)} test)")

    all_models = [
        "M1_tinyclip_8m_fp32",
        "M2_tinyclip_8m_onnx_int8",
        "M3_tinyclip_40m_fp32",
        "M4_clip_b32_fp32",
        "M5_clip_b32_onnx_int8",
        "M6_siglip2_base_fp32",
        "M7_siglip2_base_onnx_int8",
    ]
    prep_strategies = [
        "P0_raw",
        "P1_intent_clean",
        "P2_keywords",
        "P3_prompt_template",
        "P4_multi_aspect_expansion",
    ]
    search_algos = [
        "S0_flat_fp32",
        "S1_sq8_int8",
        "S2_bq_hamming",
        "S3_bq_rescore",
        "S4_hnsw",
        "S5_ivfpq",
        "S6_hybrid_bm25_rrf",
    ]
    rag_modes = [
        "R0_direct_dense",
        "R1_caption_hybrid",
        "R2_two_stage_rerank",
        "R3_video_temporal",
        "R4_adaptive_rlaif_rag",
    ]

    # 1. Precompute & profile all encoders
    model_profiles = []
    cached_embs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mk in all_models:
        vis, cap, prof = cache_embeddings_for_model(mk, units)
        cached_embs[mk] = (vis, cap)
        model_profiles.append(prof)

    # 2. Experiment A: Query Preprocessing Ablation across all 7 models (with S0_flat_fp32, R0 & R1)
    print("\n=== Experiment A: Query Preprocessing x Model Ablation ===")
    exp_a_rows = []
    for mk in all_models:
        enc = load_encoder(mk, onnx_dir=ONNX_DIR)
        vis, cap = cached_embs[mk]
        ws = VisualRAGWorkspace(units, vis, cap, search_algo="S0_flat_fp32")
        for prep in prep_strategies:
            for rag in ["R0_direct_dense", "R1_caption_hybrid"]:
                res = evaluate_config(ws, enc, pos_queries, prep_strategy=prep, rag_mode=rag)
                exp_a_rows.append({
                    "model": mk,
                    "prep_strategy": prep,
                    "rag_mode": rag,
                    "search_algo": "S0_flat_fp32",
                    **res,
                })
        enc.unload()

    # 3. Experiment B: Search Algorithm Benchmark on Mobile ONNX INT8 models + 10k scaling
    print("\n=== Experiment B: Search Algorithms x Mobile ONNX Models ===")
    exp_b_rows = []
    mobile_models = ["M2_tinyclip_8m_onnx_int8", "M5_clip_b32_onnx_int8", "M7_siglip2_base_onnx_int8"]
    for mk in mobile_models:
        enc = load_encoder(mk, onnx_dir=ONNX_DIR)
        vis, cap = cached_embs[mk]
        for algo in search_algos:
            ws = VisualRAGWorkspace(units, vis, cap, search_algo=algo)
            res = evaluate_config(ws, enc, pos_queries, prep_strategy="P3_prompt_template", rag_mode="R0_direct_dense")
            exp_b_rows.append({
                "model": mk,
                "search_algo": algo,
                "prep_strategy": "P3_prompt_template",
                "rag_mode": "R0_direct_dense",
                **res,
            })
        enc.unload()

    scaling_10k_512 = benchmark_10k_mobile_scaling(dim=512, n_items=10_000)
    scaling_10k_768 = benchmark_10k_mobile_scaling(dim=768, n_items=10_000)

    # 4. RLAIF Closed-Loop Calibration on Train Queries using Qwen2-VL-2B-Instruct
    print("\n=== Experiment C & RLAIF: Closed-Loop Reward Training + Held-Out Evaluation ===")
    rlaif_model_key = "M7_siglip2_base_onnx_int8"
    enc_rlaif = load_encoder(rlaif_model_key, onnx_dir=ONNX_DIR)
    vis_r, cap_r = cached_embs[rlaif_model_key]
    ws_train = VisualRAGWorkspace(units, vis_r, cap_r, search_algo="S1_sq8_int8")

    # Gather diverse top candidates per train query for AI Judge scoring
    train_candidates_by_q: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for q in train_queries:
        pq = preprocess_query(q["query"], "P3_prompt_template", model_supports_multilingual=True)
        q_vec = enc_rlaif.encode_texts([pq.primary_text])[0]
        # Union of top candidates from R0 and R1
        r0 = ws_train.retrieve(q["query"], enc_rlaif, "P3_prompt_template", "R0_direct_dense", top_k=4)
        r1 = ws_train.retrieve(q["query"], enc_rlaif, "P3_prompt_template", "R1_caption_hybrid", top_k=4)
        seen_m = set()
        pool = []
        for item in r0["results"] + r1["results"]:
            if item["media_id"] not in seen_m:
                seen_m.add(item["media_id"])
                # Compute 7-dim feature vector at unit_index
                idx = item["unit_index"]
                u = units[idx]
                v_sim = float(ws_train.visual_embs[idx] @ q_vec)
                c_sim = float(ws_train.caption_embs[idx] @ q_vec)
                kw_set = set(w.lower() for w in pq.keywords)
                doc_tokens = set(u.searchable_text.lower().split())
                kw_overlap = len(kw_set & doc_tokens) / max(len(kw_set), 1)
                media_bonus = 1.0 if (pq.media_type_filter and u.media_type == pq.media_type_filter) else 0.0
                # Approximate normalized RRF signals
                item["features"] = [v_sim, c_sim, max(v_sim, 0.0) * 0.5, max(c_sim, 0.0) * 0.5, 0.5 * (v_sim + c_sim), kw_overlap, media_bonus]
                pool.append(item)
        train_candidates_by_q.append((q, pool[:5]))

    # Unload encoder temporarily if needed to keep GPU VRAM clean for Qwen2-VL-2B
    enc_rlaif.unload()

    judge = QwenVLMConstitutionalJudge()
    feature_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    train_judge_logs = []

    print("Running Qwen2-VL-2B Constitutional Judge on Train split to generate AI preference pairs...")
    for q, pool in train_candidates_by_q:
        scored_pool = []
        for cand in pool:
            js = judge.score_candidate(
                query=q["query"],
                image_path=BASE_DIR / cand["thumb_path"],
                candidate_media_type=cand["media_type"],
                expected_media_type=q.get("expected_media_type"),
            )
            scored_pool.append((cand, js))
        scored_pool.sort(key=lambda x: -x[1]["overall"])
        train_judge_logs.append({
            "query_id": q["id"],
            "query": q["query"],
            "scores": [{"media_id": c["media_id"], **s} for c, s in scored_pool],
        })
        # Create preference pairs whenever AI Judge finds >= 0.8 score margin
        for i in range(len(scored_pool)):
            for j in range(i + 1, len(scored_pool)):
                if scored_pool[i][1]["overall"] - scored_pool[j][1]["overall"] >= 0.8:
                    feature_pairs.append(
                        (
                            np.array(scored_pool[i][0]["features"], dtype=np.float32),
                            np.array(scored_pool[j][0]["features"], dtype=np.float32),
                        )
                    )

    rlaif_weights = train_bradley_terry_reward_reranker(feature_pairs)
    print(f"Trained RLAIF Bradley-Terry weights from {len(feature_pairs)} AI preference pairs: {np.round(rlaif_weights, 4).tolist()}")

    # Evaluate Full RAG Matrix (Models x Mobile Search x 5 RAG Strategies) on ALL queries & Held-Out Test Split
    judge.unload()

    exp_c_rows = []
    rag_models = ["M2_tinyclip_8m_onnx_int8", "M3_tinyclip_40m_fp32", "M5_clip_b32_onnx_int8", "M7_siglip2_base_onnx_int8"]
    for mk in rag_models:
        enc = load_encoder(mk, onnx_dir=ONNX_DIR)
        vis, cap = cached_embs[mk]
        for algo in ["S1_sq8_int8", "S3_bq_rescore", "S6_hybrid_bm25_rrf"]:
            ws = VisualRAGWorkspace(units, vis, cap, search_algo=algo, rlaif_reward_weights=rlaif_weights)
            for rag in rag_modes:
                res_all = evaluate_config(ws, enc, pos_queries, prep_strategy="P3_prompt_template", rag_mode=rag)
                res_test = evaluate_config(ws, enc, test_queries, prep_strategy="P3_prompt_template", rag_mode=rag)
                exp_c_rows.append({
                    "model": mk,
                    "search_algo": algo,
                    "rag_mode": rag,
                    "prep_strategy": "P3_prompt_template",
                    **res_all,
                    "test_recall_at_1": res_test["recall_at_1"],
                    "test_mrr": res_test["mrr"],
                    "test_ndcg_at_5": res_test["ndcg_at_5"],
                })
        enc.unload()

    # 5. Held-out Test Split RLAIF Constitutional Evaluation + AB/BA Pairwise Preference + Grounded RAG Synthesis Check
    print("\n=== Held-Out Test RLAIF Judge & Debiased AB/BA Pairwise Comparison ===")
    enc_rlaif = load_encoder(rlaif_model_key, onnx_dir=ONNX_DIR)
    ws_test_no_rlaif = VisualRAGWorkspace(units, vis_r, cap_r, search_algo="S1_sq8_int8", rlaif_reward_weights=None)
    ws_test_rlaif = VisualRAGWorkspace(units, vis_r, cap_r, search_algo="S1_sq8_int8", rlaif_reward_weights=rlaif_weights)

    test_slates: dict[str, dict[str, Any]] = {}
    for q in test_queries:
        r0 = ws_test_no_rlaif.retrieve(q["query"], enc_rlaif, "P0_raw", "R0_direct_dense", top_k=3)
        r1 = ws_test_no_rlaif.retrieve(q["query"], enc_rlaif, "P3_prompt_template", "R1_caption_hybrid", top_k=3)
        r4_pre = ws_test_no_rlaif.retrieve(q["query"], enc_rlaif, "P3_prompt_template", "R4_adaptive_rlaif_rag", top_k=3)
        r4_post = ws_test_rlaif.retrieve(q["query"], enc_rlaif, "P3_prompt_template", "R4_adaptive_rlaif_rag", top_k=3)
        test_slates[q["id"]] = {
            "query": q,
            "R0_raw_baseline": r0["results"],
            "R1_caption_hybrid": r1["results"],
            "R4_pre_rlaif": r4_pre["results"],
            "R4_post_rlaif": r4_post["results"],
        }
    enc_rlaif.unload()

    # Generate grounded RAG answers with SmolVLM2 on sample test queries before loading Judge
    smolvlm = SmolVLM2Captioner()
    rag_synthesis_samples = []
    for q in test_queries[:5]:
        slate = test_slates[q["id"]]["R4_post_rlaif"]
        ans = smolvlm.synthesize_rag_answer(q["query"], slate, max_images=3)
        rag_synthesis_samples.append({
            "query_id": q["id"],
            "query": q["query"],
            "top_results": [{"media_id": r["media_id"], "score": round(r["score"], 4), "timestamp_sec": r["timestamp_sec"]} for r in slate],
            "synthesized_answer": ans,
        })
    smolvlm.unload()

    # Score Held-Out Test slates using independent Qwen2-VL-2B Judge
    judge = QwenVLMConstitutionalJudge()
    test_rubric_summary: dict[str, list[dict[str, float]]] = {
        "R0_raw_baseline": [],
        "R1_caption_hybrid": [],
        "R4_pre_rlaif": [],
        "R4_post_rlaif": [],
    }
    pairwise_results = []

    for qid, bundle in test_slates.items():
        q = bundle["query"]
        for sys_name in ["R0_raw_baseline", "R1_caption_hybrid", "R4_pre_rlaif", "R4_post_rlaif"]:
            cands = bundle[sys_name]
            if not cands:
                continue
            top1 = cands[0]
            sc1 = judge.score_candidate(
                query=q["query"],
                image_path=BASE_DIR / top1["thumb_path"],
                candidate_media_type=top1["media_type"],
                expected_media_type=q.get("expected_media_type"),
            )
            # Also evaluate Top-3 average utility
            top3_scores = [sc1["overall"]]
            for extra in cands[1:3]:
                sx = judge.score_candidate(
                    query=q["query"],
                    image_path=BASE_DIR / extra["thumb_path"],
                    candidate_media_type=extra["media_type"],
                    expected_media_type=q.get("expected_media_type"),
                )
                top3_scores.append(sx["overall"])
            test_rubric_summary[sys_name].append({
                "top1_entity": sc1["entity"],
                "top1_attribute": sc1["attribute"],
                "top1_media_fit": sc1["media_fit"],
                "top1_overall": sc1["overall"],
                "top3_mean_overall": float(np.mean(top3_scores)),
            })

        # Pairwise AB/BA debiased comparison: R4_post_rlaif (A) vs R0_raw_baseline (B)
        c_a = bundle["R4_post_rlaif"]
        c_b = bundle["R0_raw_baseline"]
        if c_a and c_b:
            if c_a[0]["media_id"] == c_b[0]["media_id"]:
                pw = {"winner": "TIE", "consistent": True, "pass_ab": "SAME", "pass_ba": "SAME"}
            else:
                pw = judge.pairwise_compare_debiased(
                    query=q["query"],
                    img_a_path=BASE_DIR / c_a[0]["thumb_path"],
                    type_a=c_a[0]["media_type"],
                    img_b_path=BASE_DIR / c_b[0]["thumb_path"],
                    type_b=c_b[0]["media_type"],
                )
            pairwise_results.append({
                "query_id": qid,
                "query": q["query"],
                "a_media_id": c_a[0]["media_id"],
                "b_media_id": c_b[0]["media_id"],
                **pw,
            })

    judge.unload()

    rlaif_summary = {
        "judge_model": "Qwen/Qwen2-VL-2B-Instruct (Apache-2.0)",
        "caption_synthesizer_model": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct (Apache-2.0)",
        "train_preference_pairs_count": len(feature_pairs),
        "learned_bradley_terry_weights": [round(float(w), 4) for w in rlaif_weights],
        "held_out_rubric_means": {
            sys_name: {
                k: round(float(np.mean([v[k] for v in vals])), 3)
                for k in ["top1_entity", "top1_attribute", "top1_media_fit", "top1_overall", "top3_mean_overall"]
            }
            for sys_name, vals in test_rubric_summary.items()
            if vals
        },
        "debiased_pairwise_r4_vs_r0": {
            "total_test_queries": len(pairwise_results),
            "r4_wins": sum(1 for p in pairwise_results if p["winner"] == "A"),
            "r0_wins": sum(1 for p in pairwise_results if p["winner"] == "B"),
            "ties_or_swapped_inconsistent": sum(1 for p in pairwise_results if p["winner"] == "TIE"),
            "consistency_rate": round(
                sum(1 for p in pairwise_results if p["consistent"]) / max(len(pairwise_results), 1), 3
            ),
            "details": pairwise_results,
        },
        "train_judge_logs": train_judge_logs,
        "rag_synthesis_samples": rag_synthesis_samples,
    }

    # Export compact Mobile Bundle (SQLite + INT8 embeddings + learned RLAIF weights)
    bundle_meta = export_mobile_bundle(
        units=units,
        visual_embs=vis_r,
        caption_embs=cap_r,
        model_key=rlaif_model_key,
        rlaif_weights=rlaif_weights,
    )

    full_results = {
        "dataset_summary": {
            "total_units": len(units),
            "num_images": sum(1 for u in units if u.media_type == "image"),
            "num_video_keyframes": sum(1 for u in units if u.media_type == "video"),
            "num_eval_queries": len(pos_queries),
            "train_queries": len(train_queries),
            "test_queries": len(test_queries),
        },
        "model_profiles": model_profiles,
        "exp_a_query_preprocessing": exp_a_rows,
        "exp_b_search_algorithms": exp_b_rows,
        "scaling_10k_dim512": scaling_10k_512,
        "scaling_10k_dim768": scaling_10k_768,
        "exp_c_rag_matrix": exp_c_rows,
        "mobile_bundle_meta": bundle_meta,
    }
    (ARTIFACTS_DIR / "benchmark_results.json").write_text(
        json.dumps(full_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "rlaif_feedback.json").write_text(
        json.dumps(rlaif_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(exp_c_rows).to_csv(ARTIFACTS_DIR / "benchmark_results.csv", index=False)
    print("\n=== ALL BENCHMARKS & RLAIF VALIDATION COMPLETE ===")
    print("Held-out RLAIF Rubric Means:", rlaif_summary["held_out_rubric_means"])
    print("Pairwise Win/Loss/Tie (R4 vs R0):", rlaif_summary["debiased_pairwise_r4_vs_r0"])


if __name__ == "__main__":
    main()
