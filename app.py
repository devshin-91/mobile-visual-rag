"""
PocketLens — Mobile-First Visual RAG (Streamlit PC Prototype + Mobile Viewport Simulator)
Design synthesized from Apple HIG (Photos), Google Material Design 3 (Search), and Microsoft Fluent 2 (SearchBox).
Uses commercially safe assets: Pretendard & Inter (SIL OFL 1.1) and Lucide SVG Icons (ISC/MIT).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.encoders import ENCODER_SPECS, load_encoder
from src.rag.pipelines import UnitRecord, VisualRAGWorkspace
from src.utils.win_asyncio import install_windows_proactor_guard, resolve_playable_video

install_windows_proactor_guard()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DOCS_DIR = BASE_DIR / "docs"
ONNX_DIR = BASE_DIR / "exports" / "onnx"
EMB_CACHE_DIR = ARTIFACTS_DIR / "emb_cache"

st.set_page_config(
    page_title="PocketLens · Mobile Visual RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Pretendard + Inter (SIL OFL 1.1) and Material 3 / Apple HIG clean styling
st.markdown(
    """
    <style>
      @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      html, body, [class*="css"] {
        font-family: 'Pretendard', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      .pill-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(37, 99, 235, 0.12);
        color: #2563EB;
        margin-right: 6px;
        margin-bottom: 4px;
      }
      .credit-badge {
        font-size: 11px;
        color: #64748B;
        line-height: 1.4;
      }
      .mobile-frame {
        max-width: 410px;
        margin: 0 auto;
        border: 2px solid rgba(148, 163, 184, 0.35);
        border-radius: 30px;
        padding: 18px 16px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.14);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_artifacts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    if (DATA_DIR / "manifest_verified.json").exists():
        out["manifest"] = json.loads((DATA_DIR / "manifest_verified.json").read_text(encoding="utf-8"))
    if (DATA_DIR / "video_keyframes.json").exists():
        out["keyframes"] = json.loads((DATA_DIR / "video_keyframes.json").read_text(encoding="utf-8"))
    if (ARTIFACTS_DIR / "captions_cache.json").exists():
        out["captions"] = json.loads((ARTIFACTS_DIR / "captions_cache.json").read_text(encoding="utf-8"))
    if (ARTIFACTS_DIR / "benchmark_results.json").exists():
        out["bench"] = json.loads((ARTIFACTS_DIR / "benchmark_results.json").read_text(encoding="utf-8"))
    if (ARTIFACTS_DIR / "rlaif_feedback.json").exists():
        out["rlaif"] = json.loads((ARTIFACTS_DIR / "rlaif_feedback.json").read_text(encoding="utf-8"))
    if (ARTIFACTS_DIR / "onnx_quantization_report.json").exists():
        out["onnx"] = json.loads((ARTIFACTS_DIR / "onnx_quantization_report.json").read_text(encoding="utf-8"))
    if (DOCS_DIR / "verified_references.json").exists():
        out["refs"] = json.loads((DOCS_DIR / "verified_references.json").read_text(encoding="utf-8"))
    return out


@st.cache_resource
def get_search_engine(model_key: str, search_algo: str, rlaif_weights_tuple: tuple[float, ...] | None):
    manifest = json.loads((DATA_DIR / "manifest_verified.json").read_text(encoding="utf-8"))
    keyframes = json.loads((DATA_DIR / "video_keyframes.json").read_text(encoding="utf-8"))
    captions = json.loads((ARTIFACTS_DIR / "captions_cache.json").read_text(encoding="utf-8"))
    by_id = {m["media_id"]: m for m in manifest}

    units: list[UnitRecord] = []
    for m in manifest:
        if m["media_type"] == "image":
            cap = captions.get(m["media_id"], m.get("description", ""))
            units.append(
                UnitRecord(
                    unit_id=m["media_id"],
                    media_id=m["media_id"],
                    media_type="image",
                    title=m["canonical_title"],
                    theme=m["theme"],
                    caption=cap,
                    searchable_text=f"{m['theme']} {m['canonical_title']} {m.get('description','')} {cap}",
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
        units.append(
            UnitRecord(
                unit_id=kf["frame_id"],
                media_id=m["media_id"],
                media_type="video",
                title=m["canonical_title"],
                theme=m["theme"],
                caption=cap,
                searchable_text=f"video {m['theme']} {m['canonical_title']} {m.get('description','')} {cap}",
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

    npz = np.load(EMB_CACHE_DIR / f"{model_key}.npz")
    enc = load_encoder(model_key, onnx_dir=ONNX_DIR)
    weights = np.array(rlaif_weights_tuple, dtype=np.float32) if rlaif_weights_tuple else None
    ws = VisualRAGWorkspace(units, npz["visual"], npz["caption"], search_algo=search_algo, rlaif_reward_weights=weights)
    return enc, ws


arts = load_artifacts()

st.title("🔍 PocketLens · 모바일 중심 자연어 사진·영상 Visual RAG")
st.caption("경량/양자화 멀티모달 인코더 × 자연어 전처리 × 모바일 벡터 검색 × 5종 Visual RAG × RLAIF 검증")

tabs = st.tabs([
    "📱 자연어 검색 & 모바일 프리뷰",
    "📊 종합 벤치마크 (Model × 전처리 × 검색 × RAG)",
    "⚖️ RLAIF 검증 & 폐루프 보상 최적화",
    "📜 데이터 크레딧 & 참고문헌 (2중 팩트체크)",
])

# === TAB 1: Interactive Search & Mobile Preview ===
with tabs[0]:
    with st.sidebar:
        st.header("⚙️ 검색 엔진 설정")
        available_models = [k for k in ENCODER_SPECS if (EMB_CACHE_DIR / f"{k}.npz").exists()]
        default_idx = available_models.index("M7_siglip2_base_onnx_int8") if "M7_siglip2_base_onnx_int8" in available_models else 0
        sel_model = st.selectbox("멀티모달 인코더", available_models, index=default_idx) if available_models else None
        sel_prep = st.selectbox(
            "자연어 전처리 전략",
            ["P3_prompt_template", "P4_multi_aspect_expansion", "P1_intent_clean", "P2_keywords", "P0_raw"],
        )
        sel_algo = st.selectbox(
            "검색 알고리즘",
            ["S1_sq8_int8", "S3_bq_rescore", "S6_hybrid_bm25_rrf", "S0_flat_fp32", "S2_bq_hamming", "S4_hnsw", "S5_ivfpq"],
        )
        sel_rag = st.selectbox(
            "Visual RAG 전략",
            ["R4_adaptive_rlaif_rag", "R1_caption_hybrid", "R2_two_stage_rerank", "R3_video_temporal", "R0_direct_dense"],
        )
        media_scope = st.radio("미디어 범위 (Scope Bar)", ["전체", "사진만", "동영상만"], horizontal=True)
        mobile_sim = st.toggle("📱 모바일 뷰포트 미리보기 (390px)", value=False)
        top_k = st.slider("Top-K 결과 수", 3, 12, 6)

    if not sel_model:
        st.warning("임베딩 캐시가 아직 생성되지 않았습니다. 먼저 벤치마크 파이프라인을 실행해주세요.")
    else:
        rlaif_w = tuple(arts.get("rlaif", {}).get("learned_bradley_terry_weights", [])) or None
        enc, ws = get_search_engine(sel_model, sel_algo, rlaif_w)

        sample_queries = [
            "지난 주말에 찍은 해변 노을 사진 좀 찾아줘",
            "물웅덩이에서 물 마시는 고양이 동영상 보여줘",
            "따뜻한 카페 라떼 커피 사진",
            "밤거리 야경 동영상",
            "자동차와 자전거가 지나가는 도로 영상",
            "우주에서 본 푸른 지구 사진",
        ]
        st.markdown("**빠른 검색 제안 (Apple HIG / Fluent 2 Pre-Search Pattern):**")
        q_cols = st.columns(3)
        chosen_chip = None
        for idx, sq in enumerate(sample_queries):
            if q_cols[idx % 3].button(sq, width="stretch"):
                chosen_chip = sq

        if chosen_chip:
            st.session_state["search_query"] = chosen_chip
        query = st.text_input(
            "자연어 검색어 입력 (한국어 · 영어 · 대화형 모두 가능)",
            value=st.session_state.get("search_query", "물웅덩이에서 물 마시는 고양이 동영상 보여줘"),
            key="query_box",
        )
        st.session_state["search_query"] = query

        explicit_filter = {"전체": None, "사진만": "image", "동영상만": "video"}[media_scope]
        res = ws.retrieve(
            raw_query=query,
            encoder=enc,
            prep_strategy=sel_prep,
            rag_mode=sel_rag,
            top_k=top_k,
            explicit_media_filter=explicit_filter,
        )

        prep_info = res["preprocessing"]
        st.markdown(
            f'<span class="pill-tag">추출 핵심어: {prep_info["core_text"]}</span>'
            f'<span class="pill-tag">매체 필터: {prep_info["media_type_filter"] or "전체"}</span>'
            f'<span class="pill-tag">시간 힌트 제거: {prep_info["temporal_hint"] or "없음"}</span>'
            f'<span class="pill-tag">검색 지연: {res["latency_ms"]} ms</span>'
            f'<span class="pill-tag">인덱스 RAM: {res["index_ram_kb"]} KB</span>',
            unsafe_allow_html=True,
        )

        def _render_cards(results_list: list[dict[str, Any]], cols_count: int) -> None:
            cols = st.columns(cols_count)
            for idx, r in enumerate(results_list):
                with cols[idx % cols_count]:
                    with st.container(border=True):
                        if r["media_type"] == "video":
                            st.image(str(BASE_DIR / r["thumb_path"]), width="stretch")
                            st.caption(
                                f"🎬 **#{idx+1} [{r['score']:.3f}]** · ⏱️ 매칭 구간: `{r['window_start_sec']}s ~ {r['window_end_sec']}s`"
                            )
                            if st.toggle("▶️ 영상 재생", key=f"play_{r['media_id']}_{idx}"):
                                play_rel, mime_fmt = resolve_playable_video(r["media_path"], BASE_DIR)
                                st.video(
                                    str(BASE_DIR / play_rel),
                                    format=mime_fmt,
                                    start_time=int(max(0, (r["timestamp_sec"] or 0))),
                                )
                        else:
                            st.image(str(BASE_DIR / r["thumb_path"]), width="stretch")
                            st.caption(f"🖼️ **#{idx+1} [{r['score']:.3f}]** · {r['title'][:40]}")
                        st.markdown(
                            f'<div class="credit-badge"><b>캡션:</b> {r["caption"][:110]}...<br>'
                            f'<b>출처:</b> <a href="{r["source_page_url"]}" target="_blank">{r["source_platform"]}</a> · '
                            f'<a href="{r["license_url"]}" target="_blank">{r["license_short"]}</a> ({r["author"][:28]})</div>',
                            unsafe_allow_html=True,
                        )

        if mobile_sim:
            _, mid_col, _ = st.columns([1, 1.3, 1])
            with mid_col:
                st.markdown('<div class="mobile-frame">', unsafe_allow_html=True)
                st.markdown("##### 📱 PocketLens Mobile Preview (390px)")
                _render_cards(res["results"], cols_count=2)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            _render_cards(res["results"], cols_count=3)

# === TAB 2: Full Benchmarks ===
with tabs[1]:
    bench = arts.get("bench")
    if not bench:
        st.info("벤치마크 결과(`artifacts/benchmark_results.json`)가 아직 없습니다.")
    else:
        st.subheader("1. 경량 모델 & ONNX INT8 양자화 프로파일")
        df_prof = pd.DataFrame(bench["model_profiles"])
        if arts.get("onnx"):
            df_onnx = pd.DataFrame(arts["onnx"])
            c1, c2 = st.columns(2)
            c1.dataframe(df_onnx, width="stretch")
            fig_q = px.bar(
                df_onnx,
                x="base_model",
                y=["total_fp32_mb", "total_int8_mb"],
                barmode="group",
                title="FP32 vs ONNX INT8 모델 용량 비교 (MB)",
            )
            c2.plotly_chart(fig_q, width="stretch")
        st.dataframe(df_prof, width="stretch")

        st.subheader("2. 자연어 전처리 전략 비교 (통문장 P0 vs 의도분리 P1 vs 키워드 P2 vs 프롬프트정규화 P3 vs 다각확장 P4)")
        df_a = pd.DataFrame(bench["exp_a_query_preprocessing"])
        df_a_r1 = df_a[df_a["rag_mode"] == "R1_caption_hybrid"]
        fig_a = px.bar(
            df_a_r1,
            x="model",
            y="mrr",
            color="prep_strategy",
            barmode="group",
            title="모델별 자연어 전처리 전략에 따른 MRR (R1 Hybrid RAG 기준)",
        )
        st.plotly_chart(fig_a, width="stretch")
        st.dataframe(df_a[["model", "prep_strategy", "rag_mode", "recall_at_1", "recall_at_5", "mrr", "ndcg_at_5"]], width="stretch")

        st.subheader("3. 모바일 10,000장 규모 벡터 검색 알고리즘 스케일링 (RAM vs Recall@10 vs Latency)")
        df_10k = pd.DataFrame(bench["scaling_10k_dim768"])
        c1, c2 = st.columns(2)
        fig_ram = px.scatter(
            df_10k,
            x="ram_mb",
            y="recall_at_10_vs_exact",
            size="latency_p50_ms",
            color="algorithm",
            text="algorithm",
            title="10,000장 갤러리 기준 인덱스 RAM(MB) vs Recall@10 (768d)",
        )
        c1.plotly_chart(fig_ram, width="stretch")
        c2.dataframe(df_10k, width="stretch")

        st.subheader("4. 통합 조합 매트릭스 (Encoder × Search × 5종 Visual RAG)")
        df_c = pd.DataFrame(bench["exp_c_rag_matrix"])
        sel_bench_algo = st.selectbox("RAG 매트릭스 검색 백엔드 선택", sorted(df_c["search_algo"].unique().tolist()), key="bench_algo_sel")
        fig_c = px.bar(
            df_c[df_c["search_algo"] == sel_bench_algo],
            x="model",
            y="ndcg_at_5",
            color="rag_mode",
            barmode="group",
            title=f"{sel_bench_algo} 모바일 검색 백엔드에서의 5종 Visual RAG nDCG@5 비교",
        )
        st.plotly_chart(fig_c, width="stretch")
        st.dataframe(
            df_c[["model", "search_algo", "rag_mode", "recall_at_1", "recall_at_5", "mrr", "ndcg_at_5", "test_mrr", "search_latency_ms"]],
            width="stretch",
        )

# === TAB 3: RLAIF Evaluation ===
with tabs[2]:
    rlaif = arts.get("rlaif")
    if not rlaif:
        st.info("RLAIF 검증 결과(`artifacts/rlaif_feedback.json`)가 아직 없습니다.")
    else:
        st.subheader("⚖️ 독립 VLM Judge 기반 RLAIF 검증 & 폐루프 보상 최적화")
        st.markdown(
            f"- **AI Judge 모델 (Self-Enhancement Bias 차단):** `{rlaif['judge_model']}`\n"
            f"- **RAG 캡션/응답 합성 모델:** `{rlaif['caption_synthesizer_model']}`\n"
            f"- **Train AI 선호 쌍 수:** `{rlaif['train_preference_pairs_count']}`쌍\n"
            f"- **학습된 Bradley-Terry 가중치:** `{rlaif['learned_bradley_terry_weights']}`"
        )
        rubric_df = pd.DataFrame(rlaif["held_out_rubric_means"]).T.reset_index().rename(columns={"index": "pipeline"})
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.bar(
                rubric_df,
                x="pipeline",
                y=["top1_entity", "top1_attribute", "top1_media_fit", "top1_overall", "top3_mean_overall"],
                barmode="group",
                title="Held-Out Test Split RLAIF Constitutional Rubric 점수 (0~5)",
            ),
            width="stretch",
        )
        pw = rlaif["debiased_pairwise_r4_vs_r0"]
        c2.metric("R4 (RLAIF 적용) vs R0 (Raw) 승률", f"{pw['r4_wins']} 승 / {pw['ties_or_swapped_inconsistent']} 무 / {pw['r0_wins']} 패")
        c2.metric("AB/BA 위치 스왑 일관성 (Position-Bias Check)", f"{pw['consistency_rate']*100:.1f}%")
        c2.dataframe(pd.DataFrame(pw["details"]), width="stretch")

        st.subheader("SmolVLM2 기반 Grounded Visual RAG 응답 예시")
        for s in rlaif.get("rag_synthesis_samples", []):
            with st.expander(f"질의: {s['query']}"):
                st.write("**합성 응답:**", s["synthesized_answer"])
                st.json(s["top_results"])

# === TAB 4: Credits & References ===
with tabs[3]:
    st.subheader("📜 상업적 이용 가능 데이터셋 & 문헌 2중 팩트체크 내역")
    if arts.get("manifest"):
        df_m = pd.DataFrame(arts["manifest"])
        st.dataframe(
            df_m[["media_id", "media_type", "theme", "canonical_title", "author", "license_short", "source_platform", "fact_check_count"]],
            width="stretch",
        )
    if arts.get("refs"):
        st.markdown("### 학술 논문 레퍼런스 (arXiv API 2중 팩트체크)")
        df_p = pd.DataFrame(arts["refs"]["arxiv_papers"])
        st.dataframe(df_p[["key", "arxiv_id", "published", "title", "topic", "url", "verified"]], width="stretch")
