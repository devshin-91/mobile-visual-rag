# PocketLens: 모바일 중심 자연어 사진·영상 검색기 (Visual RAG & RLAIF)

스마트폰 갤러리처럼 **사진과 동영상이 섞인 개인 미디어 컬렉션**을 자연어(한국어·영어·대화형 문장)로 검색하고, 모바일 온디바이스(ONNX INT8 + SQLite 경량 인덱스) 환경까지 즉시 배포할 수 있도록 설계·벤치마크한 컴퓨터 비전 & Visual RAG 프로젝트입니다.

---

## 1. 주요 특징 및 설계 철학

- **모바일 우선(Mobile-First) 경량화 & 양자화:**
  - 네이티브 초경량 모델(`TinyCLIP-ViT-8M`, `TinyCLIP-ViT-40M`)과 대형 모델의 Dynamic INT8 양자화(`CLIP-ViT-B/32`, `SigLIP2-Base`)를 ONNX Runtime CPU 환경에서 직접 실측 비교했습니다.
  - 상업적 배포가 가능한 **`MIT` / `Apache-2.0`** 오픈 가중치만 채택했습니다(`apple-amlr` 및 `cc-by-nc-4.0` 모델은 감사 후 상용 런타임 스택에서 제외).
- **문헌 기반 자연어 전처리 5종 비교 ("통문장 vs 전처리"):**
  - `P0_raw`, `P1_intent_clean`, `P2_keywords`, `P3_prompt_template`, `P4_multi_aspect_expansion`의 5단계 전처리를 구현해, 구문 유지·대화형 노이즈 분리·키워드화의 장단점을 모델 계보별로 실증했습니다.
- **모바일 벡터 검색 7종 × Visual RAG 5종 통합 매트릭스:**
  - `Flat FP32`, `SQ8 INT8`, `Binary Quantization(1-bit)`, `BQ + Top-K Rescore`, `HNSW`, `IVF-PQ`, `Hybrid Dense + BM25 (RRF)`를 실측하고, **10,000장 모바일 갤러리 스케일링**을 별도 프로파일링했습니다.
  - 비디오는 **중앙값 배경 대비 전경 활동 피크 검출(Foreground Activity Peak Detection)** 로 핵심 키프레임을 추출해 매칭 타임스탬프(`[t_start, t_end]`)를 반환합니다.
- **독립 VLM 기반 RLAIF 검증 & 폐루프 보상 학습:**
  - 캡션/응답 모델(`SmolVLM2-500M-Video-Instruct`)과 분리된 독립 VLM(`Qwen/Qwen2-VL-2B-Instruct`)으로 Train Split(16개 질의)에서 **80쌍의 AI 선호 데이터**를 수집하고 Bradley-Terry 보상 가중치를 학습한 뒤, Held-Out Test Split(16개 질의)에서 검증했습니다.
- **PC 프로토타입(Streamlit) + 모바일 PWA(FastAPI) + HTML 보고서:**
  - Apple HIG, Material Design 3, Microsoft Fluent 2 검색 가이드라인을 벤치마크하고, `Lucide Icons(ISC/MIT)`·`Pretendard/Inter(SIL OFL 1.1)`로 UI를 구축했습니다.

---

## 2. 핵심 실측 결과 요약

### A. 경량 & ONNX INT8 양자화 프로파일
| 모델 | 라이선스 | FP32 크기 | ONNX INT8 크기 | 압축률 | CPU Text P50 |
|---|---|---|---|---|---|
| `TinyCLIP-ViT-8M` (`M1→M2`) | `MIT` | `89.71 MB` | **`22.96 MB`** | `3.91×` | **`6.45 ms`** |
| `CLIP-ViT-B/32` (`M4→M5`) | `MIT` | `577.53 MB` | `145.68 MB` | `3.96×` | `54.51 ms` |
| `SigLIP2-Base` (`M6→M7`) | `Apache-2.0` | `1431.73 MB` | `364.48 MB` | `3.93×` | `68.95 ms` |

### B. 통합 RAG 벤치마크 (`S1_sq8_int8` 모바일 인덱스 기준, 32개 한·영 질의)
| 모델 | RAG 전략 | 전체 Recall@1 | 전체 MRR | 전체 nDCG@5 | Held-Out Test MRR | Test nDCG@5 |
|---|---|---|---|---|---|---|
| `M2_tinyclip_8m_onnx_int8` (`22.96 MB`) | `R0_direct_dense` | `0.7812` | `0.8542` | `0.8515` | `0.9062` | `0.9071` |
| `M2_tinyclip_8m_onnx_int8` (`22.96 MB`) | **`R4_adaptive_rlaif_rag`** | **`0.9375`** | **`0.9557`** | **`0.9624`** | **`1.0000`** | **`0.9917`** |
| `M5_clip_b32_onnx_int8` (`145.68 MB`) | **`R4_adaptive_rlaif_rag`** | **`0.9688`** | **`0.9844`** | **`0.9849`** | **`1.0000`** | **`0.9929`** |
| `M7_siglip2_base_onnx_int8` (`364.48 MB`) | `R0_direct_dense` | `0.8438` | `0.8656` | `0.8241` | `0.8438` | `0.8114` |
| `M7_siglip2_base_onnx_int8` (`364.48 MB`) | **`R4_adaptive_rlaif_rag`** | **`0.9375`** | **`0.9635`** | **`0.9605`** | **`0.9688`** | **`0.9574`** |

### C. 모바일 10,000장 스케일링 (`dim=512` / `dim=768`)
- **`S1_sq8_int8`:** `4.902 MB` (`512d`) / `7.343 MB` (`768d`)로 FP32 대비 **4× 메모리 절감**, Recall@10 **`0.9967` / `0.9900`** 유지.
- **`S3_bq_rescore`:** 핫 RAM **`0.610 MB` (`512d`) / `0.916 MB` (`768d`)** 로 **32× 메모리 절감**, 실제 코퍼스에서는 Top-K 재채점으로 `S0_flat_fp32`와 동일한 MRR 복원.

---

## 3. 프로젝트 구조 및 산출물 문서

- **실행형 UI 및 서버:**
  - [`app.py`](app.py) — Streamlit PC 프로토타입 + 모바일 390px 뷰포트 프리뷰 + 인터랙티브 벤치마크/RLAIF 대시보드
  - [`mobile_web/index.html`](mobile_web/index.html) & [`src/mobile_bridge/api_server.py`](src/mobile_bridge/api_server.py) — 스마트폰 브라우저 LAN 접속용 모바일 PWA
  - [`report.html`](report.html) — 차트·표·크레딧·문헌이 포함된 단독 HTML 리포트
- **상세 기술 문서 (`docs/`):**
  - [`docs/LITERATURE_AND_QUERY_PREPROCESSING.md`](docs/LITERATURE_AND_QUERY_PREPROCESSING.md) — 26편 arXiv 논문 조사 & 자연어 전처리 이론·실측 결과
  - [`docs/BENCHMARK_METHODOLOGY_AND_RLAIF.md`](docs/BENCHMARK_METHODOLOGY_AND_RLAIF.md) — 4축 실험 설계 및 RLAIF 검증 프로토콜
  - [`docs/UI_UX_DESIGN_BENCHMARK.md`](docs/UI_UX_DESIGN_BENCHMARK.md) — Apple HIG · Material Design 3 · Fluent 2 비교 및 상용 UI 에셋 감사
  - [`docs/MOBILE_DEPLOYMENT_GUIDE.md`](docs/MOBILE_DEPLOYMENT_GUIDE.md) — ONNX Runtime Mobile(Android/iOS) 및 SQLite 번들 배포 가이드
  - [`CREDITS.md`](CREDITS.md) & [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) — 2중 팩트체크 완료된 61개 오픈 라이선스 미디어(사진 51·영상 10) 및 모델/에셋 저작권 표기

---

## 4. 빠른 실행 방법 (Windows PowerShell)

```powershell
# 1) Streamlit PC 프로토타입 & 벤치마크 탐색기 실행
.\.venv\Scripts\streamlit.exe run app.py

# 2) 모바일 PWA 브리지 서버 실행 (PC & 같은 Wi-Fi 스마트폰 접속 가능)
.\.venv\Scripts\uvicorn.exe src.mobile_bridge.api_server:app --host 0.0.0.0 --port 8000

# 3) 단위 테스트 실행
.\.venv\Scripts\python.exe -c "from tests.test_core import *; test_query_preprocessing_strategies(); test_mobile_search_algorithms(); test_rlaif_parse_and_bradley_terry(); print('OK')"

# 4) HTML 보고서 재생성
.\.venv\Scripts\python.exe -m scripts.generate_html_report
```

---

## 5. 라이선스 (Multi-Part License & Commercial Compliance)

본 프로젝트는 AI/ML 엔지니어링 실무 표준에 따라 **소스 코드 · 기술 문서 · 제3자 에셋의 라이선스 적용 범위를 명확히 분리(Multi-Part Licensing)** 하여 배포합니다.

- **원본 소스 코드 (`src/`, `app.py`, `mobile_web/`, `scripts/`, `tests/`):** [**Apache License 2.0**](LICENSE) (`Copyright 2026 Chansu Shin <devshin91@gmail.com>`)
- **벤치마크 리포트 및 기술 문서 (`docs/`, `report.html`, `README.md`):** [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/) (출처 표기 시 인용·공유 허용)
- **제3자 오픈 라이선스 미디어·모델 가중치·UI 에셋 (`data/`, `exports/onnx/`):** 재라이선스(Relicense) 대상이 아니며, [`NOTICE`](NOTICE), [`CREDITS.md`](CREDITS.md), [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)에 2중 팩트체크된 각 원 라이선스(`CC0 1.0`, `Public Domain`, `CC BY 3.0`, `CC BY 4.0`, `Apache-2.0`, `MIT`, `ISC`, `SIL OFL 1.1`)를 따릅니다.
