# Multi-Axis Benchmark Methodology & RLAIF Evaluation Protocol

본 문서는 모바일 중심 자연어 사진·영상 검색기(**PocketLens**)의 종합 실험 설계와 **RLAIF(Reinforcement Learning from AI Feedback)** 기반 성능 검증 및 폐루프 최적화 프로토콜을 정의합니다.

---

## 1. 4축 결합 실험 매트릭스 (`Model × QueryPrep × SearchAlgo × RAG`)

실제 모바일 배포 시나리오를 포괄하기 위해 아래 4개 축을 체계적으로 교차 검증합니다.

### 축 1: 멀티모달 인코더 & 양자화 (`src/models/encoders.py`, `src/models/export_quantize.py`)
모두 **100% 상업적 이용이 허용된 오픈 가중치(`MIT`, `Apache-2.0`)**만 채택했습니다.

| 코드 | 모델 ID | 아키텍처 계열 | 다국어 | 추론 런타임 |
|---|---|---|---|---|
| `M1_tinyclip_8m_fp32` | `wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` | 네이티브 초경량 증류 (8M/3M) | 영문 중심 | PyTorch FP32 |
| `M2_tinyclip_8m_onnx_int8` | 위 모델 Dynamic INT8 변환 | 초경량 INT8 양자화 | 영문 중심 | ONNX Runtime CPU |
| `M3_tinyclip_40m_fp32` | `wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M` | 네이티브 경량 증류 (40M/19M) | 영문 중심 | PyTorch FP32 |
| `M4_clip_b32_fp32` | `openai/clip-vit-base-patch32` | 표준 베이스라인 (~151M) | 영문 중심 | PyTorch FP32 |
| `M5_clip_b32_onnx_int8` | 위 모델 Dynamic INT8 변환 | 무거운 모델 양자화 압축 | 영문 중심 | ONNX Runtime CPU |
| `M6_siglip2_base_fp32` | `google/siglip2-base-patch16-224` | SigLIP 2 (86M ViT-B + 다국어) | 한·영 네이티브 | PyTorch FP32 |
| `M7_siglip2_base_onnx_int8` | 위 모델 Dynamic INT8 변환 | 다국어 모바일 INT8 | 한·영 네이티브 | ONNX Runtime CPU |

### 축 2: 자연어 전처리 전략 (`src/preprocessing/query_processor.py`)
- `P0_raw`: 사용자 원문 그대로 인코딩 (대화형 불용어·매체 제약 포함)
- `P1_intent_clean`: 비시각적 시간/의도 어구 제거 + 매체(`image`/`video`) 필터 분리 + 구문 유지
- `P2_keywords`: 명사/형용사 핵심 키워드만 추출 (문맥·구문 손실 검증)
- `P3_prompt_template`: `a photo of {core}` / `a video frame showing {core}` 정규화 (+ 경량 한영 브리지)
- `P4_multi_aspect_expansion`: 다중 시각 속성 패러프레이즈 앙상블 임베딩

### 축 3: 모바일 지향 검색 알고리즘 (`src/indexers/search_algorithms.py`)
- `S0_flat_fp32`: 정확 코사인 유사도 (Recall 100% 기준선)
- `S1_sq8_int8`: 스칼라 INT8 양자화 (4× RAM 압축)
- `S2_bq_hamming`: 1-bit Binary Quantization + POPCNT 해밍 거리 (32× RAM 압축)
- `S3_bq_rescore`: 2단계 모바일 검색 (1-bit BQ 후보 추출 → Top-K FP32 정밀 재채점)
- `S4_hnsw`: HNSW 근접 그래프 (`M=16, ef=48`)
- `S5_ivfpq`: FAISS IVF-PQ (클러스터 + 부분공간 코드북 압축)
- `S6_hybrid_bm25_rrf`: 밀집 벡터 + BM25 어휘 검색의 Reciprocal Rank Fusion (`k=60`)
- 추가 스트레스 테스트: **스마트폰 갤러리 10,000장(`N=10,000`, `dim=512 & 768`)** 기준 RAM(MB), Recall@10, P50/P95 지연(ms) 실측.

### 축 4: 적용 가능 Visual RAG 전략 (`src/rag/pipelines.py`)
- `R0_direct_dense`: 캡션 없이 순수 이미지/키프레임 임베딩 검색
- `R1_caption_hybrid`: 경량 VLM(`SmolVLM2-500M-Video-Instruct`) 오프라인 캡션/태그 병합 하이브리드 RRF
- `R2_two_stage_rerank`: 양자화 1차 검색 → 정밀 시각+캡션 다중 신호 재정렬
- `R3_video_temporal`: 적응형 비디오 키프레임 추출 + Top-M 시간 창 풀링(`[t_start, t_end]` 구간 반환)
- `R4_adaptive_rlaif_rag`: 질의 의도 분해(매체 필터) + 하이브리드 검색 + **RLAIF 학습 보상 가중치 기반 재정렬** + VLM 근거 요약 합성

---

## 2. RLAIF 성능 검증 & 폐루프 보상 최적화 프로토콜 (`src/eval/rlaif_judge.py`)

### (1) 독립 헌법적(Constitutional) VLM Judge 구성
- `Survey on LLM-as-a-Judge` (`arXiv:2411.15594`)가 경고한 **Self-Enhancement Bias(자가 강화 편향)** 를 방지하기 위해, RAG 캡션/응답 생성 모델(`SmolVLM2-500M-Video-Instruct`)과 완전히 다른 모델 계열인 **`Qwen/Qwen2-VL-2B-Instruct` (Apache-2.0)** 를 독립 RLAIF 심판으로 사용합니다.
- 4대 헌법적 채점 원칙(각 0~5점):
  1. **Visual Grounding:** 질의가 요구한 핵심 개체가 이미지/프레임에 실제로 보이는가?
  2. **Action & Attribute Precision:** 색상, 동작, 배경, 조명, 공간 관계가 일치하는가?
  3. **Media Type & Intent Fit:** 동영상/사진 요청 의도와 후보 미디어 유형이 부합하는가?
  4. **Overall Utility:** 실제 모바일 사용자 입장에서 상위 검색 결과로 만족스러운가?

### (2) 위치 편향 제거 쌍대 비교 (AB / BA Position Swap)
- `MLLM-as-a-Judge` (`arXiv:2402.04788`)에 따라 동일 질의에 대해 후보 `(A, B)` 순서와 `(B, A)` 순서를 모두 평가합니다. 두 순서에서 일관되게 선택된 경우만 승/패로 인정하고, 불일치 시 `TIE (Position-Ambiguous)`로 처리합니다.

### (3) 폐루프 Bradley-Terry 보상 모델 학습 (Train / Held-Out Test Split)
- 전체 평가 질의를 **Train (60%)** 와 **Held-Out Test (40%)** 로 엄격히 분리합니다.
- Train 질의에서 VLM Judge가 0.8점 이상 차이를 보인 `(chosen, rejected)` 쌍으로부터 7차원 교차 모달 특징 벡터 `[v_sim, c_sim, rrf_v, rrf_c, rrf_h, kw_overlap, media_bonus]`에 대한 **Bradley-Terry 로지스틱 선호 보상 모델**을 학습합니다:
  $$P(\text{chosen} \succ \text{rejected}) = \sigma\left(w^\top (x_{\text{chosen}} - x_{\text{rejected}})\right)$$
- 학습된 가중치 $w$를 `R4_adaptive_rlaif_rag`에 탑재해, 전혀 보지 못한 **Held-Out Test 질의**에서 RLAIF 이전(`R4_pre_rlaif`)과 Raw 베이스라인(`R0_raw_baseline`) 대비 실제 개선폭을 검증합니다.

---

## 3. 실측 검증 결과 요약

- **AI 선호 쌍 수집 및 학습:** Train Split(16개 질의)에서 `Qwen/Qwen2-VL-2B-Instruct`가 0.8점 이상 차이를 판별한 **80쌍**의 선호 데이터를 확보해 Bradley-Terry 가중치 `w = [0.2405 (v_sim), 0.3817 (c_sim), 0.0752 (rrf_v), 0.1713 (rrf_c), 0.2226 (rrf_h), 0.3550 (kw_overlap), 0.4874 (media_bonus)]`를 학습했습니다.
- **Held-Out Test 성능 도약 (`S1_sq8_int8` 모바일 인덱스 기준):**
  - `M2_tinyclip_8m_onnx_int8` (`22.96 MB`): `R0 (MRR 0.9062 / nDCG@5 0.9071)` → **`R4_adaptive_rlaif_rag (MRR 1.0000 / nDCG@5 0.9917)`**
  - `M5_clip_b32_onnx_int8` (`145.68 MB`): `R0 (MRR 0.9688 / nDCG@5 0.9571)` → **`R4_adaptive_rlaif_rag (MRR 1.0000 / nDCG@5 0.9929)`**
  - `M7_siglip2_base_onnx_int8` (`364.48 MB`): `R0 (MRR 0.8438 / nDCG@5 0.8114)` → **`R4_adaptive_rlaif_rag (MRR 0.9688 / nDCG@5 0.9574)`**
- **VLM 헌법적 루브릭(Top-3 Mean Overall, 0~5점):** `3.396 (R0)` → `3.542 (R4 Pre-RLAIF)` → **`3.646 (R4 Post-RLAIF)`**
- **AB/BA 위치 스왑 편향 탐지 결과:** `Qwen2-VL-2B`는 서로 다른 2개 이미지를 나란히 제시한 5개 테스트 질의(`q04, q08, q10, q26, q32`) 모두에서 순서와 무관하게 첫 번째(`1`)를 선택하는 **Position-1 Primacy Bias**를 보였고(`consistency_rate = 68.8%`), AB/BA 스왑 가드가 이를 정확히 검출·무효화(`TIE`) 처리했습니다.

