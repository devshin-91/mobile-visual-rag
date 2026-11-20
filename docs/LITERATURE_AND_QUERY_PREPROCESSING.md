# 문헌 조사, 자연어 전처리 이론 & 2중 팩트체크 레퍼런스

본 문서는 **모바일 중심 자연어 사진·영상 검색기(Visual RAG)** 설계를 위해 arXiv API 및 공식 GitHub/HuggingFace API를 통해 **2중 팩트체크**한 26편의 학술 논문과 아키텍처 근거를 정리합니다.

---

## 1. 핵심 질문: "자연어 질의는 통째로 넣어야 할까, 전처리해야 할까?"

선행 연구들을 교차 검증한 결과, **"무조건 통째로"도, "무조건 키워드화"도 정답이 아니며, 질의의 성격과 인코더 특성에 따라 3가지 상충 효과가 발생**합니다.

### (1) 순수 키워드 추출(Bag-of-Words)이 실패하는 이유
- **CLIP (`arXiv:2103.00020`, Radford et al., 2021)** 및 **VisRet (`arXiv:2505.20291`, Wu et al., 2025)** 에 따르면, 듀얼 인코더(Dual-Encoder) 텍스트 타워는 자연어 문맥(구문, 수식어, 동작 관계)을 기반으로 학습되었으나, 여전히 **개념들의 가방(bag of concepts)** 처럼 동작하는 취약점이 존재합니다.
- 이 상태에서 문장을 단순 명사 키워드(`"강아지, 해변"`)로 쪼개면 행동(`"뛰어노는"`), 공간 관계(`"위에 있는"`), 시점 정보까지 소실되어 오탐(false positive)이 증가합니다.

### (2) 반대로 "생(Raw) 통문장"이 실패하는 4가지 실전 모바일 원인
1. **대화형·비시각적 노이즈 (Conversational Filler & Metadata):**
   - 스마트폰 갤러리 검색에서는 `"지난 주말에 찍은 해변 강아지 동영상 좀 찾아줘"`처럼 **비시각적 시간/의도 어구**(`지난 주말에`, `좀 찾아줘`)와 **매체 제약**(`동영상`)이 섞여 들어옵니다.
   - **PlugIR (`arXiv:2406.03411`, Lee et al., 2024)** 및 **InteractiveQR (`arXiv:2404.18746`, Zhu et al., 2024)** 는 대화형·장황한 질의가 어휘 불일치(vocabulary mismatch)와 노이즈를 유발하므로, 질의를 시각적 핵심 표현으로 재작성(Query Rewriting)해야 검색 재현율(Recall)이 향상됨을 실증했습니다.
2. **토큰 길이 병목 (77-Token Cap & Effective <20 Tokens):**
   - **Long-CLIP (`arXiv:2403.15378`, Zhang et al., 2024)** 은 표준 CLIP의 최대 입력이 77토큰으로 제한될 뿐 아니라 **실제 유효 길이(effective length)가 20토큰 미만**임을 보였습니다. 불필요한 서두/어미가 앞단을 차지하면 핵심 시각 단서가 잘리거나 희석됩니다.
3. **패러프레이즈(동의어/어순 변화) 민감도:**
   - **ParaCLIP (`arXiv:2402.15120`, Kim et al., 2024)** 은 동일 의미의 문장이라도 표현 방식이 달라지면 CLIP 텍스트 임베딩 공간에서 검색 순위가 크게 흔들리는 문제를 지적하며 다각적 패러프레이즈 정렬의 필요성을 제시했습니다.
4. **다국어/교차언어 분포 이동 (Multilingual Distribution Shift):**
   - **GRAPE (`arXiv:2509.23370`, Zhang et al., 2025)** 는 다국어·장문 질의에서 발생하는 분포 이동(distributional shift)을 질의 재작성으로 보완해 리트리버 재학습 없이도 평균 Recall@10을 +4.9% 개선했습니다.
   - 반면 **SigLIP 2 (`arXiv:2502.14786`, Tschannen et al., 2025)** 처럼 다국어 코퍼스로 직접 학습된 인코더는 한국어 원문을 그대로 이해하지만, **TinyCLIP (`arXiv:2309.12314`)** 같은 초경량 영문 중심 인코더는 한국어 생 입력 시 붕괴하므로 **경량 온디바이스 매체 분리 + 시각 핵심어 영문 정규화**가 결정적 역할을 합니다.

### (3) 본 프로젝트의 5단계 전처리 설계 (`src/preprocessing/query_processor.py`)
| 전략 코드 | 방식 | 문법 유지 | 비시각 필터 분리 | 다중 관점 확장 |
|---|---|---|---|---|
| `P0_raw` | 사용자 입력 원문 통째로 인코딩 | ✅ | ❌ | ❌ |
| `P1_intent_clean` | 대화형 불용어 제거 + 매체(`image`/`video`) 필터 분리 | ✅ | ✅ | ❌ |
| `P2_keywords` | 핵심 명사/형용사 키워드만 추출 (통사 정보 손실 측정용) | ❌ | ✅ | ❌ |
| `P3_prompt_template` | `a photo of {core}` / `a video frame showing {core}` 정규화 (+ 경량 한영 브리지) | ✅ | ✅ | ❌ |
| `P4_multi_aspect_expansion` | 다중 시각 속성 패러프레이즈 앙상블 임베딩 (Centroid) | ✅ | ✅ | ✅ |

---

## 2. 경량 멀티모달 인코더 & 상업 라이선스 감사

1. **CLIP (`arXiv:2103.00020`, Radford et al., 2021):** 자연어 감독 기반 Zero-Shot 이미지-텍스트 정렬의 기초 (`MIT`).
2. **SigLIP (`arXiv:2303.15343`, Zhai et al., 2023):** 전역 Softmax 대신 Pairwise Sigmoid 손실을 도입해 작은 배치에서도 효율적이고 정밀한 임베딩 학습.
3. **SigLIP 2 (`arXiv:2502.14786`, Tschannen et al., 2025):** 캡셔닝 사전학습 + 자기지도 손실 + 다국어/공정성 데이터 믹스를 결합해 ViT-B(86M)부터 다국어 검색과 밀집 특징(localization/dense features)을 크게 강화 (`Apache-2.0`).
4. **TinyCLIP (`arXiv:2309.12314`, Wu et al., 2023):** Affinity Mimicking과 Weight Inheritance 증류를 통해 ViT-8M/16 모델이 8.9% 파라미터만으로 CLIP ViT-B/16 대비 +3.5% ImageNet Top-1을 달성 (`MIT`).
5. **MobileCLIP (`arXiv:2311.17049`, Vasu et al., 2024) & MobileCLIP2 (`arXiv:2508.20691`, Faghri et al., 2025):** Multi-modal Reinforced Training으로 3~15ms 모바일 추론을 달성했으나, HuggingFace 가중치가 `apple-amlr`(**비상업 연구 전용**) 라이선스이므로 본 프로젝트에서는 **상용 스택에서 제외**하고 문헌 비교군으로만 참조합니다.
6. **MobileViCLIP (`arXiv:2508.07312`, Yang et al., 2025):** 시간적 구조 재파라미터화를 통해 모바일 환경의 비디오-텍스트 검색을 가속하는 아키텍처 방향을 제시합니다.

---

## 3. 모바일 벡터 검색 & Visual RAG 아키텍처 레퍼런스

- **HNSW (`arXiv:1603.09320`, Malkov & Yashunin, 2016):** 계층적 근접 그래프로 로그 시간 ANN 탐색을 달성하나, 모바일에서는 그래프 인접 리스트로 인한 RAM 오버헤드 고려 필요.
- **Product Quantization (`DOI:10.1109/TPAMI.2010.57`, Jégou et al., 2011):** 부분공간 코드북 양자화(IVF-PQ)로 대규모 벡터 메모리를 압축.
- **Reciprocal Rank Fusion (`SIGIR 2009`, Cormack et al., 2009):** 점수 스케일이 다른 밀집 벡터(Visual/Caption)와 희소 검색(BM25) 순위를 `1/(k + rank)`로 무모수 결합.
- **VisRAG (`arXiv:2410.10594`, Yu et al., 2024) & ColPali (`arXiv:2407.01449`, Faysse et al., 2024):** OCR 파싱 손실 없이 이미지를 직접 임베딩하는 Vision-first RAG 패러다임 제시.
- **Re-ranking MMRAG (`arXiv:2501.04695`, Mortaheb et al., 2025) & R3G (`arXiv:2602.00104`, Chen et al., 2026):** 단순 코사인 상위 K의 환각을 줄이기 위해 Coarse Retrieval → Fine-Grained Reranking 2단계 파이프라인 채택.
- **Visual-RAG (`arXiv:2502.16636`, Wu et al., 2025) & Multi-RAG (`arXiv:2505.23990`, Mao et al., 2025):** 검색된 이미지·비디오 프레임 근거를 VLM 응답에 정확히 귀속(grounding)시키는 프레임워크.

---

## 4. RLAIF & 편향 완화 검증 방법론 레퍼런스

- **Constitutional AI (`arXiv:2212.08073`, Bai et al., 2022) & RLAIF (`arXiv:2309.00267`, Lee et al., 2023):** 명시적 원칙(Constitution) 기반 AI 선호 피드백으로 보상 모델(RM)을 학습해 인간 라벨링 비용 없이 RLHF 수준 정렬 달성.
- **RLAIF-V (`arXiv:2405.17220`, Yu et al., 2024):** 오픈소스 MLLM 피드백만으로 객체 환각을 80.7%까지 감소시킨 멀티모달 RLAIF 실증.
- **LLM-as-a-Judge (`arXiv:2306.05685`, Zheng et al., 2023), MLLM-as-a-Judge (`arXiv:2402.04788`, Chen et al., 2024), Survey on LLM-as-a-Judge (`arXiv:2411.15594`, Gu et al., 2024):**
  - MLLM Judge는 **Pairwise Comparison**에서는 높은 분별력을 보이나 **Position Bias(위치 편향)** 와 **Self-Enhancement Bias(자가 선호 편향)** 위험이 존재합니다.
  - 따라서 본 프로젝트에서는 캡션/응답 모델(`SmolVLM2-500M-Video-Instruct`)과 완전히 다른 모델 계열인 `Qwen/Qwen2-VL-2B-Instruct`를 독립 Judge로 배치하고, 모든 쌍대 비교에서 **AB / BA 순서 스왑 일관성 검증**을 적용했습니다.

---

## 5. 실측 검증 결과 요약 ("통문장 vs 전처리")

32개 한·영·대화형·구문형·OCR·비디오 평가 질의(`R0_direct_dense` 기준)에서 실측한 결과, 문헌에서 도출한 가설이 정확히 재현되었습니다:

| 모델 (아키텍처 특성) | `P0_raw` (통문장) MRR / nDCG@5 | `P1_intent_clean` (노이즈·필터 분리) | `P2_keywords` (키워드 나열) | `P3_prompt_template` | `P4_multi_aspect_expansion` |
|---|---|---|---|---|---|
| `M2_tinyclip_8m_onnx_int8` (영문 초경량 INT8) | `0.5339` / `0.5211` (대화형 `0.333`) | `0.8917` / `0.8821` | **`0.9271`** / **`0.9019`** | `0.8542` / `0.8540` | `0.8906` / `0.8845` |
| `M3_tinyclip_40m_fp32` (영문 경량) | `0.5141` / `0.5133` | `0.9453` / `0.9268` | `0.9297` / `0.9169` | **`0.9500`** / `0.9318` | `0.9437` / **`0.9383`** |
| `M6_siglip2_base_fp32` (다국어 네이티브) | `0.9844` / `0.9759` | **`0.9844`** / **`0.9814`** | `0.9281` / `0.9047` (**-0.077 하락**) | `0.9688` / `0.9779` | `0.9688` / `0.9728` |
| `M7_siglip2_base_onnx_int8` (다국어 INT8) | `0.8594` / `0.8148` | `0.8698` / `0.8308` | **`0.8719`** / **`0.8476`** | `0.8656` / `0.8241` | `0.8594` / `0.8314` |

**핵심 결론:**
1. **영문 경량 모델(`TinyCLIP`)에 한국어·대화형 통문장(`P0_raw`)을 그대로 넣으면 붕괴**합니다(MRR `0.51~0.53`). 매체 필터 분리 및 경량 한영 정규화(`P1/P2/P3`)가 필수입니다.
2. **다국어 네이티브 모델(`SigLIP 2 FP32`)에 순수 키워드 추출(`P2_keywords`)을 적용하면 오히려 구문 문맥이 파괴되어 nDCG@5가 `0.9814 → 0.9047`로 급락**합니다. 반면 **구문은 유지하되 비시각적 대화형 노이즈와 매체 제약만 분리하는 `P1_intent_clean`** 이 최고 성능(`nDCG@5 0.9814`)을 달성했습니다.

---

## 6. 2중 팩트체크 완료 arXiv 논문 전체 목록

전체 26편의 원문 메타데이터 스냅샷은 [`docs/verified_references.json`](verified_references.json)에서 확인할 수 있으며, 모든 항목은 `export.arxiv.org` API로 제목·저자·게재일·초록이 2차 검증되었습니다.
