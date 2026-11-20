# UI / UX Design Benchmark & Commercial Asset Specification

모바일 중심 자연어 사진·영상 검색기(**PocketLens**)의 인터페이스를 설계하기 위해 글로벌 3대 디자인 시스템(**Apple HIG**, **Google Material Design 3**, **Microsoft Fluent 2**)의 검색·미디어 갤러리 가이드라인을 비교·벤치마킹하고, 상업적 이용이 100% 허용된 오픈소스 UI 에셋만 채택했습니다.

---

## 1. 글로벌 3대 디자인 시스템 검색 UI/UX 벤치마크

| 비교 항목 | Apple HIG (iOS Photos 패러다임) | Google Material Design 3 (MD3) | Microsoft Fluent 2 | **PocketLens 채택 설계** |
|---|---|---|---|---|
| **검색 진입점 (Entry Point)** | 하단 탭 바의 전용 검색 탭 또는 하단 툴바에 배치해 엄지손가락 도달성 극대화 | 상단 Contained Search Bar → 포커스 시 확장되는 Search View 패턴 | 반응형 SearchBox (작은 화면에서는 돋보기 아이콘으로 축소 후 확장) | **하이브리드:** 상단 Sticky Contained Search Bar + 하단 3-Tab Thumb Bar (`검색 · 라이브러리 · 크레딧/성능`) |
| **검색 전 상태 (Zero-Input Discovery)** | 최근 검색어, 인물/장소/카테고리 스마트 제안 칩 노출 | 최근 검색 칩과 시맨틱 추천 태그를 래핑 그리드로 노출 | 포커스 즉시 최근 검색 토픽을 신뢰성 있게 상단 노출 후 자동완성으로 전환 | 한국어·영어 **추천 자연어 프롬프트 필(Chips)** 제공 (탭 1회로 즉시 검색 실행) |
| **범위 제한 (Scoping & Filtering)** | 검색창 직하단에 Scope Bar / 필터 토큰 배치 (사진·동영상·스크린샷 구분) | 가로 스크롤 가능한 **Filter Chips** 로 즉각 토글 | 모듈형 필터 버튼 및 텍스트 존재 시에만 나타나는 Conditional Clear(`X`) 버튼 | **Scope Filter Chips** (`전체 · 사진만 · 동영상만 · RAG 모드`) + 조건부 Clear 버튼 |
| **결과 그리드 & 미디어 뷰** | 고밀도 2~3열 카드/정사각 썸네일 + 탭 시 부드러운 상세 시트 전환 | 컴팩트 폰 뷰포트에서 데스크톱까지 유동적으로 확장되는 반응형 그리드 | 명확한 메타데이터 계층과 접근성 중심의 고대비 타이포그래피 | 모바일 2열 4:3 카드(`mobile_web/index.html`) / PC 3열 그리드 + 동영상 **매칭 타임스탬프(`#t=...`)** 점프 재생 |
| **지능형 피드백 (AI Transparency)** | Apple Intelligence 스타일의 자연어 맥락 이해 | Surface Container(`surface-container`) 톤 계층으로 AI 요약 분리 | 예측 정렬 및 명확한 출처 신뢰도 표기 | **Query Decomposition Trace** 배지(추출 핵심어, 필터, 지연 ms) + RAG 근거 카드 + 카드별 라이선스/크레딧 배지 |

### 참고 디자인 가이드라인 공식 출처 (Fact-Checked)
1. **Apple Human Interface Guidelines — Searching:** [https://developer.apple.com/design/human-interface-guidelines/searching](https://developer.apple.com/design/human-interface-guidelines/searching)
   - 핵심 원칙: 핵심 기능으로서의 검색 배치(Photos 앱 전용 탭), 최근 검색 및 제안 표시, Scope Bar/Token을 통한 결과 필터링, 현재 검색 범위의 명확한 표시.
2. **Apple Photos User Guide (Natural Language Search):** [https://support.apple.com/guide/iphone/search-for-photos-iph049e9b0a6/ios](https://support.apple.com/guide/iphone/search-for-photos-iph049e9b0a6/ios)
   - 핵심 원칙: 복합 자연어 질의 지원 및 동영상 내 특정 장면/프레임 검색 경험.
3. **Google Material Design 3 — Search:** [https://m3.material.io/components/search/guidelines](https://m3.material.io/components/search/guidelines)
   - 핵심 원칙: Contained Search Bar, leading search icon + hint text + trailing action 구조, Search Bar → Search View 전환 모션, Surface Container 색역 계층.
4. **Microsoft Fluent 2 — SearchBox:** [https://fluent2.microsoft.design/components/web/react/core/searchbox/usage](https://fluent2.microsoft.design/components/web/react/core/searchbox/usage)
   - 핵심 원칙: 부모 컨테이너 폭에 따른 반응형 대응, 입력 텍스트 존재 시에만 표시되는 Clear 트리거, 최근 토픽 → 자동완성 전환.

---

## 2. 상업적 이용 가능 UI/UX 오픈소스 에셋 목록 (2중 팩트체크 완료)

| 에셋 | 용도 | 라이선스 (Raw `LICENSE` 재검증) | 출처 URL |
|---|---|---|---|
| **Lucide Icons** | 검색, 닫기, 스파클, 필터, 갤러리, 방패 SVG 아이콘 | **ISC License** (+ Feather 유래 **MIT**) | [https://github.com/lucide-icons/lucide](https://github.com/lucide-icons/lucide) |
| **Pretendard** | 한국어/영문 모바일·데스크톱 UI 기본 산세리프 폰트 | **SIL Open Font License 1.1** | [https://github.com/orioncactus/pretendard](https://github.com/orioncactus/pretendard) |
| **Inter** | 숫자/영문 UI 보조 폰트 (지연 ms, 점수 배지 가독성) | **SIL Open Font License 1.1** | [https://github.com/rsms/inter](https://github.com/rsms/inter) |

> **검증 메모:** GitHub SPDX API는 다중 저작권(ISC+MIT 또는 다중 파생 폰트 합본) 저장소인 `lucide-icons/lucide`와 `orioncactus/pretendard`를 `NOASSERTION`으로 표기했으나, 각 저장소의 원본 `LICENSE` 파일을 직접 다운로드하여 **ISC License** 및 **SIL OFL 1.1**임을 2차 팩트체크로 확정했습니다.
