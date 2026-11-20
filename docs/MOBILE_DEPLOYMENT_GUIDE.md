# Mobile Deployment Architecture & Edge Engineering Guide

**PocketLens**는 데스크톱(PC)에서 먼저 알고리즘·전처리·RAG·RLAIF를 완결성 있게 검증한 뒤, **모바일 온디바이스(Android / iOS / Mobile PWA)** 환경으로 즉시 이식할 수 있도록 설계되었습니다.

---

## 1. PC 프로토타입 → 모바일 온디바이스 이식 아키텍처

```text
[PC Research & Calibration Pipeline]
  1) PyTorch FP32/FP16 모델 (TinyCLIP-8M, CLIP-B/32, SigLIP2-Base)
  2) ONNX Export + Dynamic INT8 Quantization (exports/onnx/*.onnx)
  3) SmolVLM2-500M 오프라인 캡션/태그 생성 + 적응형 비디오 키프레임 추출
  4) Qwen2-VL-2B RLAIF Judge → Train Split 기반 Bradley-Terry 가중치 학습
  5) SQLite + SQ8 INT8 모바일 번들 패킹 (exports/mobile_bundle/gallery_index.sqlite)
         │
         ├──► [Option A: Native Mobile On-Device (Android / iOS)]
         │     • ONNX Runtime Mobile (NNAPI / CoreML EP)로 텍스트·이미지 INT8 추론
         │     • SQLite + INT8/Binary Vector로 네트워크 없이 100% 오프라인 검색
         │
         └──► [Option B: Mobile-First PWA via FastAPI LAN Bridge]
               • `uvicorn src.mobile_bridge.api_server:app --host 0.0.0.0 --port 8000`
               • 모바일 브라우저(`http://<PC-IP>:8000`)에서 Apple HIG + M3 UI 즉시 체험
```

---

## 2. 왜 모바일에서 Dual-Encoder + INT8/BQ Two-Stage가 핵심인가?

1. **초기 인덱싱 vs 실시간 질의 비대칭성:**
   - 스마트폰 갤러리의 사진/동영상 키프레임 인코딩(Vision Tower)은 **충전 중 백그라운드(iOS `BGProcessingTask` / Android `WorkManager`)** 에서 1회만 수행합니다.
   - 반면 사용자가 검색창에 타이핑할 때는 **Text Tower + Vector Search**만 실행되므로, Text Tower의 **ONNX INT8 양자화**와 **SQLite INT8/Binary 인덱스**가 체감 반응 속도와 배터리 소모를 결정합니다.
2. **메모리 예산(RAM Budget)과 OS Kill 회피:**
   - 모바일 OS는 백그라운드/포그라운드 앱의 RAM 점유율이 급증하면(예: HNSW 그래프가 Raw 벡터 대비 2~3배 메모리를 점유할 때) 즉시 프로세스를 종료할 수 있습니다.
   - **SQ8 INT8 (4× 압축)** 은 정밀도 손실 없이 RAM을 1/4로 줄이며, **Two-Stage Binary Quantization (`BQ + Rescore`)** 은 핫 인덱스 RAM을 **32× 압축**해 10,000장 이상의 갤러리도 수백 KB~수 MB 수준으로 검색할 수 있습니다.

---

## 3. 모바일 앱 통합 스니펫 (ONNX Runtime Mobile & SQLite)

### Android (Kotlin + ONNX Runtime Mobile)
```kotlin
// dependencies: implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")
val ortEnv = OrtEnvironment.getEnvironment()
val sessionOptions = OrtSession.SessionOptions().apply {
    addNnapi() // 모바일 NPU/DSP 하드웨어 가속 활용
}
val textSession = ortEnv.createSession(
    assets.open("M1_tinyclip_8m_fp32_text_int8.onnx").readBytes(),
    sessionOptions
)
```

### iOS (Swift + ONNX Runtime / CoreML EP)
```swift
let env = try ORTEnv(loggingLevel: .warning)
let options = try ORTSessionOptions()
try options.appendCoreMLExecutionProvider(with: ORTCoreMLExecutionProviderOptions())
let textSession = try ORTSession(
    env: env,
    modelPath: Bundle.main.path(forResource: "M1_tinyclip_8m_fp32_text_int8", ofType: "onnx")!,
    sessionOptions: options
)
```

---

## 4. 로컬에서 PC Streamlit 및 모바일 PWA 실행하기

### 1) PC Streamlit 프로토타입 & 벤치마크 대시보드
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
- 사이드바에서 **`📱 모바일 뷰포트 미리보기 (390px)`** 를 켜면 PC에서도 스마트폰 UI 레이아웃을 즉시 확인할 수 있습니다.

### 2) 모바일 PWA 서버 (스마트폰 실제 브라우저 접속용)
```powershell
.\.venv\Scripts\uvicorn.exe src.mobile_bridge.api_server:app --host 0.0.0.0 --port 8000
```
- PC 브라우저: `http://localhost:8000`
- 같은 Wi-Fi의 스마트폰 브라우저: `http://<PC의 로컬 IP>:8000`
