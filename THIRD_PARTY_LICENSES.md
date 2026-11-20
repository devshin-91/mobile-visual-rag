# Third-Party Licenses & Commercial Compliance Audit

Every model, font, icon pack, software library, and media asset included in **PocketLens (Mobile Visual RAG)** was audited across two verification passes against its canonical source/API and raw `LICENSE` file.

---

## 1. Multimodal & VLM Models (Selected vs. Excluded)

| Model ID | Role in Project | License | Commercial Use | Audit Notes |
|---|---|---|---|---|
| `wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` | Ultra-lightweight dual encoder (`M1`, `M2 INT8`) | **MIT** | ✅ Allowed | Native 8M vision / 3M text distilled CLIP (`89.47 MB` safetensors). |
| `wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M` | Compact dual encoder (`M3`) | **MIT** | ✅ Allowed | 40M vision / 19M text distilled CLIP (`321.25 MB` safetensors). |
| `openai/clip-vit-base-patch32` | Heavy baseline + ONNX INT8 (`M4`, `M5`) | **MIT** | ✅ Allowed | Verified via `openai/CLIP` `LICENSE` (Copyright 2021 OpenAI, MIT). |
| `google/siglip2-base-patch16-224` | Multilingual native dual encoder (`M6`, `M7 INT8`) | **Apache-2.0** | ✅ Allowed | Verified via HF API (`license:apache-2.0`); Korean & English native support. |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | Edge captioner & grounded RAG synthesizer | **Apache-2.0** | ✅ Allowed | Verified via HF API (`license:apache-2.0`); ~500M params. |
| `Qwen/Qwen2-VL-2B-Instruct` | Independent RLAIF Visual Judge | **Apache-2.0** | ✅ Allowed | Verified via HF API (`license:apache-2.0`); avoids self-enhancement bias. |
| `apple/MobileCLIP-S0` | *Literature benchmark reference only* | `apple-amlr` | ❌ **Excluded** | Apple Machine Learning Research License restricts to non-commercial research. |
| `jinaai/jina-clip-v2` | *Literature benchmark reference only* | `cc-by-nc-4.0` | ❌ **Excluded** | Non-Commercial clause (`NC`) violates commercial readiness requirement. |

---

## 2. UI / UX Design Assets (Fonts & Icons)

| Asset | Source Repository | Verified License | Commercial Use |
|---|---|---|---|
| **Lucide Icons** | `https://github.com/lucide-icons/lucide` | **ISC License** (core) + **MIT License** (Feather-derived) | ✅ Allowed (retain notice in `THIRD_PARTY_LICENSES.md`) |
| **Pretendard Font** | `https://github.com/orioncactus/pretendard` | **SIL Open Font License 1.1 (OFL-1.1)** | ✅ Allowed (cannot resell standalone font files) |
| **Inter Font** | `https://github.com/rsms/inter` | **SIL Open Font License 1.1 (OFL-1.1)** | ✅ Allowed (cannot resell standalone font files) |

### Lucide ISC License Notice
```text
ISC License

Copyright (c) 2026 Lucide Icons and Contributors

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.
```

---

## 3. Core Vector Search & Serving Libraries

| Library | Role | Verified License |
|---|---|---|
| `nmslib/hnswlib` | Graph-based HNSW ANN index | **Apache-2.0** |
| `facebookresearch/faiss` | IVF-PQ & Flat vector indexing | **MIT** |
| `rank-bm25` | Lexical BM25 sparse scorer for Hybrid RRF | **Apache-2.0** |
| `onnxruntime` / `onnx` | Cross-platform FP32/INT8 edge inference | **MIT** / **Apache-2.0** |
| `streamlit` / `fastapi` | PC prototype UI & Mobile PWA bridge | **Apache-2.0** / **MIT** |

---

## 4. Media Dataset Compliance

All 61 photos and videos are sourced strictly from **CC0 1.0** (48 items), **Public Domain (NASA Media Usage Guidelines / Wikimedia PD)** (5 items), **CC BY 3.0** (2 items), and **CC BY 4.0** (`intel-iot-devkit/sample-videos`, 6 items) — with **zero** ShareAlike (`SA`) or Non-Commercial (`NC`) items. Complete item-by-item attributions and SHA-256 checksums are documented in [`CREDITS.md`](CREDITS.md).

---

## 5. Repository Multi-Part Licensing Summary

- **Original Source Code (`src/`, `app.py`, `mobile_web/`, `scripts/`, `tests/`):** [**Apache License 2.0**](LICENSE) (`Copyright 2026 Chansu Shin <devshin91@gmail.com>`)
- **Benchmark Reports & Documentation (`docs/`, `report.html`, `README.md`):** **Creative Commons Attribution 4.0 International (`CC BY 4.0`)**
- **Third-Party Media, Model Weights & UI Assets (`data/`, `exports/onnx/`):** Governed by their respective original licenses itemized above and in [`CREDITS.md`](CREDITS.md) / [`NOTICE`](NOTICE).
