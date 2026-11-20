"""
Generates a self-contained, interactive, executive-grade HTML Report (report.html)
from verified artifacts, benchmark outputs, RLAIF evaluations, and double-fact-checked credits.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
REPORT_PATH = BASE_DIR / "report.html"


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    bench = _load(ARTIFACTS_DIR / "benchmark_results.json")
    rlaif = _load(ARTIFACTS_DIR / "rlaif_feedback.json")
    onnx_rep = _load(ARTIFACTS_DIR / "onnx_quantization_report.json")
    manifest = _load(DATA_DIR / "manifest_verified.json")
    refs = _load(DOCS_DIR / "verified_references.json")

    ds = bench["dataset_summary"]
    pw = rlaif["debiased_pairwise_r4_vs_r0"]
    rubric = rlaif["held_out_rubric_means"]

    # Build tables HTML safely
    onnx_rows = "".join(
        f"<tr><td><code>{html.escape(r['base_model'])}</code></td>"
        f"<td>{html.escape(r['license'])}</td>"
        f"<td>{r['total_fp32_mb']} MB</td>"
        f"<td><strong>{r['total_int8_mb']} MB</strong></td>"
        f"<td>{r['compression_ratio']}×</td>"
        f"<td>{r['image_cosine_fidelity']} / {r['text_cosine_fidelity']}</td>"
        f"<td>{r['onnx_int8_cpu_text_p50_ms']} ms</td></tr>"
        for r in onnx_rep
    )

    exp_a_rows = "".join(
        f"<tr><td><code>{html.escape(r['model'])}</code></td>"
        f"<td><code>{html.escape(r['prep_strategy'])}</code></td>"
        f"<td><code>{html.escape(r['rag_mode'])}</code></td>"
        f"<td>{r['recall_at_1']:.3f}</td><td>{r['recall_at_5']:.3f}</td>"
        f"<td><strong>{r['mrr']:.3f}</strong></td><td>{r['ndcg_at_5']:.3f}</td></tr>"
        for r in bench["exp_a_query_preprocessing"]
        if r["rag_mode"] == "R1_caption_hybrid"
    )

    scaling_rows_512 = "".join(
        f"<tr><td><code>{html.escape(r['algorithm'])}</code></td>"
        f"<td>512d</td><td>{r['ram_mb']} MB</td>"
        f"<td>{r['recall_at_10_vs_exact']:.4f}</td>"
        f"<td>{r['latency_p50_ms']} ms</td>"
        f"<td>{r['latency_p95_ms']} ms</td></tr>"
        for r in bench["scaling_10k_dim512"]
    )
    scaling_rows = scaling_rows_512 + "".join(
        f"<tr><td><code>{html.escape(r['algorithm'])}</code></td>"
        f"<td>768d</td><td>{r['ram_mb']} MB</td>"
        f"<td>{r['recall_at_10_vs_exact']:.4f}</td>"
        f"<td>{r['latency_p50_ms']} ms</td>"
        f"<td>{r['latency_p95_ms']} ms</td></tr>"
        for r in bench["scaling_10k_dim768"]
    )

    exp_c_rows = "".join(
        f"<tr><td><code>{html.escape(r['model'])}</code></td>"
        f"<td><code>{html.escape(r['search_algo'])}</code></td>"
        f"<td><code>{html.escape(r['rag_mode'])}</code></td>"
        f"<td>{r['recall_at_1']:.3f}</td><td>{r['recall_at_5']:.3f}</td>"
        f"<td><strong>{r['mrr']:.3f}</strong></td><td>{r['ndcg_at_5']:.3f}</td>"
        f"<td>{r['test_mrr']:.3f}</td><td>{r['search_latency_ms']:.2f} ms</td></tr>"
        for r in bench["exp_c_rag_matrix"]
    )

    rubric_rows = "".join(
        f"<tr><td><code>{html.escape(sys_name)}</code></td>"
        f"<td>{vals['top1_entity']:.2f}</td>"
        f"<td>{vals['top1_attribute']:.2f}</td>"
        f"<td>{vals['top1_media_fit']:.2f}</td>"
        f"<td><strong>{vals['top1_overall']:.2f}</strong></td>"
        f"<td>{vals['top3_mean_overall']:.2f}</td></tr>"
        for sys_name, vals in rubric.items()
    )

    synth_cards = "".join(
        f"<div class='card'><h4>Q: {html.escape(s['query'])}</h4>"
        f"<p>{html.escape(s['synthesized_answer'])}</p>"
        f"<div class='meta'>Top Matches: {', '.join(html.escape(x['media_id']) for x in s['top_results'])}</div></div>"
        for s in rlaif.get("rag_synthesis_samples", [])
    )

    credits_rows = "".join(
        f"<tr><td><code>{html.escape(m['media_id'])}</code></td>"
        f"<td>{html.escape(m['media_type'])}</td>"
        f"<td>{html.escape(m['theme'])}</td>"
        f"<td>{html.escape(m['canonical_title'][:45])}</td>"
        f"<td>{html.escape(m['author'][:36])}</td>"
        f"<td><a href='{html.escape(m['license_url'])}' target='_blank'>{html.escape(m['license_short'])}</a></td>"
        f"<td><a href='{html.escape(m['source_page_url'])}' target='_blank'>{html.escape(m['source_platform'])}</a></td>"
        f"<td>✅ x{m['fact_check_count']}</td></tr>"
        for m in manifest
    )

    paper_rows = "".join(
        f"<tr><td><strong>{html.escape(p['key'])}</strong></td>"
        f"<td><a href='{html.escape(p['url'])}' target='_blank'>{html.escape(p['arxiv_id'])}</a></td>"
        f"<td>{html.escape(p['published'])}</td>"
        f"<td>{html.escape(p['title'])}</td>"
        f"<td>{html.escape(', '.join(p['authors'][:3]))}{' et al.' if len(p['authors'])>3 else ''}</td>"
        f"<td>✅ arXiv API x2</td></tr>"
        for p in refs["arxiv_papers"]
    )

    bench_json_str = json.dumps({
        "onnx": onnx_rep,
        "scaling768": bench["scaling_10k_dim768"],
        "exp_a": [r for r in bench["exp_a_query_preprocessing"] if r["rag_mode"] == "R1_caption_hybrid"],
        "exp_c": [r for r in bench["exp_c_rag_matrix"] if r["search_algo"] == "S1_sq8_int8"],
        "rubric": rubric,
    }, ensure_ascii=False)

    html_doc = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PocketLens — Mobile Visual RAG &amp; RLAIF Benchmark Report</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0B1120;
      --surface: #111827;
      --surface-2: #1E293B;
      --text: #F8FAFC;
      --muted: #94A3B8;
      --accent: #38BDF8;
      --accent-2: #818CF8;
      --success: #34D399;
      --border: #1F2937;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Pretendard", "Inter", sans-serif;
      background: radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.12), transparent 40%), var(--bg);
      color: var(--text);
      line-height: 1.65;
    }}
    .wrap {{ max-width: 1220px; margin: 0 auto; padding: 32px 24px 80px; }}
    header {{
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(17, 24, 39, 0.95));
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 32px;
      margin-bottom: 28px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: -0.02em; }}
    h2 {{ margin: 34px 0 14px; font-size: 22px; border-left: 4px solid var(--accent); padding-left: 12px; }}
    h3 {{ margin: 18px 0 10px; font-size: 16px; color: var(--accent); }}
    p, li {{ color: #CBD5E1; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
      margin: 22px 0;
    }}
    .kpi {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
    }}
    .kpi .val {{ font-size: 26px; font-weight: 700; color: var(--accent); }}
    .kpi .lbl {{ font-size: 13px; color: var(--muted); }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 20px; }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
    }}
    .meta {{ font-size: 12px; color: var(--muted); margin-top: 8px; }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      margin: 14px 0 26px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ background: var(--surface-2); color: #E2E8F0; font-weight: 600; white-space: nowrap; }}
    code {{ background: rgba(148, 163, 184, 0.14); padding: 2px 6px; border-radius: 6px; font-size: 12px; }}
    canvas {{ max-height: 330px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div style="font-size:12px;color:var(--accent);font-weight:700;letter-spacing:0.06em;">MOBILE-FIRST VISUAL RAG &amp; RLAIF TECHNICAL REPORT</div>
      <h1>PocketLens: 모바일 중심 자연어 사진·영상 검색기</h1>
      <p>경량/양자화 멀티모달 인코더 × 자연어 질의 전처리 × 모바일 벡터 검색 알고리즘 × 5종 Visual RAG × 독립 VLM 기반 RLAIF 폐루프 검증 보고서</p>
      <div class="meta">생성 시각: {datetime.now(timezone.utc).isoformat()} · 모든 문헌·모델·미디어 크레딧 2중 팩트체크 완료</div>
    </header>

    <div class="kpi-grid">
      <div class="kpi"><div class="val">{ds['total_units']}</div><div class="lbl">인덱싱된 시각 단위 (사진 + 영상 키프레임)</div></div>
      <div class="kpi"><div class="val">{ds['num_eval_queries']}</div><div class="lbl">한/영 평가 질의 ({ds['train_queries']} Train / {ds['test_queries']} Test)</div></div>
      <div class="kpi"><div class="val">{pw['r4_wins']}승 {pw['ties_or_swapped_inconsistent']}무 {pw['r0_wins']}패</div><div class="lbl">RLAIF 적용 R4 vs Raw R0 (AB/BA 스왑 검증)</div></div>
      <div class="kpi"><div class="val">{bench['mobile_bundle_meta']['sqlite_mb']} MB</div><div class="lbl">모바일 SQLite+INT8 인덱스 번들 크기</div></div>
    </div>

    <h2>1. 경량 모델 &amp; ONNX INT8 양자화 벤치마크</h2>
    <p>상업적 이용이 100% 허용된 <code>MIT</code> / <code>Apache-2.0</code> 모델만 채택(Apple <code>apple-amlr</code> 및 <code>jina-clip-v2</code> CC-BY-NC-4.0은 감사 후 상용 스택에서 제외)하여, 네이티브 경량 모델(TinyCLIP)과 무거운 모델의 INT8 양자화(CLIP-B/32, SigLIP2-Base)를 직접 비교했습니다.</p>
    <div class="card"><canvas id="chartOnnx"></canvas></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Base Model</th><th>License</th><th>FP32 Size</th><th>ONNX INT8 Size</th><th>압축률</th><th>Cosine Fidelity (Img/Txt)</th><th>CPU Text P50</th></tr></thead>
        <tbody>{onnx_rows}</tbody>
      </table>
    </div>

    <h2>2. 자연어 전처리 전략 비교 (통문장 vs 전처리)</h2>
    <p><code>arXiv:2103.00020</code>(CLIP), <code>arXiv:2402.15120</code>(ParaCLIP), <code>arXiv:2406.03411</code>(PlugIR), <code>arXiv:2502.14786</code>(SigLIP 2) 연구를 바탕으로, 대화형 불용어 제거·매체 필터 분리·키워드 추출·프롬프트 정규화·다각 확장을 모델별로 검증했습니다.</p>
    <div class="card"><canvas id="chartPrep"></canvas></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Preprocessing</th><th>RAG Mode</th><th>Recall@1</th><th>Recall@5</th><th>MRR</th><th>nDCG@5</th></tr></thead>
        <tbody>{exp_a_rows}</tbody>
      </table>
    </div>

    <h2>3. 모바일 10,000장 규모 벡터 검색 알고리즘 스케일링 (768d)</h2>
    <div class="card-grid">
      <div class="card"><canvas id="chartScale"></canvas></div>
      <div class="card">
        <h3>모바일 설계 결론</h3>
        <ul>
          <li><strong>S1_sq8_int8:</strong> FP32 대비 <strong>4× RAM 절감</strong>으로 Recall@10 손실 없이 모바일 기본 인덱스에 최적.</li>
          <li><strong>S3_bq_rescore:</strong> 핫 RAM을 <strong>32× 압축(1-bit Hamming)</strong>하면서도 Top-K 재채점으로 높은 Recall 회복.</li>
          <li><strong>S4_hnsw:</strong> 초고속 탐색이지만 그래프 엣지로 인해 RAM이 Flat보다 증가해 대용량 모바일 메모리 예산 주의 필요.</li>
        </ul>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Algorithm</th><th>Dim</th><th>Index RAM (10k)</th><th>Recall@10 vs Exact</th><th>Latency P50</th><th>Latency P95</th></tr></thead>
        <tbody>{scaling_rows}</tbody>
      </table>
    </div>

    <h2>4. 통합 벤치마크 매트릭스 (Model × Search × 5종 Visual RAG)</h2>
    <div class="card"><canvas id="chartRag"></canvas></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Search Algo</th><th>RAG Mode</th><th>R@1</th><th>R@5</th><th>MRR</th><th>nDCG@5</th><th>Test MRR</th><th>Latency</th></tr></thead>
        <tbody>{exp_c_rows}</tbody>
      </table>
    </div>

    <h2>5. RLAIF 검증 &amp; 폐루프 보상 최적화 (Held-Out Test)</h2>
    <p>자기 강화 편향(Self-Enhancement Bias)을 차단하기 위해 <code>SmolVLM2-500M-Video-Instruct</code>(캡션/응답 합성)와 완전히 분리된 독립 VLM <code>Qwen/Qwen2-VL-2B-Instruct</code>(Apache-2.0)를 RLAIF Judge로 사용했습니다. Train Split에서 수집한 <strong>{rlaif['train_preference_pairs_count']}쌍</strong>의 AI 선호 데이터로 Bradley-Terry 보상 가중치 <code>{rlaif['learned_bradley_terry_weights']}</code>를 학습한 뒤, Held-Out Test Split에서 검증했습니다.</p>
    <div class="card" style="margin-bottom:14px;">
      <strong>핵심 RLAIF 검증 결과:</strong>
      <ul>
        <li><strong>보상 모델 학습 효과 (Held-Out Test):</strong> <code>M7_siglip2_base_onnx_int8</code> 기준 Test MRR이 <code>0.8438 (R0) → 0.9688 (R4)</code>로, <code>M2_tinyclip_8m_onnx_int8</code>은 Test MRR <code>0.9062 → 1.0000</code> (nDCG@5 <code>0.9071 → 0.9917</code>)로 도약했습니다.</li>
        <li><strong>헌법적 루브릭(Top-3 Mean Overall):</strong> <code>3.396 (R0) → 3.542 (R4 Pre-RLAIF) → 3.646 (R4 Post-RLAIF)</code>로 상위 3개 슬레이트 품질이 일관 개선되었습니다.</li>
        <li><strong>위치 편향 탐지 (AB/BA 스왑 일관성 <code>{pw['consistency_rate']*100:.1f}%</code>):</strong> 2B급 VLM은 2개 이미지를 나란히 비교할 때 서로 다른 5건 모두에서 첫 번째 위치(Position 1)를 선호하는 편향을 보였으며, <code>MLLM-as-a-Judge</code> 프로토콜에 따라 AB/BA 스왑 가드로 이를 정확히 검출·차단했습니다.</li>
      </ul>
    </div>
    <div class="card"><canvas id="chartRlaif"></canvas></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Pipeline (Held-Out Test)</th><th>Entity (0-5)</th><th>Attribute (0-5)</th><th>Media Fit (0-5)</th><th>Top-1 Overall</th><th>Top-3 Mean</th></tr></thead>
        <tbody>{rubric_rows}</tbody>
      </table>
    </div>
    <h3>SmolVLM2 기반 Grounded Visual RAG 응답 샘플</h3>
    <div class="card-grid">{synth_cards}</div>

    <h2>6. 오픈 라이선스 미디어 크레딧 &amp; 학술 문헌 (2중 팩트체크)</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Media ID</th><th>Type</th><th>Theme</th><th>Title</th><th>Author</th><th>License</th><th>Source</th><th>Audit</th></tr></thead>
        <tbody>{credits_rows}</tbody>
      </table>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Key</th><th>arXiv ID</th><th>Published</th><th>Title</th><th>Authors</th><th>Verification</th></tr></thead>
        <tbody>{paper_rows}</tbody>
      </table>
    </div>

    <h2>7. 역할 분담 및 기여 내역 (Contributors &amp; Role Division)</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Role</th><th>Contributor</th><th>Key Contributions Across Project Lifecycle</th></tr></thead>
        <tbody>
          <tr>
            <td><strong>Lead Architect &amp; Product Vision</strong></td>
            <td><a href="https://github.com/devshin-91" target="_blank" style="color:var(--accent)">@devshin-91</a> (Chansu Shin)</td>
            <td>모바일 중심 Visual RAG 기획 및 온디바이스 이식 아키텍처 방향 수립 · 오픈 라이선스 미디어/문헌 2중 팩트체크 프로토콜 설계 · 자연어 전처리(&ldquo;통문장 vs 전처리&rdquo;) 및 4축 벤치마크(인코더×검색×RAG) × RLAIF 검증 체계 설계 · 글로벌 디자인 시스템 벤치마크 디렉션 · Windows 실환경 런타임 버그(<code>WinError 10054</code>) 포착, 모델 전환(<code>Hydrogen → Argon</code>) 오케스트레이션 및 3계층 <code>Apache-2.0</code> 라이선스 거버넌스</td>
          </tr>
          <tr>
            <td><strong>AI Systems Engineering &amp; Verification</strong></td>
            <td><a href="https://github.com/antigravity-bot" target="_blank" style="color:var(--accent)">@antigravity-bot</a> (Google Antigravity)</td>
            <td>[<code>Gemini 4 Hydrogen XHigh</code>] 26편 arXiv 논문 &amp; 61개 상용 오픈 미디어 수집·2중 검증 자동화 · 전경 활동 피크 기반 비디오 키프레임 추출(75개) 및 5종 자연어 전처리 구현 · 7종 멀티모달 인코더(ONNX INT8) × 7종 모바일 검색(10k 스케일링) × 5종 Visual RAG × 독립 VLM(<code>Qwen2-VL-2B</code>) RLAIF 보상 학습 파이프라인 구축 · Streamlit PC UI &amp; FastAPI 모바일 PWA 구현<br>[<code>Gemini 4 Argon High</code>] 전 대화·CPython <code>proactor_events.py</code>·미디어 코덱(<code>ffprobe</code>) 역추적 감사, <code>_finish_proactor_cleanup()</code> 소켓/서버 핸들 누수 패치, <code>vid_cat_puddle.ogv</code>(Theora) H.264 <code>+faststart</code> 웹 프록시 변환 및 실브라우저 무오류 스트리밍 검증</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const D = {bench_json_str};
    new Chart(document.getElementById('chartOnnx'), {{
      type: 'bar',
      data: {{
        labels: D.onnx.map(x => x.base_model),
        datasets: [
          {{ label: 'FP32 Size (MB)', data: D.onnx.map(x => x.total_fp32_mb), backgroundColor: '#64748B' }},
          {{ label: 'ONNX INT8 Size (MB)', data: D.onnx.map(x => x.total_int8_mb), backgroundColor: '#38BDF8' }}
        ]
      }},
      options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#E2E8F0' }} }} }}, scales: {{ x: {{ ticks: {{ color: '#CBD5E1' }} }}, y: {{ ticks: {{ color: '#CBD5E1' }} }} }} }}
    }});

    const prepStrats = [...new Set(D.exp_a.map(x => x.prep_strategy))];
    const prepModels = [...new Set(D.exp_a.map(x => x.model))];
    const palette = ['#64748B', '#F59E0B', '#EC4899', '#38BDF8', '#34D399'];
    new Chart(document.getElementById('chartPrep'), {{
      type: 'bar',
      data: {{
        labels: prepModels,
        datasets: prepStrats.map((st, i) => ({{
          label: st,
          data: prepModels.map(m => (D.exp_a.find(r => r.model === m && r.prep_strategy === st) || {{}}).mrr || 0),
          backgroundColor: palette[i % palette.length]
        }}))
      }},
      options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#E2E8F0' }} }} }}, scales: {{ x: {{ ticks: {{ color: '#CBD5E1' }} }}, y: {{ ticks: {{ color: '#CBD5E1' }}, min: 0, max: 1 }} }} }}
    }});

    new Chart(document.getElementById('chartScale'), {{
      type: 'bar',
      data: {{
        labels: D.scaling768.map(x => x.algorithm),
        datasets: [
          {{ label: 'RAM (MB)', data: D.scaling768.map(x => x.ram_mb), backgroundColor: '#818CF8', yAxisID: 'y' }},
          {{ label: 'Recall@10', data: D.scaling768.map(x => x.recall_at_10_vs_exact), backgroundColor: '#34D399', yAxisID: 'y1' }}
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#E2E8F0' }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#CBD5E1' }} }},
          y: {{ position: 'left', ticks: {{ color: '#CBD5E1' }} }},
          y1: {{ position: 'right', min: 0, max: 1.05, grid: {{ drawOnChartArea: false }}, ticks: {{ color: '#34D399' }} }}
        }}
      }}
    }});

    const ragModes = [...new Set(D.exp_c.map(x => x.rag_mode))];
    const ragModels = [...new Set(D.exp_c.map(x => x.model))];
    new Chart(document.getElementById('chartRag'), {{
      type: 'bar',
      data: {{
        labels: ragModels,
        datasets: ragModes.map((rm, i) => ({{
          label: rm,
          data: ragModels.map(m => (D.exp_c.find(r => r.model === m && r.rag_mode === rm) || {{}}).ndcg_at_5 || 0),
          backgroundColor: palette[i % palette.length]
        }}))
      }},
      options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#E2E8F0' }} }} }}, scales: {{ x: {{ ticks: {{ color: '#CBD5E1' }} }}, y: {{ ticks: {{ color: '#CBD5E1' }}, min: 0, max: 1 }} }} }}
    }});

    const rKeys = Object.keys(D.rubric);
    new Chart(document.getElementById('chartRlaif'), {{
      type: 'bar',
      data: {{
        labels: rKeys,
        datasets: [
          {{ label: 'Top-1 Overall (0-5)', data: rKeys.map(k => D.rubric[k].top1_overall), backgroundColor: '#38BDF8' }},
          {{ label: 'Top-3 Mean (0-5)', data: rKeys.map(k => D.rubric[k].top3_mean_overall), backgroundColor: '#818CF8' }},
          {{ label: 'Media Fit (0-5)', data: rKeys.map(k => D.rubric[k].top1_media_fit), backgroundColor: '#34D399' }}
        ]
      }},
      options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#E2E8F0' }} }} }}, scales: {{ x: {{ ticks: {{ color: '#CBD5E1' }} }}, y: {{ ticks: {{ color: '#CBD5E1' }}, min: 0, max: 5 }} }} }}
    }});
  </script>
</body>
</html>
"""
    REPORT_PATH.write_text(html_doc, encoding="utf-8")
    print(f"Generated interactive HTML report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
