"""
Export Vision & Text encoders to ONNX (FP32) and Dynamic INT8 Quantization for Mobile On-Device Deployment.
Verifies numerical cosine fidelity between PyTorch FP32 and ONNX INT8, and records disk footprint & CPU latency.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image

from src.models.encoders import ENCODER_SPECS, PyTorchDualEncoder, _extract_feature_tensor, load_encoder

BASE_DIR = Path(__file__).resolve().parents[2]
ONNX_DIR = BASE_DIR / "exports" / "onnx"


class VisionWrapper(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = _extract_feature_tensor(self.base_model.get_image_features(pixel_values=pixel_values))
        return torch.nn.functional.normalize(feats, p=2, dim=-1)


class TextWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, use_attention_mask: bool = True) -> None:
        super().__init__()
        self.base_model = base_model
        self.use_attention_mask = use_attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_attention_mask and attention_mask is not None:
            feats = _extract_feature_tensor(self.base_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask))
        else:
            feats = _extract_feature_tensor(self.base_model.get_text_features(input_ids=input_ids))
        return torch.nn.functional.normalize(feats, p=2, dim=-1)


def export_and_quantize_one(base_key: str) -> dict[str, Any]:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    spec = ENCODER_SPECS[base_key]
    pt_enc = PyTorchDualEncoder(spec, device="cpu")
    model = pt_enc.model
    proc = pt_enc.processor

    dummy_img = Image.new("RGB", (224, 224), color=(120, 150, 180))
    img_in = proc(images=[dummy_img], return_tensors="pt")["pixel_values"]
    txt_in = proc(
        text=["a dog running on a sunny beach"],
        padding="max_length",
        max_length=spec.max_text_len,
        truncation=True,
        return_tensors="pt",
    )

    vis_fp32_path = ONNX_DIR / f"{base_key}_vision_fp32.onnx"
    txt_fp32_path = ONNX_DIR / f"{base_key}_text_fp32.onnx"
    vis_int8_path = ONNX_DIR / f"{base_key}_vision_int8.onnx"
    txt_int8_path = ONNX_DIR / f"{base_key}_text_int8.onnx"

    vis_wrap = VisionWrapper(model).eval()
    with torch.no_grad():
        torch.onnx.export(
            vis_wrap,
            (img_in,),
            str(vis_fp32_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )

    has_mask = "attention_mask" in txt_in and "siglip" not in spec.hf_id.lower()
    txt_wrap = TextWrapper(model, use_attention_mask=has_mask).eval()
    with torch.no_grad():
        if has_mask:
            torch.onnx.export(
                txt_wrap,
                (txt_in["input_ids"], txt_in["attention_mask"]),
                str(txt_fp32_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["text_embeds"],
                dynamic_axes={
                    "input_ids": {0: "batch"},
                    "attention_mask": {0: "batch"},
                    "text_embeds": {0: "batch"},
                },
                opset_version=17,
                dynamo=False,
            )
        else:
            torch.onnx.export(
                txt_wrap,
                (txt_in["input_ids"],),
                str(txt_fp32_path),
                input_names=["input_ids"],
                output_names=["text_embeds"],
                dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )

    # Dynamic INT8 quantization (try full dynamic quantization, fallback to MatMul/Gemm if Gather/Embed raises ORT kernel error)
    quantize_dynamic(str(vis_fp32_path), str(vis_int8_path), weight_type=QuantType.QInt8)
    quantize_dynamic(str(txt_fp32_path), str(txt_int8_path), weight_type=QuantType.QInt8)
    try:
        _test_sess = ort.InferenceSession(str(txt_int8_path), providers=["CPUExecutionProvider"])
        del _test_sess
    except Exception as q_err:
        print(f"  Full INT8 text ORT check failed ({q_err}); retrying MatMul/Gemm-only INT8...")
        quantize_dynamic(
            str(txt_fp32_path),
            str(txt_int8_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )

    # Verify numerical fidelity against PyTorch FP32
    pt_img_emb = pt_enc.encode_images([dummy_img])[0]
    pt_txt_emb = pt_enc.encode_texts(["a dog running on a sunny beach"])[0]
    pt_enc.unload()

    onnx_key = [k for k, v in ENCODER_SPECS.items() if v.is_onnx and v.base_key == base_key][0]
    onnx_enc = load_encoder(onnx_key, onnx_dir=ONNX_DIR)
    ox_img_emb = onnx_enc.encode_images([dummy_img])[0]
    ox_txt_emb = onnx_enc.encode_texts(["a dog running on a sunny beach"])[0]

    # CPU latency benchmark (PyTorch CPU vs ONNX INT8 CPU)
    t_ox_txt = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = onnx_enc.encode_texts(["a dog running on a sunny beach"])
        t_ox_txt.append((time.perf_counter() - t0) * 1000.0)
    t_ox_img = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = onnx_enc.encode_images([dummy_img])
        t_ox_img.append((time.perf_counter() - t0) * 1000.0)
    onnx_enc.unload()

    vis_fp32_mb = vis_fp32_path.stat().st_size / (1024**2)
    txt_fp32_mb = txt_fp32_path.stat().st_size / (1024**2)
    vis_int8_mb = vis_int8_path.stat().st_size / (1024**2)
    txt_int8_mb = txt_int8_path.stat().st_size / (1024**2)
    fp32_total = vis_fp32_mb + txt_fp32_mb
    int8_total = vis_int8_mb + txt_int8_mb

    return {
        "base_model": base_key,
        "onnx_int8_model": onnx_key,
        "license": spec.license_name,
        "vision_fp32_mb": round(vis_fp32_mb, 2),
        "text_fp32_mb": round(txt_fp32_mb, 2),
        "total_fp32_mb": round(fp32_total, 2),
        "vision_int8_mb": round(vis_int8_mb, 2),
        "text_int8_mb": round(txt_int8_mb, 2),
        "total_int8_mb": round(int8_total, 2),
        "compression_ratio": round(fp32_total / max(int8_total, 1e-6), 2),
        "image_cosine_fidelity": round(float(np.dot(pt_img_emb, ox_img_emb)), 5),
        "text_cosine_fidelity": round(float(np.dot(pt_txt_emb, ox_txt_emb)), 5),
        "onnx_int8_cpu_text_p50_ms": round(float(np.percentile(t_ox_txt, 50)), 2),
        "onnx_int8_cpu_image_p50_ms": round(float(np.percentile(t_ox_img, 50)), 2),
    }


def main() -> None:
    report = []
    for base_key in ["M1_tinyclip_8m_fp32", "M4_clip_b32_fp32", "M6_siglip2_base_fp32"]:
        print(f"Exporting & quantizing {base_key}...")
        res = export_and_quantize_one(base_key)
        report.append(res)
        print(" ", res)
    out = BASE_DIR / "artifacts" / "onnx_quantization_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved ONNX quantization report -> {out}")


if __name__ == "__main__":
    main()
