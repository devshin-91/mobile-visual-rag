"""
Unified Lightweight Multimodal Dual-Encoder Wrappers (PyTorch FP32/FP16 & ONNX INT8).

All selected models are 100% commercially usable open-weights:
- TinyCLIP-ViT-8M-16-Text-3M-YFCC15M (MIT, ~11M params, ultra-mobile)
- TinyCLIP-ViT-40M-32-Text-19M-LAION400M (MIT, compact mobile)
- CLIP-ViT-B/32 (MIT, classic heavy baseline + ONNX INT8 quantized comparison)
- SigLIP2-base-patch16-224 (Apache 2.0, native multilingual + ONNX INT8 quantized)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

if TYPE_CHECKING:
    import torch


def _load_processor_cached(hf_id: str):
    try:
        return AutoProcessor.from_pretrained(hf_id, local_files_only=True)
    except Exception:
        return AutoProcessor.from_pretrained(hf_id)


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    hf_id: str
    license_name: str
    multilingual: bool
    max_text_len: int
    embedding_dim: int
    is_onnx: bool = False
    base_key: str = ""
    description: str = ""


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "M1_tinyclip_8m_fp32": EncoderSpec(
        key="M1_tinyclip_8m_fp32",
        hf_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        license_name="MIT",
        multilingual=False,
        max_text_len=77,
        embedding_dim=512,
        description="Ultra-lightweight distilled CLIP (8M Vision + 3M Text)",
    ),
    "M2_tinyclip_8m_onnx_int8": EncoderSpec(
        key="M2_tinyclip_8m_onnx_int8",
        hf_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
        license_name="MIT",
        multilingual=False,
        max_text_len=77,
        embedding_dim=512,
        is_onnx=True,
        base_key="M1_tinyclip_8m_fp32",
        description="ONNX INT8 quantized TinyCLIP-8M for mobile NPU/CPU",
    ),
    "M3_tinyclip_40m_fp32": EncoderSpec(
        key="M3_tinyclip_40m_fp32",
        hf_id="wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        license_name="MIT",
        multilingual=False,
        max_text_len=77,
        embedding_dim=512,
        description="Compact distilled CLIP (40M Vision + 19M Text)",
    ),
    "M4_clip_b32_fp32": EncoderSpec(
        key="M4_clip_b32_fp32",
        hf_id="openai/clip-vit-base-patch32",
        license_name="MIT",
        multilingual=False,
        max_text_len=77,
        embedding_dim=512,
        description="Standard CLIP ViT-B/32 baseline (~151M params)",
    ),
    "M5_clip_b32_onnx_int8": EncoderSpec(
        key="M5_clip_b32_onnx_int8",
        hf_id="openai/clip-vit-base-patch32",
        license_name="MIT",
        multilingual=False,
        max_text_len=77,
        embedding_dim=512,
        is_onnx=True,
        base_key="M4_clip_b32_fp32",
        description="ONNX INT8 quantized CLIP ViT-B/32 (compressed heavy model)",
    ),
    "M6_siglip2_base_fp32": EncoderSpec(
        key="M6_siglip2_base_fp32",
        hf_id="google/siglip2-base-patch16-224",
        license_name="Apache-2.0",
        multilingual=True,
        max_text_len=64,
        embedding_dim=768,
        description="Multilingual SigLIP 2 Base (Korean+English native)",
    ),
    "M7_siglip2_base_onnx_int8": EncoderSpec(
        key="M7_siglip2_base_onnx_int8",
        hf_id="google/siglip2-base-patch16-224",
        license_name="Apache-2.0",
        multilingual=True,
        max_text_len=64,
        embedding_dim=768,
        is_onnx=True,
        base_key="M6_siglip2_base_fp32",
        description="ONNX INT8 quantized SigLIP 2 Base (multilingual mobile)",
    ),
}


def _l2_norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _extract_feature_tensor(out: Any) -> "torch.Tensor":
    """Handle both Tensor and BaseModelOutputWithPooling across transformers versions."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0, :]
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"Unsupported output type from feature extractor: {type(out)}")


class PyTorchDualEncoder:
    def __init__(self, spec: EncoderSpec, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel

        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = _load_processor_cached(spec.hf_id)
        try:
            self.model = AutoModel.from_pretrained(spec.hf_id, local_files_only=True).to(self.device).eval()
        except Exception:
            self.model = AutoModel.from_pretrained(spec.hf_id).to(self.device).eval()

    def encode_images(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        import torch

        embs = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = [img.convert("RGB") for img in images[i : i + batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                feats = _extract_feature_tensor(self.model.get_image_features(pixel_values=pixel_values))
                embs.append(feats.float().cpu().numpy())
        return _l2_norm(np.vstack(embs).astype(np.float32))

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch

        embs = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                if "siglip" in self.spec.hf_id.lower():
                    batch = [t.lower() for t in batch]
                inputs = self.processor(
                    text=batch,
                    padding="max_length",
                    max_length=self.spec.max_text_len,
                    truncation=True,
                    return_tensors="pt",
                )
                kwargs: dict[str, Any] = {"input_ids": inputs["input_ids"].to(self.device)}
                if "attention_mask" in inputs:
                    kwargs["attention_mask"] = inputs["attention_mask"].to(self.device)
                feats = _extract_feature_tensor(self.model.get_text_features(**kwargs))
                embs.append(feats.float().cpu().numpy())
        return _l2_norm(np.vstack(embs).astype(np.float32))

    def unload(self) -> None:
        import gc
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ONNXDualEncoder:
    """CPU-executed ONNX Runtime wrapper simulating mobile NPU/CPU inference."""

    def __init__(self, spec: EncoderSpec, onnx_dir: Path) -> None:
        self.spec = spec
        self.base_key = spec.base_key
        self.processor = _load_processor_cached(spec.hf_id)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4
        vis_path = onnx_dir / f"{self.base_key}_vision_int8.onnx"
        txt_path = onnx_dir / f"{self.base_key}_text_int8.onnx"
        assert vis_path.exists() and txt_path.exists(), f"Missing ONNX files in {onnx_dir} for {self.base_key}"
        self.vision_sess = ort.InferenceSession(str(vis_path), sess_opts, providers=["CPUExecutionProvider"])
        self.text_sess = ort.InferenceSession(str(txt_path), sess_opts, providers=["CPUExecutionProvider"])
        self.vis_input_names = [i.name for i in self.vision_sess.get_inputs()]
        self.txt_input_names = [i.name for i in self.text_sess.get_inputs()]

    def encode_images(self, images: list[Image.Image], batch_size: int = 8) -> np.ndarray:
        embs = []
        for i in range(0, len(images), batch_size):
            batch = [img.convert("RGB") for img in images[i : i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="np")
            feed = {"pixel_values": inputs["pixel_values"].astype(np.float32)}
            out = self.vision_sess.run(None, feed)[0]
            embs.append(out.astype(np.float32))
        return _l2_norm(np.vstack(embs))

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if "siglip" in self.spec.hf_id.lower():
                batch = [t.lower() for t in batch]
            inputs = self.processor(
                text=batch,
                padding="max_length",
                max_length=self.spec.max_text_len,
                truncation=True,
                return_tensors="np",
            )
            feed: dict[str, Any] = {"input_ids": inputs["input_ids"].astype(np.int64)}
            if "attention_mask" in self.txt_input_names and "attention_mask" in inputs:
                feed["attention_mask"] = inputs["attention_mask"].astype(np.int64)
            out = self.text_sess.run(None, feed)[0]
            embs.append(out.astype(np.float32))
        return _l2_norm(np.vstack(embs))

    def unload(self) -> None:
        del self.vision_sess
        del self.text_sess


def load_encoder(model_key: str, onnx_dir: Path | None = None, device: str | None = None):
    spec = ENCODER_SPECS[model_key]
    if spec.is_onnx:
        assert onnx_dir is not None
        return ONNXDualEncoder(spec, onnx_dir=onnx_dir)
    return PyTorchDualEncoder(spec, device=device)


def profile_encoder_latency(encoder: Any, sample_image: Image.Image, sample_text: str, runs: int = 15) -> dict[str, float]:
    # Warmup
    _ = encoder.encode_images([sample_image])
    _ = encoder.encode_texts([sample_text])
    img_times, txt_times = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = encoder.encode_images([sample_image])
        img_times.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        _ = encoder.encode_texts([sample_text])
        txt_times.append((time.perf_counter() - t1) * 1000.0)
    return {
        "image_encode_p50_ms": round(float(np.percentile(img_times, 50)), 2),
        "text_encode_p50_ms": round(float(np.percentile(txt_times, 50)), 2),
    }
