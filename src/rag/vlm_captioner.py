"""
Lightweight Edge-Compatible VLM Captioner & Grounded RAG Synthesizer.
Uses HuggingFaceTB/SmolVLM2-500M-Video-Instruct (Apache 2.0, ~500M params)
to pre-index rich visual captions/tags on-device and synthesize grounded answers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_DIR = Path(__file__).resolve().parents[2]
CAPTIONS_CACHE = BASE_DIR / "artifacts" / "captions_cache.json"
SMOLVLM2_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


class SmolVLM2Captioner:
    def __init__(self, model_id: str = SMOLVLM2_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self.device).eval()

    @torch.inference_mode()
    def describe_image(self, image: Image.Image, prompt_text: str | None = None) -> str:
        img = image.convert("RGB").copy()
        img.thumbnail((640, 640))
        question = prompt_text or (
            "Describe the main subjects, actions, colors, setting, and any visible text in this image "
            "in two concise factual sentences, followed by 5 comma-separated visual keywords."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[img], return_tensors="pt").to(self.device)
        if self.device == "cuda" and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        gen = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        out = self.processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]
        return " ".join(out.split()).strip()

    @torch.inference_mode()
    def synthesize_rag_answer(
        self,
        user_query: str,
        top_items: list[dict[str, Any]],
        max_images: int = 3,
    ) -> str:
        if not top_items:
            return "질문과 일치하는 사진 또는 동영상을 찾지 못했습니다."

        pil_images: list[Image.Image] = []
        content_blocks: list[dict[str, Any]] = []
        context_lines: list[str] = []

        for rank, item in enumerate(top_items[:max_images], start=1):
            img_path = BASE_DIR / item["thumb_path"]
            if img_path.exists():
                im = Image.open(img_path).convert("RGB")
                im.thumbnail((448, 448))
                pil_images.append(im)
                content_blocks.append({"type": "image"})
            t_info = f" (timestamp {item['timestamp_sec']:.1f}s)" if item.get("media_type") == "video" and item.get("timestamp_sec") is not None else ""
            context_lines.append(
                f"[{rank}] {item['media_id']} ({item['media_type']}{t_info}): {item.get('caption', item.get('title', ''))}"
            )

        prompt_txt = (
            f"User search query: {user_query}\n"
            f"Retrieved gallery candidates:\n" + "\n".join(context_lines) + "\n"
            "Based strictly on the shown candidate images and metadata, explain in 2 concise sentences "
            "which candidate(s) best match the user query and cite their IDs [1], [2], etc. Do not invent unseen objects."
        )
        content_blocks.append({"type": "text", "text": prompt_txt})
        messages = [{"role": "user", "content": content_blocks}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt,
            images=pil_images if pil_images else None,
            return_tensors="pt",
        ).to(self.device)
        if self.device == "cuda" and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        gen = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        out = self.processor.batch_decode(
            gen[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]
        return " ".join(out.split()).strip()

    def unload(self) -> None:
        import gc
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def precompute_captions(
    manifest_path: Path,
    keyframes_path: Path,
    cache_path: Path = CAPTIONS_CACHE,
) -> dict[str, str]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if cache_path.exists():
        existing = json.loads(cache_path.read_text(encoding="utf-8"))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keyframes = json.loads(keyframes_path.read_text(encoding="utf-8")) if keyframes_path.exists() else []

    targets: list[tuple[str, Path]] = []
    for item in manifest:
        if item["media_type"] == "image" and item["media_id"] not in existing:
            targets.append((item["media_id"], BASE_DIR / item["local_path"]))
    for kf in keyframes:
        if kf["frame_id"] not in existing:
            targets.append((kf["frame_id"], BASE_DIR / kf["frame_path"]))

    if not targets:
        print(f"All {len(existing)} captions already cached at {cache_path}")
        return existing

    print(f"Generating VLM captions for {len(targets)} items using {SMOLVLM2_ID}...")
    cap = SmolVLM2Captioner()
    for i, (key, img_path) in enumerate(targets, start=1):
        with Image.open(img_path) as im:
            desc = cap.describe_image(im)
        existing[key] = desc
        if i % 15 == 0 or i == len(targets):
            cache_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  Captioned {i}/{len(targets)}")
    cap.unload()
    return existing


if __name__ == "__main__":
    precompute_captions(
        BASE_DIR / "data" / "manifest_verified.json",
        BASE_DIR / "data" / "video_keyframes.json",
    )
