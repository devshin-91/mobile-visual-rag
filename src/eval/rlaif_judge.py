"""
RLAIF Evaluation & Closed-Loop Reward Calibration Module.

References:
- Bai et al., 2022 (arXiv:2212.08073): Constitutional AI
- Lee et al., 2023 (arXiv:2309.00267): RLAIF vs. RLHF
- Yu et al., 2024 (arXiv:2405.17220): RLAIF-V
- Chen et al., 2024 (arXiv:2402.04788): MLLM-as-a-Judge
- Gu et al., 2024 (arXiv:2411.15594): A Survey on LLM-as-a-Judge (Position Bias Mitigation)

Uses independent Apache-2.0 VLM (Qwen/Qwen2-VL-2B-Instruct) to score retrieved visual candidates,
run AB/BA swapped pairwise preferences, and train a Bradley-Terry Reward Reranker on Train queries
to verify closed-loop RLAIF improvements on Held-Out Test queries.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parents[2]
QWEN_JUDGE_ID = "Qwen/Qwen2-VL-2B-Instruct"

VISUAL_CONSTITUTION = """
Constitutional Evaluation Principles for Mobile Photo & Video Visual RAG:
1. Visual Grounding (0-5): Are the key entities/objects requested in the query clearly visible in the image/frame?
2. Action & Attribute Precision (0-5): Do the colors, actions, setting, lighting, and spatial relationships match?
3. Media Type & Intent Alignment (0-5): If the query asks for video/motion or photo/still, does the candidate media type fit?
4. Overall Utility (0-5): Would a real smartphone user be satisfied seeing this as a top search result?
"""


def _parse_json_score(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return {
                "entity": float(np.clip(float(obj.get("entity", 0)), 0, 5)),
                "attribute": float(np.clip(float(obj.get("attribute", 0)), 0, 5)),
                "media_fit": float(np.clip(float(obj.get("media_fit", 0)), 0, 5)),
                "overall": float(np.clip(float(obj.get("overall", 0)), 0, 5)),
                "reason": str(obj.get("reason", ""))[:200],
            }
        except Exception:
            pass
    nums = [float(x) for x in re.findall(r"\b([0-5](?:\.\d+)?)\b", text)]
    val = nums[-1] if nums else 2.0
    return {"entity": val, "attribute": val, "media_fit": val, "overall": val, "reason": text[:160]}


class QwenVLMConstitutionalJudge:
    def __init__(self, model_id: str = QWEN_JUDGE_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=128 * 28 * 28,
            max_pixels=384 * 28 * 28,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self.device).eval()

    @torch.inference_mode()
    def score_candidate(
        self,
        query: str,
        image_path: Path,
        candidate_media_type: str,
        expected_media_type: str | None = None,
    ) -> dict[str, Any]:
        with Image.open(image_path) as raw_im:
            im = raw_im.convert("RGB").copy()
            im.thumbnail((448, 448))

        prompt_text = (
            f"{VISUAL_CONSTITUTION}\n"
            f"User Search Query: \"{query}\"\n"
            f"Candidate Media Type: {candidate_media_type} (Query requested media constraint: {expected_media_type or 'any'})\n"
            "Score this retrieved image strictly according to the visual evidence you see. "
            "Respond ONLY with valid JSON: "
            '{"entity": <0-5>, "attribute": <0-5>, "media_fit": <0-5>, "overall": <0-5>, "reason": "<short factual reason>"}'
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": im},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[chat_text],
            images=[im],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        gen = self.model.generate(**inputs, max_new_tokens=110, do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1] :]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return _parse_json_score(out)

    @torch.inference_mode()
    def pairwise_compare_debiased(
        self,
        query: str,
        img_a_path: Path,
        type_a: str,
        img_b_path: Path,
        type_b: str,
    ) -> dict[str, Any]:
        """
        Position-bias-mitigated pairwise judgment (AB + BA swap).
        Returns 'A', 'B', or 'TIE' (if judge disagrees upon position swap or declares tie).
        """
        def _single_pass(first_path: Path, first_t: str, second_path: Path, second_t: str) -> str:
            with Image.open(first_path) as im1_r, Image.open(second_path) as im2_r:
                im1 = im1_r.convert("RGB").copy()
                im2 = im2_r.convert("RGB").copy()
                im1.thumbnail((384, 384))
                im2.thumbnail((384, 384))

            prompt = (
                f"User Search Query: \"{query}\"\n"
                f"Candidate [1] is a {first_t}. Candidate [2] is a {second_t}.\n"
                "Which candidate visually matches the user query better? "
                'Reply strictly with JSON: {"winner": "1" | "2" | "TIE", "reason": "<brief>"}'
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": im1},
                    {"type": "image", "image": im2},
                    {"type": "text", "text": prompt},
                ],
            }]
            txt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[txt], images=[im1, im2], padding=True, return_tensors="pt").to(self.device)
            gen = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
            out = self.processor.batch_decode(gen[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
            if '"winner": "1"' in out or "'1'" in out or "winner: 1" in out.lower():
                return "1"
            if '"winner": "2"' in out or "'2'" in out or "winner: 2" in out.lower():
                return "2"
            return "TIE"

        pass_ab = _single_pass(img_a_path, type_a, img_b_path, type_b)
        pass_ba = _single_pass(img_b_path, type_b, img_a_path, type_a)
        mapped_ba = "A" if pass_ba == "2" else ("B" if pass_ba == "1" else "TIE")
        mapped_ab = "A" if pass_ab == "1" else ("B" if pass_ab == "2" else "TIE")

        if mapped_ab == mapped_ba:
            return {"winner": mapped_ab, "consistent": True, "pass_ab": pass_ab, "pass_ba": pass_ba}
        return {"winner": "TIE", "consistent": False, "pass_ab": pass_ab, "pass_ba": pass_ba}

    def unload(self) -> None:
        import gc
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_bradley_terry_reward_reranker(
    feature_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Closed-loop RLAIF Reward Calibration:
    Fits a Bradley-Terry pairwise preference model P(chosen > rejected) = sigmoid(w^T (x_c - x_r))
    using AI Feedback preference pairs collected ONLY on the Train query split.
    """
    if len(feature_pairs) < 4:
        return np.array([0.55, 0.25, 0.12, 0.08, 0.08, 0.08, 0.12], dtype=np.float32)

    X_diff = []
    y = []
    for x_c, x_r in feature_pairs:
        diff = x_c - x_r
        X_diff.append(diff)
        y.append(1)
        X_diff.append(-diff)
        y.append(0)

    X_arr = np.vstack(X_diff)
    y_arr = np.array(y)
    clf = LogisticRegression(C=0.5, fit_intercept=False, max_iter=500, random_state=42)
    clf.fit(X_arr, y_arr)
    w = clf.coef_[0].astype(np.float32)
    # Blend slightly with prior heuristic so uninformative dimensions stay non-negative/stable
    prior = np.array([0.50, 0.25, 0.10, 0.06, 0.08, 0.06, 0.10], dtype=np.float32)
    w_norm = w / max(float(np.linalg.norm(w)), 1e-6)
    blended = 0.70 * np.maximum(w_norm, 0.0) + 0.30 * prior
    return blended.astype(np.float32)
