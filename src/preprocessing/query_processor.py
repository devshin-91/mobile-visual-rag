"""
Natural Language Query Preprocessing Strategies for Mobile Visual RAG.

Grounded in:
- CLIP (Radford et al., 2021, arXiv:2103.00020): prompt framing & 77-token context sensitivity
- Long-CLIP (Zhang et al., 2024, arXiv:2403.15378): effective token truncation
- ParaCLIP (Kim et al., 2024, arXiv:2402.15120): paraphrase sensitivity in CLIP text encoders
- InteractiveQR (Zhu et al., 2024, arXiv:2404.18746) & PlugIR (Lee et al., 2024, arXiv:2406.03411):
  LLM/rule query rewriting & conversational filler removal vs raw queries
- SigLIP 2 (Tschannen et al., 2025, arXiv:2502.14786): native multilingual visual-text encoding
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

StrategyName = Literal[
    "P0_raw",
    "P1_intent_clean",
    "P2_keywords",
    "P3_prompt_template",
    "P4_multi_aspect_expansion",
]

# Conversational / non-visual mobile voice & chat wrapper patterns (Korean & English)
KO_FILLER_PATTERNS = [
    r"(?:지난\s*(?:주말|달|주|해|여름|겨울|봄|가을)에\s*(?:찍은|촬영한|저장한)?)",
    r"(?:어제|오늘|최근에|작년에|아까)\s*(?:찍은|촬영한|저장한)?",
    r"(?:내\s*(?:폰|갤러리|사진첩|앨범)(?:에|에서)?\s*(?:있는)?)",
    r"(?:동영상|비디오|영상|클립|사진|이미지|스냅샷)(?:을|를|이|가|들)?",
    r"(?:좀\s*)?(?:찾아줘|보여줘|검색해줘|알려줘|찾아봐|꺼내줘|띄워줘|있을까\??|어디\s*있지\??)",
    r"(?:부탁해|빨리|혹시|그|저|이)\s*",
]

EN_FILLER_PATTERNS = [
    r"\b(?:please\s+)?(?:find|show|search|get|look\s+for|pull\s+up)\s+(?:me\s+)?",
    r"\b(?:all\s+)?(?:the\s+)?(?:photos?|pictures?|images?|videos?|clips?|footage)\s+(?:of|with|showing|from)\b",
    r"\b(?:taken|shot|recorded)\s+(?:last\s+\w+|yesterday|recently|in\s+my\s+(?:phone|gallery|camera\s+roll))\b",
    r"\bfrom\s+my\s+(?:gallery|camera\s+roll|album)\b",
]

# Lightweight bilingual visual lexicon for offline on-device Korean -> English canonicalization
# (used by English-only lightweight encoders like TinyCLIP/CLIP when no heavy LLM is available on mobile)
KO_TO_EN_VISUAL_DICT = {
    "강아지": "dog",
    "개": "dog",
    "멍멍이": "puppy dog",
    "고양이": "cat",
    "야옹이": "cat",
    "물웅덩이": "puddle of water",
    "마시는": "drinking",
    "점프하는": "jumping",
    "뛰어노는": "running and playing",
    "뛰어오르는": "jumping",
    "검은색과 흰색": "black and white",
    "흑백": "black and white",
    "검은": "black",
    "흰": "white",
    "커피": "coffee",
    "카페": "cafe",
    "라떼": "latte coffee cup",
    "커피잔": "coffee cup",
    "원두": "coffee beans",
    "음식": "food",
    "피자": "pizza",
    "빵": "bread bakery",
    "과일": "fruits and vegetables",
    "채소": "vegetables",
    "바나나": "banana",
    "사과": "apple",
    "바다": "ocean beach",
    "해변": "beach",
    "파도": "ocean waves",
    "노을": "sunset",
    "일몰": "sunset",
    "일출": "sunrise",
    "산": "mountain",
    "숲": "forest trees",
    "폭포": "waterfall",
    "나무": "trees",
    "꽃": "flower",
    "벚꽃": "cherry blossom",
    "해바라기": "sunflower",
    "밤거리": "night street",
    "밤": "night",
    "야경": "night city view",
    "거리": "street",
    "도시": "city skyline",
    "다리": "bridge",
    "자동차": "car",
    "차": "car",
    "도로": "road street",
    "자전거": "bicycle",
    "기차": "train",
    "트램": "tram",
    "교실": "classroom",
    "책상": "desk",
    "노트북": "laptop",
    "책": "books",
    "물병": "bottle",
    "병": "bottle",
    "가게": "retail store aisle",
    "마트": "store aisle",
    "사람": "person people",
    "걷는": "walking",
    "로켓": "rocket launch",
    "우주": "space",
    "지구": "planet earth from space",
    "화성": "mars rover surface",
    "성운": "nebula stars",
    "은하": "galaxy stars",
    "맑은": "sunny clear",
    "화창한": "sunny bright",
    "눈": "snow",
    "비": "rain",
    "동물": "animal",
    "새": "bird",
    "하늘": "sky",
}

VISUAL_EXPANSION_MAP = {
    "dog": ["canine pet outdoors", "happy dog portrait"],
    "cat": ["feline pet", "domestic cat indoor or outdoor"],
    "coffee": ["espresso latte mug on table", "cafe beverage"],
    "sunset": ["golden hour orange sky horizon", "evening twilight beach"],
    "beach": ["coastline sand ocean waves", "seaside shore"],
    "mountain": ["alpine peak hiking trail", "scenic mountain landscape"],
    "forest": ["green woodland trees path", "nature trail"],
    "night": ["dark evening city lights", "illuminated street at night"],
    "street": ["urban road buildings pedestrians", "city street view"],
    "car": ["automobile vehicle on road", "street traffic cars"],
    "bicycle": ["cyclist riding bike on street", "bicycle vehicle"],
    "train": ["railway locomotive tracks", "passenger train"],
    "flower": ["blooming colorful petals macro", "garden blossoms"],
    "laptop": ["computer workspace desk", "modern office desk setup"],
    "rocket": ["spacecraft liftoff exhaust flame", "nasa rocket launch"],
    "earth": ["blue planet globe from orbit", "earth horizon from space station"],
    "nebula": ["deep space colorful stars dust", "astronomical telescope image"],
    "mars": ["red rocky martian desert rover", "mars planet landscape"],
    "fruit": ["fresh apples oranges bananas display", "grocery produce"],
    "classroom": ["students desks indoor lecture room", "school classroom"],
    "store": ["supermarket shelves shopping aisle", "retail store interior"],
    "bottle": ["plastic or glass beverage bottles", "containers on shelf"],
    "jumping": ["leaping mid-air motion", "dynamic jump action"],
    "drinking": ["sipping water from puddle or bowl", "animal drinking"],
}

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "with", "for", "to", "from", "by",
    "is", "are", "was", "were", "and", "or", "that", "this", "it", "my", "me",
    "find", "show", "please", "photo", "picture", "image", "video", "clip",
    "사진", "이미지", "동영상", "영상", "비디오", "찾아줘", "보여줘", "검색", "좀", "있는", "찍은",
}


@dataclass
class ProcessedQuery:
    raw_query: str
    strategy: str
    media_type_filter: str | None = None   # "image" | "video" | None
    primary_text: str = ""
    bilingual_english_text: str = ""
    keywords: list[str] = field(default_factory=list)
    expanded_variants: list[str] = field(default_factory=list)
    extracted_temporal_hint: str | None = None


def detect_media_filter(query: str) -> str | None:
    low = query.lower()
    if any(tok in low for tok in ["동영상", "비디오", "영상", "클립", "video", "clip", "footage", "movie"]):
        return "video"
    if any(tok in low for tok in ["사진", "이미지", "스냅샷", "photo", "picture", "image", "snapshot"]):
        return "image"
    return None


def detect_temporal_hint(query: str) -> str | None:
    for hint in ["어제", "오늘", "작년", "지난주", "지난달", "yesterday", "last week", "last month", "last year"]:
        if hint in query.lower():
            return hint
    return None


def strip_conversational_wrappers(query: str) -> str:
    text = query.strip()
    for pat in KO_FILLER_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    for pat in EN_FILLER_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    # Remove dangling Korean particles at word boundaries after stripping
    text = re.sub(r"\s+(?:을|를|이|가|은|는|에서|에|의|로|으로)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ?!.,")
    return text if text else query.strip()


def translate_ko_to_en_lightweight(text: str) -> str:
    """
    On-device zero-LLM fallback translator for English-trained tiny dual encoders (e.g. TinyCLIP).
    Preserves existing English tokens while mapping known Korean visual terms cleanly.
    """
    if not re.search(r"[가-힣]", text):
        return text
    out = text
    # Sort longest phrases first so "검은색과 흰색" matches before "검은"
    for ko, en in sorted(KO_TO_EN_VISUAL_DICT.items(), key=lambda kv: -len(kv[0])):
        if ko in out:
            out = out.replace(ko, f" {en} ")
    # Remove leftover Hangul particles/suffixes if mixed
    out = re.sub(r"[가-힣]+", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out if out else text


def extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9가-힣]+", text.lower())
    kws = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    # Deduplicate preserving order
    seen: set[str] = set()
    uniq = []
    for k in kws:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def preprocess_query(
    query: str,
    strategy: StrategyName,
    model_supports_multilingual: bool = False,
) -> ProcessedQuery:
    raw = query.strip()
    media_filter = detect_media_filter(raw)
    temporal_hint = detect_temporal_hint(raw)

    if strategy == "P0_raw":
        return ProcessedQuery(
            raw_query=raw,
            strategy=strategy,
            media_type_filter=None,  # P0 does NOT apply metadata decomposition!
            primary_text=raw,
            bilingual_english_text=raw,
            keywords=extract_keywords(raw),
            expanded_variants=[raw],
            extracted_temporal_hint=None,
        )

    core = strip_conversational_wrappers(raw)
    en_core = translate_ko_to_en_lightweight(core)
    kws_ko_en = extract_keywords(core)
    kws_en = extract_keywords(en_core)

    if strategy == "P1_intent_clean":
        chosen = core if model_supports_multilingual else en_core
        return ProcessedQuery(
            raw_query=raw,
            strategy=strategy,
            media_type_filter=media_filter,
            primary_text=chosen,
            bilingual_english_text=en_core,
            keywords=kws_ko_en,
            expanded_variants=[chosen],
            extracted_temporal_hint=temporal_hint,
        )

    if strategy == "P2_keywords":
        kw_list = kws_ko_en if model_supports_multilingual else kws_en
        kw_str = ", ".join(kw_list) if kw_list else (core if model_supports_multilingual else en_core)
        return ProcessedQuery(
            raw_query=raw,
            strategy=strategy,
            media_type_filter=media_filter,
            primary_text=kw_str,
            bilingual_english_text=", ".join(kws_en) if kws_en else en_core,
            keywords=kw_list,
            expanded_variants=[kw_str],
            extracted_temporal_hint=temporal_hint,
        )

    if strategy == "P3_prompt_template":
        prefix = "a video frame showing " if media_filter == "video" else "a photo of "
        if model_supports_multilingual and re.search(r"[가-힣]", core):
            # Keep fluent native Korean + clean prompt framing (for SigLIP 2)
            primary = f"{core} 사진" if media_filter != "video" else f"{core} 동영상 장면"
        else:
            primary = f"{prefix}{en_core}"
        return ProcessedQuery(
            raw_query=raw,
            strategy=strategy,
            media_type_filter=media_filter,
            primary_text=primary,
            bilingual_english_text=f"{prefix}{en_core}",
            keywords=kws_en,
            expanded_variants=[primary],
            extracted_temporal_hint=temporal_hint,
        )

    if strategy == "P4_multi_aspect_expansion":
        prefix = "a video frame showing " if media_filter == "video" else "a photo of "
        base = core if (model_supports_multilingual and re.search(r"[가-힣]", core)) else f"{prefix}{en_core}"
        variants = [base, f"{prefix}{en_core}"]
        for kw in kws_en:
            if kw in VISUAL_EXPANSION_MAP:
                for exp in VISUAL_EXPANSION_MAP[kw]:
                    variants.append(f"{prefix}{en_core}, {exp}")
        # Deduplicate keeping max 4 variants
        dedup = []
        for v in variants:
            if v not in dedup:
                dedup.append(v)
        dedup = dedup[:4]
        return ProcessedQuery(
            raw_query=raw,
            strategy=strategy,
            media_type_filter=media_filter,
            primary_text=dedup[0],
            bilingual_english_text=f"{prefix}{en_core}",
            keywords=kws_en,
            expanded_variants=dedup,
            extracted_temporal_hint=temporal_hint,
        )

    raise ValueError(f"Unknown preprocessing strategy: {strategy}")
