"""
Collect openly licensed, commercially usable photos and videos from multiple verified sources:
1. Wikimedia Commons (strictly CC0 / Public Domain / CC-BY / CC-BY-SA)
2. GitHub intel-iot-devkit/sample-videos (CC-BY-4.0)
3. NASA Image & Video Library (Public Domain)

Saves raw API metadata snapshots in data/raw_metadata/ for Pass-2 double fact-checking.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_MEDIA_DIR = DATA_DIR / "raw"
RAW_META_DIR = DATA_DIR / "raw_metadata"
PASS1_MANIFEST = DATA_DIR / "manifest_pass1.json"

USER_AGENT = "MobileVisualRAG-DatasetCurator/1.0 (https://github.com/mobile-visual-rag; open-research)"

ALLOWED_LICENSES = {
    "cc0",
    "cc-zero",
    "cc0 1.0",
    "public domain",
    "pd",
    "cc by 3.0",
    "cc-by-3.0",
    "cc by 4.0",
    "cc-by-4.0",
    "cc by-sa 3.0",
    "cc-by-sa-3.0",
    "cc by-sa 4.0",
    "cc-by-sa-4.0",
}

BLOCKED_KEYWORDS = [
    "attack", "killing", "dead", "blood", "gore", "nsfw", "nude", "porn",
    "war", "corpse", "funeral", "screenshot", "diagram", "chart", "logo",
    "map of", "flag of", "coat of arms", "text page", "scanned",
]

WIKIMEDIA_PHOTO_CATEGORIES = [
    ("dog", 'filetype:bitmap incategory:"CC-Zero" (dog OR puppy) (beach OR grass OR park) -MET -museum -drawing -painting -manuscript -illustration -etching -lithograph -ceramics'),
    ("cat", 'filetype:bitmap incategory:"CC-Zero" (cat OR kitten) (sleeping OR pet OR window) -MET -museum -drawing -painting -manuscript -illustration -etching -lithograph -ceramics'),
    ("coffee", 'filetype:bitmap incategory:"CC-Zero" (coffee cup OR cappuccino OR latte) -MET -museum -drawing -painting -manuscript -illustration'),
    ("food", 'filetype:bitmap incategory:"CC-Zero" (pizza OR pasta OR bakery OR burger) -MET -museum -drawing -painting -manuscript -illustration'),
    ("beach_sunset", 'filetype:bitmap incategory:"CC-Zero" (beach sunset OR ocean sunset OR coastline) -MET -museum -drawing -painting -manuscript'),
    ("mountain_forest", 'filetype:bitmap incategory:"CC-Zero" (mountain lake OR forest trail OR waterfall) -MET -museum -drawing -painting -manuscript -map'),
    ("city_night", 'filetype:bitmap incategory:"CC-Zero" (night street OR night market OR city night) -MET -museum -drawing -painting -manuscript -map'),
    ("transport", 'filetype:bitmap incategory:"CC-Zero" (bicycle street OR passenger train OR tram) -MET -museum -drawing -painting -manuscript -diagram'),
    ("workspace_tech", 'filetype:bitmap incategory:"CC-Zero" (laptop desk OR notebook pen desk) -MET -museum -drawing -painting -manuscript -icon'),
    ("flowers", 'filetype:bitmap incategory:"CC-Zero" (cherry blossom OR sunflower OR flower garden) -MET -museum -drawing -painting -manuscript -illustration'),
]

CURATED_WIKIMEDIA_VIDEOS = [
    ("vid_chueca_night", "File:Calle de Chueca por la noche.webm", "city_night"),
    ("vid_cat_puddle", "File:Black and white cat drinks from a puddle.ogv", "cat"),
    ("vid_cat_jump_back", "File:Cat jumping backwards.webm", "cat"),
    ("vid_cat_salto", "File:Salto - Cat Jump.webm", "cat"),
]

INTEL_GITHUB_VIDEOS = [
    ("vid_intel_cars", "car-detection.mp4", "transport"),
    ("vid_intel_street_bike", "person-bicycle-car-detection.mp4", "transport"),
    ("vid_intel_fruits", "fruit-and-vegetable-detection.mp4", "food"),
    ("vid_intel_bottles", "bottle-detection.mp4", "workspace_tech"),
    ("vid_intel_store", "store-aisle-detection.mp4", "city_night"),
    ("vid_intel_classroom", "classroom.mp4", "workspace_tech"),
]

NASA_PHOTO_QUERIES = [
    ("nasa_rocket", "rocket launch pad", "space"),
    ("nasa_earth", "earth from international space station", "space"),
    ("nasa_nebula", "james webb nebula stars", "space"),
    ("nasa_mars", "mars perseverance rover surface", "space"),
]


def _clean_html(raw: str | None) -> str:
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return " ".join(text.split()).strip()


def _is_commercial_license(lic_short: str | None) -> bool:
    if not lic_short:
        return False
    norm = " ".join(lic_short.lower().replace("-", " ").split())
    if "nc" in norm.split() or "noncommercial" in norm or "fair use" in norm or "nd" in norm.split():
        return False
    for allowed in ALLOWED_LICENSES:
        if norm == allowed.replace("-", " ") or norm.startswith(allowed.replace("-", " ")):
            return True
    return "public domain" in norm or norm.startswith("cc0") or norm.startswith("cc by ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return
    parts = urllib.parse.urlsplit(url)
    safe_path = urllib.parse.quote(urllib.parse.unquote(parts.path), safe="/:@&+$,;=-._~!*()'")
    safe_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, safe_path, parts.query, parts.fragment))
    for attempt in range(4):
        try:
            req = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=45) as resp, dest.open("wb") as out:
                while True:
                    buf = resp.read(65536)
                    if not buf:
                        break
                    out.write(buf)
            return
        except urllib.error.HTTPError as he:
            if he.code == 429 and attempt < 3:
                time.sleep(2.5 * (attempt + 1))
                continue
            raise


def _wikimedia_query(params: dict[str, Any]) -> dict[str, Any]:
    q = urllib.parse.urlencode(params)
    url = f"https://commons.wikimedia.org/w/api.php?{q}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            if he.code == 429 and attempt < 3:
                time.sleep(2.5 * (attempt + 1))
                continue
            raise
    return {}


def collect_wikimedia_photos(per_cat: int = 5) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for theme, gsr in WIKIMEDIA_PHOTO_CATEGORIES:
        data = _wikimedia_query({
            "action": "query",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrlimit": 20,
            "gsrsearch": gsr,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": 1024,
            "format": "json",
        })
        pages = data.get("query", {}).get("pages", {})
        added = 0
        for _, p in pages.items():
            if added >= per_cat:
                break
            title = p.get("title", "")
            if title in seen_titles:
                continue
            low_title = title.lower()
            if any(b in low_title for b in BLOCKED_KEYWORDS) or " met " in low_title or "met_" in low_title or "metropolitan" in low_title:
                continue
            ii = (p.get("imageinfo") or [{}])[0]
            mime = ii.get("mime", "")
            if mime not in {"image/jpeg", "image/png"}:
                continue
            width, height = ii.get("width", 0), ii.get("height", 0)
            if width < 400 or height < 300:
                continue
            meta = ii.get("extmetadata", {})
            lic_short = _clean_html((meta.get("LicenseShortName") or {}).get("value"))
            if not _is_commercial_license(lic_short):
                continue
            lic_url = _clean_html((meta.get("LicenseUrl") or {}).get("value")) or "https://creativecommons.org/publicdomain/zero/1.0/"
            artist = _clean_html((meta.get("Artist") or {}).get("value")) or "Wikimedia Commons Contributor"
            desc = _clean_html((meta.get("ImageDescription") or {}).get("value"))
            dl_url = ii.get("thumburl") or ii.get("url")
            page_url = ii.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            if not dl_url:
                continue

            ext = ".jpg" if mime == "image/jpeg" else ".png"
            media_id = f"wm_{theme}_{len(records):03d}"
            dest = RAW_MEDIA_DIR / "images" / f"{media_id}{ext}"
            try:
                _download(dl_url, dest)
            except Exception as exc:
                print(f"Skip {title}: {exc}")
                continue

            raw_meta_file = RAW_META_DIR / f"{media_id}.json"
            raw_meta_file.write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")

            seen_titles.add(title)
            added += 1
            records.append({
                "media_id": media_id,
                "media_type": "image",
                "theme": theme,
                "source_platform": "Wikimedia Commons",
                "canonical_title": title,
                "page_id": p.get("pageid"),
                "author": artist,
                "license_short": lic_short,
                "license_url": lic_url,
                "commercial_allowed": True,
                "attribution_required": "cc0" not in lic_short.lower() and "public domain" not in lic_short.lower(),
                "source_page_url": page_url,
                "download_url": dl_url,
                "original_url": ii.get("url"),
                "description": desc[:300],
                "local_path": str(dest.relative_to(BASE_DIR)).replace("\\", "/"),
                "raw_metadata_path": str(raw_meta_file.relative_to(BASE_DIR)).replace("\\", "/"),
                "sha256": _sha256(dest),
                "bytes": dest.stat().st_size,
                "pass1_verified_at": datetime.now(timezone.utc).isoformat(),
            })
            time.sleep(0.15)
    return records


def collect_wikimedia_videos() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # Query curated safe videos + search additional CC0/PD nature/city videos
    titles_to_fetch = list(CURATED_WIKIMEDIA_VIDEOS)
    extra_search = _wikimedia_query({
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrlimit": 15,
        "gsrsearch": 'filetype:video ("CC0" OR "Public domain") ("ocean" OR "waves" OR "train" OR "street" OR "bird" OR "fireworks")',
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "format": "json",
    }).get("query", {}).get("pages", {})
    for _, p in extra_search.items():
        t = p.get("title", "")
        sz = (p.get("imageinfo") or [{}])[0].get("size", 0)
        if 200_000 < sz < 16_000_000 and not any(b in t.lower() for b in BLOCKED_KEYWORDS):
            if t not in [x[1] for x in titles_to_fetch]:
                titles_to_fetch.append((f"vid_wm_{len(titles_to_fetch):02d}", t, "general_video"))
        if len(titles_to_fetch) >= 8:
            break

    for media_id, title, theme in titles_to_fetch:
        data = _wikimedia_query({
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "format": "json",
        })
        pages = data.get("query", {}).get("pages", {})
        for _, p in pages.items():
            ii = (p.get("imageinfo") or [{}])[0]
            mime = ii.get("mime", "")
            if "video" not in mime and "ogg" not in mime:
                continue
            if ii.get("size", 0) > 20_000_000:
                continue
            meta = ii.get("extmetadata", {})
            lic_short = _clean_html((meta.get("LicenseShortName") or {}).get("value"))
            if not _is_commercial_license(lic_short):
                continue
            lic_url = _clean_html((meta.get("LicenseUrl") or {}).get("value")) or "https://creativecommons.org/publicdomain/zero/1.0/"
            artist = _clean_html((meta.get("Artist") or {}).get("value")) or "Wikimedia Commons Contributor"
            desc = _clean_html((meta.get("ImageDescription") or {}).get("value"))
            dl_url = ii.get("url")
            page_url = ii.get("descriptionurl")
            ext = ".webm" if "webm" in (dl_url or "").lower() else ".ogv"
            dest = RAW_MEDIA_DIR / "videos" / f"{media_id}{ext}"
            try:
                _download(dl_url, dest)
            except Exception as exc:
                print(f"Skip video {title}: {exc}")
                continue

            raw_meta_file = RAW_META_DIR / f"{media_id}.json"
            raw_meta_file.write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")
            records.append({
                "media_id": media_id,
                "media_type": "video",
                "theme": theme,
                "source_platform": "Wikimedia Commons",
                "canonical_title": title,
                "page_id": p.get("pageid"),
                "author": artist,
                "license_short": lic_short,
                "license_url": lic_url,
                "commercial_allowed": True,
                "attribution_required": "cc0" not in lic_short.lower() and "public domain" not in lic_short.lower(),
                "source_page_url": page_url,
                "download_url": dl_url,
                "original_url": dl_url,
                "description": desc[:300],
                "local_path": str(dest.relative_to(BASE_DIR)).replace("\\", "/"),
                "raw_metadata_path": str(raw_meta_file.relative_to(BASE_DIR)).replace("\\", "/"),
                "sha256": _sha256(dest),
                "bytes": dest.stat().st_size,
                "pass1_verified_at": datetime.now(timezone.utc).isoformat(),
            })
            time.sleep(0.15)
    return records


def collect_intel_github_videos() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    repo_url = "https://api.github.com/repos/intel-iot-devkit/sample-videos"
    repo_meta = json.loads(
        urllib.request.urlopen(urllib.request.Request(repo_url, headers={"User-Agent": USER_AGENT}), timeout=15).read()
    )
    spdx = (repo_meta.get("license") or {}).get("spdx_id", "")
    assert spdx == "CC-BY-4.0", f"Unexpected Intel sample-videos license: {spdx}"

    for media_id, filename, theme in INTEL_GITHUB_VIDEOS:
        dl_url = f"https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/{filename}"
        page_url = f"https://github.com/intel-iot-devkit/sample-videos/blob/master/{filename}"
        dest = RAW_MEDIA_DIR / "videos" / f"{media_id}.mp4"
        _download(dl_url, dest)
        raw_meta_file = RAW_META_DIR / f"{media_id}.json"
        raw_meta_file.write_text(json.dumps({"repo": repo_meta, "filename": filename}, indent=2), encoding="utf-8")

        records.append({
            "media_id": media_id,
            "media_type": "video",
            "theme": theme,
            "source_platform": "GitHub (intel-iot-devkit/sample-videos)",
            "canonical_title": filename,
            "page_id": repo_meta.get("id"),
            "author": "Intel Corporation (intel-iot-devkit)",
            "license_short": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "commercial_allowed": True,
            "attribution_required": True,
            "source_page_url": page_url,
            "download_url": dl_url,
            "original_url": dl_url,
            "description": f"Open-source Computer Vision sample video: {filename}",
            "local_path": str(dest.relative_to(BASE_DIR)).replace("\\", "/"),
            "raw_metadata_path": str(raw_meta_file.relative_to(BASE_DIR)).replace("\\", "/"),
            "sha256": _sha256(dest),
            "bytes": dest.stat().st_size,
            "pass1_verified_at": datetime.now(timezone.utc).isoformat(),
        })
    return records


def collect_nasa_photos() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for media_id, query, theme in NASA_PHOTO_QUERIES:
        u = f"https://images-api.nasa.gov/search?q={urllib.parse.quote(query)}&media_type=image&page_size=5"
        d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": USER_AGENT}), timeout=20).read())
        items = d.get("collection", {}).get("items", [])
        for it in items:
            dat = (it.get("data") or [{}])[0]
            links = it.get("links") or []
            thumb = next((l.get("href") for l in links if l.get("rel") == "preview"), None)
            if not thumb:
                continue
            nasa_id = dat.get("nasa_id", media_id)
            dest = RAW_MEDIA_DIR / "images" / f"{media_id}.jpg"
            try:
                _download(thumb, dest)
            except Exception as exc:
                print(f"Skip NASA {nasa_id}: {exc}")
                continue
            raw_meta_file = RAW_META_DIR / f"{media_id}.json"
            raw_meta_file.write_text(json.dumps(it, indent=2, ensure_ascii=False), encoding="utf-8")

            records.append({
                "media_id": media_id,
                "media_type": "image",
                "theme": theme,
                "source_platform": "NASA Image and Video Library",
                "canonical_title": dat.get("title", nasa_id),
                "page_id": nasa_id,
                "author": f"NASA / {dat.get('center', 'NASA')} ({dat.get('photographer') or 'Public Domain'})",
                "license_short": "Public domain (NASA Media Usage Guidelines)",
                "license_url": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
                "commercial_allowed": True,
                "attribution_required": False,
                "source_page_url": f"https://images.nasa.gov/details/{urllib.parse.quote(nasa_id)}",
                "download_url": thumb,
                "original_url": thumb,
                "description": (dat.get("description") or "")[:300],
                "local_path": str(dest.relative_to(BASE_DIR)).replace("\\", "/"),
                "raw_metadata_path": str(raw_meta_file.relative_to(BASE_DIR)).replace("\\", "/"),
                "sha256": _sha256(dest),
                "bytes": dest.stat().st_size,
                "pass1_verified_at": datetime.now(timezone.utc).isoformat(),
            })
            break
    return records


def main() -> None:
    RAW_META_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    print("Collecting Wikimedia Commons CC0 photos...")
    all_records.extend(collect_wikimedia_photos(per_cat=5))
    print("Collecting NASA Public Domain photos...")
    all_records.extend(collect_nasa_photos())
    print("Collecting Wikimedia Commons CC0/PD/CC-BY videos...")
    all_records.extend(collect_wikimedia_videos())
    print("Collecting Intel GitHub CC-BY-4.0 videos...")
    all_records.extend(collect_intel_github_videos())

    PASS1_MANIFEST.write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")
    imgs = sum(1 for r in all_records if r["media_type"] == "image")
    vids = sum(1 for r in all_records if r["media_type"] == "video")
    print(f"Pass-1 collected {len(all_records)} media assets ({imgs} images, {vids} videos) -> {PASS1_MANIFEST}")


if __name__ == "__main__":
    main()
