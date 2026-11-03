"""Documentary portraits resolved by person identity, with public source credits."""

import hashlib
import html
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.models.schema import MaterialInfo, VideoAspect

USER_AGENT = "MoneyPrinterTurbo/1.3.9 (documentary portraits; https://github.com/harry0703/MoneyPrinterTurbo)"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
_NON_PHOTO = re.compile(
    r"\b(drawing|painting|illustration|sculpture|statue|caricature|cartoon|"
    r"ai.generated|ai-assisted|artificial intelligence|midjourney|stable diffusion|"
    r"dibujo|pintura|ilustración|escultura|estatua|generad[oa]s? por ia)\b", re.I
)


class PortraitNotFound(ValueError):
    pass


def find_portraits(person: str, limit: int = 20) -> list[dict]:
    """Collect reviewed archive photographs before the canonical Wiki portrait.

    An identity is not an image ID: each appearance needs a different photograph.
    The bundled catalog records the actual source and author for each archive asset.
    """
    catalog_path = Path(__file__).with_name("documentary_photos.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    photos = [
        {**entry, "provider": "editorial_archive"}
        for entry in catalog
        if entry.get("reviewed") is True
        and entry.get("person", "").casefold() == person.casefold()
    ]
    try:
        primary = find_portrait(person)
    except (PortraitNotFound, requests.RequestException):
        if not photos:
            raise
    else:
        photos.append(primary)
        # Structured depicts statements constrain the gallery to the resolved
        # human, unlike a generic stock search for the person's name.
        try:
            body = _api(
                "https://commons.wikimedia.org/w/api.php", action="query",
                generator="search", gsrnamespace=6, gsrlimit=min(30, limit),
                gsrsearch=f"haswbstatement:P180={primary['asset_id']} filetype:bitmap",
                prop="imageinfo", iiprop="url|size|mime|extmetadata",
            )
        except (PortraitNotFound, requests.RequestException):
            body = {}
        for page in body.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {})
            description = " ".join(_plain(meta.get(k, {}).get("value", "")) for k in ("ImageDescription", "Categories", "ObjectName"))
            if _NON_PHOTO.search(description) or re.search(r"\b(death|dead|tomb|grave|mural|graffiti|actor|impersonator)\b", description, re.I):
                continue
            if info.get("mime") not in {"image/jpeg", "image/png", "image/webp"} or min(info.get("width", 0), info.get("height", 0)) < 320:
                continue
            url = info.get("url", "")
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname != "upload.wikimedia.org":
                continue
            license_name = _plain(meta.get("LicenseShortName", {}).get("value", ""))
            if not license_name or any(p["url"] == url for p in photos):
                continue
            photos.append({
                "url": url, "person": person, "asset_id": page.get("title", ""),
                "source_page": info.get("descriptionurl", ""),
                "creator": {"name": _plain(meta.get("Artist", {}).get("value", ""))},
                "license": license_name,
                "license_url": _plain(meta.get("LicenseUrl", {}).get("value", "")),
            })
    if not photos:
        raise PortraitNotFound(f"No hay fotografías documentadas de {person}")
    return photos[:limit]


def photo_fingerprints(image_path: str) -> tuple[int, ...]:
    """Difference hashes also reject common resized and center-cropped copies."""
    with Image.open(image_path) as source:
        gray = ImageOps.exif_transpose(source).convert("L")
        fingerprints = []
        for fraction in (1.0, 0.85, 0.7):
            width, height = gray.size
            dx, dy = width * (1 - fraction) / 2, height * (1 - fraction) / 2
            pixels = list(gray.crop((dx, dy, width - dx, height - dy)).resize((9, 8), Image.Resampling.LANCZOS).getdata())
            fingerprint = 0
            for y in range(8):
                for x in range(8):
                    fingerprint = (fingerprint << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
            fingerprints.append(fingerprint)
    return tuple(fingerprints)


def _api(url: str, **params) -> dict:
    with requests.get(
        url, params={"format": "json", **params},
        headers={"User-Agent": USER_AGENT}, timeout=(10, 30),
    ) as response:
        response.raise_for_status()
        body = response.json()
    if "error" in body:
        raise PortraitNotFound("Wikimedia could not resolve the portrait request")
    return body


def _plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", str(value or ""))).strip()


def find_portrait(person: str) -> dict:
    """Resolve an exact biography to its Wikidata human entity and P18 photos.

    Never search generic stock terms or use an actor portraying the person.
    Missing or ambiguous identities stop the job instead of inventing a face.
    """
    for language in ("es", "en"):
        body = _api(
            f"https://{language}.wikipedia.org/w/api.php", action="query",
            titles=person, redirects=1, prop="pageprops", ppprop="wikibase_item",
        )
        pages = body.get("query", {}).get("pages", {}).values()
        entity_id = next((p.get("pageprops", {}).get("wikibase_item") for p in pages), None)
        if not entity_id:
            continue
        entity = _api(
            "https://www.wikidata.org/w/api.php", action="wbgetentities",
            ids=entity_id, props="claims",
        ).get("entities", {}).get(entity_id, {})
        claims = entity.get("claims", {})
        human = any(
            c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == "Q5"
            for c in claims.get("P31", [])
        )
        if not human:
            continue
        for claim in claims.get("P18", []):
            filename = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
            if not isinstance(filename, str) or claim.get("rank") == "deprecated":
                continue
            images = _api(
                "https://commons.wikimedia.org/w/api.php", action="query",
                titles=f"File:{filename}", prop="imageinfo",
                iiprop="url|size|mime|extmetadata", iiurlwidth=1600,
            ).get("query", {}).get("pages", {}).values()
            info = next((p["imageinfo"][0] for p in images if p.get("imageinfo")), {})
            meta = info.get("extmetadata", {})
            def field(key):
                return _plain(meta.get(key, {}).get("value", ""))
            description = " ".join((filename, field("ImageDescription"), field("Categories"), field("ObjectName")))
            if info.get("mime") not in {"image/jpeg", "image/png", "image/webp"} or _NON_PHOTO.search(description):
                continue
            if min(info.get("width", 0), info.get("height", 0)) < 100:
                continue
            url = info.get("url", "") if info.get("size", 0) <= MAX_IMAGE_BYTES else info.get("thumburl", "")
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname not in {"upload.wikimedia.org", "thumb.wikimedia.org"}:
                continue
            license_name = field("LicenseShortName")
            if not license_name:
                continue
            return {
                "url": url, "person": person, "asset_id": entity_id,
                "source_page": info.get("descriptionurl", ""),
                "creator": {"name": field("Artist") or field("Credit")},
                "license": license_name, "license_url": field("LicenseUrl"),
                "attribution": field("Attribution") or field("Credit"),
                "rendition": {"width": info["width"], "height": info["height"]},
            }
    raise PortraitNotFound(
        f"No se encontró una fotografía documentada de {person}. "
        "Revisa el nombre o utiliza fotografías propias en Material local."
    )


def download_portrait(portrait: dict, save_dir: str, duration: int, video_aspect=None) -> MaterialInfo:
    """Bounded download, complete decode and atomic publication of a photo."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    identity = json.dumps([portrait["url"], portrait.get("crop"), video_aspect], sort_keys=True)
    destination = str(Path(save_dir) / (hashlib.sha256(identity.encode()).hexdigest() + ".jpg"))
    fd, staged = tempfile.mkstemp(prefix=".portrait-", suffix=".jpg", dir=save_dir)
    os.close(fd)
    try:
        with requests.get(
            portrait["url"], headers={"User-Agent": "Mozilla/5.0"},
            stream=True, timeout=(10, 60), allow_redirects=False,
        ) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise PortraitNotFound("Wikimedia returned no photograph")
            size = 0
            with open(staged, "wb") as output:
                for chunk in response.iter_content(64 * 1024):
                    size += len(chunk)
                    if size > MAX_IMAGE_BYTES:
                        raise PortraitNotFound("Wikimedia photograph exceeds the download limit")
                    output.write(chunk)
        with Image.open(staged) as image:
            image.load()
            image = ImageOps.exif_transpose(image)
            if portrait.get("crop"):
                left, top, right, bottom = portrait["crop"]
                if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                    raise PortraitNotFound("Invalid archive photograph crop")
                image = image.crop((round(left * image.width), round(top * image.height), round(right * image.width), round(bottom * image.height)))
            image.thumbnail((1920, 1920))
            if video_aspect is not None:
                # Keep the complete archive photograph visible. Center-cropping
                # a landscape group shot can remove the named protagonist.
                width, height = VideoAspect(video_aspect).to_resolution()
                background = ImageOps.fit(image.convert("RGB"), (width, height))
                background = ImageEnhance.Brightness(background.filter(ImageFilter.GaussianBlur(30))).enhance(0.6)
                foreground = ImageOps.contain(image, (int(width * 0.92), int(height * 0.92)))
                background.paste(foreground, ((width - foreground.width) // 2, (height - foreground.height) // 2))
                image = background
            width, height = image.size
            image.convert("RGB").save(staged, "JPEG", quality=95)
        os.replace(staged, destination)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)
    item = MaterialInfo()
    item.provider = portrait.get("provider", "wikimedia")
    item.url = destination
    item.duration = duration
    item.source_info = {
        **portrait, "search_term": portrait["person"],
        "rendition": {"width": width, "height": height},
    }
    return item
