"""Documentary portraits resolved by person identity, with public source credits."""

import hashlib
import html
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import requests
from PIL import Image

from app.models.schema import MaterialInfo

USER_AGENT = "MoneyPrinterTurbo/1.3.9 (documentary portraits; https://github.com/harry0703/MoneyPrinterTurbo)"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
_NON_PHOTO = re.compile(
    r"\b(drawing|painting|illustration|sculpture|statue|caricature|cartoon|"
    r"ai.generated|ai-assisted|artificial intelligence|midjourney|stable diffusion|"
    r"dibujo|pintura|ilustración|escultura|estatua|generad[oa]s? por ia)\b", re.I
)


class PortraitNotFound(ValueError):
    pass


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


def download_portrait(portrait: dict, save_dir: str, duration: int) -> MaterialInfo:
    """Bounded download, complete decode and atomic publication of a photo."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    destination = str(Path(save_dir) / (hashlib.sha256(portrait["url"].encode()).hexdigest() + ".jpg"))
    fd, staged = tempfile.mkstemp(prefix=".portrait-", suffix=".jpg", dir=save_dir)
    os.close(fd)
    try:
        with requests.get(
            portrait["url"], headers={"User-Agent": USER_AGENT},
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
            image.thumbnail((1920, 1920))
            width, height = image.size
            image.convert("RGB").save(staged, "JPEG", quality=95)
        os.replace(staged, destination)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)
    item = MaterialInfo()
    item.provider = "wikimedia"
    item.url = destination
    item.duration = duration
    item.source_info = {
        **portrait, "search_term": portrait["person"],
        "rendition": {"width": width, "height": height},
    }
    return item
