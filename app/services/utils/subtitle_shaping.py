"""Capa de adaptación del modelado (shaping) de subtítulos en alfabeto árabe.

El árabe, el persa, el urdu y otras escrituras en alfabeto árabe deben elegir primero la forma de cada letra según el contexto y enlazarlas,
y luego maquetarse de derecha a izquierda para mostrarse correctamente; los signos vocálicos también deben posicionarse sobre su letra mediante reglas OpenType.
Pillow solo hace el modelado cuando está compilado con Raqm, y las instalaciones habituales como Windows no tienen Raqm,
así que las letras salen separadas, en orden invertido y con los signos vocálicos desplazados.

Aquí los subtítulos con letras árabes se entregan a khatt (modelado HarfBuzz + rasterizado FreeType,
incluido en el propio wheel, sin depender de bibliotecas del sistema) para medirlos y dibujarlos; khatt nunca llama a la maquetación de texto de Pillow,
así que el resultado es el mismo tenga o no Pillow Raqm, y tampoco hay doble modelado.
Los demás idiomas siguen usando el TextClip nativo de MoviePy, y el resultado del renderizado no cambia.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

import khatt
from khatt.fonts import font_path as khatt_font_path
from khatt.video import TextClip as ShapedTextClip
from loguru import logger

# Bloques Unicode relacionados con el alfabeto árabe: básico, suplemento, extensiones A/B y formas de presentación A/B
# (sin la marca de orden de bytes U+FEFF).
_ARABIC_SCRIPT_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x0870, 0x089F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFC),
)

# Marca de dirección: invisible, solo determina la dirección de una línea de texto como párrafo.
RLM = "‏"
LRM = "‎"

# Ninguna de las fuentes incluidas en el proyecto tiene glifos árabes. Si a la fuente elegida le faltan caracteres se usa la que trae khatt
# Noto Naskh Arabic(SIL OFL 1.1), que cubre a la vez árabe, persa y urdu.
FALLBACK_FONT = "noto-naskh"


def _is_arabic_char(char: str) -> bool:
    # Solo se miran letras, marcas y símbolos (clases L/M/S): la coma, el signo de interrogación y los dígitos arábigo-índicos
    # también los comparten el divehi (Thaana), el siríaco y el N'Ko, así que no permiten concluir que sea alfabeto árabe.
    return unicodedata.category(char)[0] in "LMS" and any(
        start <= ord(char) <= end for start, end in _ARABIC_SCRIPT_RANGES
    )


def needs_shaping(text: str | None) -> bool:
    """Si el texto contiene letras árabes que necesitan modelado."""
    return any(_is_arabic_char(char) for char in text or "")


def is_rtl_paragraph(text: str) -> bool:
    """Determina la dirección del párrafo por el primer carácter de dirección fuerte (reglas P2/P3 del algoritmo bidireccional de Unicode)."""
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in ("R", "AL"):
            return True
        if direction == "L":
            return False
    return False


def direction_mark(text: str) -> str:
    """Marca de dirección coherente con la dirección del párrafo de ``text``.

    khatt trata cada línea como un párrafo independiente y determina su dirección por el primer carácter de dirección fuerte al inicio de la línea. Las líneas
    partidas automáticamente deben conservar la dirección del párrafo original; de lo contrario, las líneas que empiezan con inglés o números invertirían el orden de las palabras, así que
    se añade esta marca delante de cada línea.
    """
    return RLM if is_rtl_paragraph(text) else LRM


def resolve_font(font_path: str, text: str) -> tuple[str, bool]:
    """Devuelve el archivo de fuente usado realmente para dibujar ``text`` y si se recurrió a la fuente de respaldo.

    Solo se recurre al respaldo cuando a la fuente elegida le faltan glifos de letras árabes; si lo que falta son solo emojis u
    otros caracteres, se mantiene la fuente elegida por el usuario, y khatt los omite como antes y emite una advertencia.
    """
    try:
        missing = khatt.missing_characters(text, font_path)
    except (KeyError, ValueError) as exc:
        # Aunque el archivo de fuente no exista o khatt no pueda leerlo, se garantiza que los subtítulos sean legibles.
        logger.warning(f"cannot check subtitle font glyphs, using {FALLBACK_FONT}: {exc}")
        return str(khatt_font_path(FALLBACK_FONT)), True
    if any(_is_arabic_char(char) for char in missing):
        return str(khatt_font_path(FALLBACK_FONT)), True
    return font_path, False


def fallback_fonts(shaped_font: str, font_path: str) -> list[str]:
    """Al dibujar subtítulos en alfabeto árabe, qué fuentes usar para los caracteres que ``shaped_font`` no puede dibujar.

    Al usar la fuente de respaldo, los caracteres de un mismo subtítulo que esa fuente no tiene (por ejemplo, chino
    tras texto árabe) los sigue dibujando la fuente elegida por el usuario; khatt recurre carácter a carácter por tramos, sin perder caracteres.
    """
    if not font_path or shaped_font == font_path:
        return []
    try:
        khatt_font_path(font_path)
    except (KeyError, ValueError):
        return []  # No se puede leer la fuente elegida; resolve_font ya avisó de ello
    return [font_path]


def video_font(font_path: str, subtitles: Iterable[str]) -> str:
    """Determina una única fuente para todos los subtítulos en alfabeto árabe de un video.

    Si cada subtítulo eligiera su fuente por separado, los subtítulos de un mismo video podrían alternar entre dos fuentes,
    así que se determina una vez para todos los subtítulos; el uso de la fuente de respaldo y los caracteres no dibujables se avisan una sola vez cada uno.
    """
    text = "".join(subtitle for subtitle in subtitles if needs_shaping(subtitle))
    if not text:
        return font_path
    font, fell_back = resolve_font(font_path, text)
    if fell_back:
        logger.warning(
            f"subtitle font has no Arabic-script glyphs, using {FALLBACK_FONT} "
            f"for Arabic-script subtitles: {font_path}"
        )
    skipped = khatt.missing_characters(
        text, font, fallback_fonts=fallback_fonts(font, font_path)
    )
    if skipped:
        # Por ejemplo, emojis: ni la fuente de respaldo ni la elegida tienen esos glifos, y se omiten al dibujar.
        logger.warning(
            "characters in Arabic-script subtitles that the subtitle font cannot "
            f"draw are skipped: {''.join(skipped[:20])}"
        )
    return font


def supports_text(font_path: str, text: str) -> bool:
    """Si la fuente que dibuja realmente ``text`` (si hace falta, la de respaldo más la elegida) puede dibujar todos sus caracteres."""
    font, _ = resolve_font(font_path, text)
    return not khatt.missing_characters(
        text, font, fallback_fonts=fallback_fonts(font, font_path)
    )


def text_width(
    text: str, font_path: str, font_size: int, fallbacks: Iterable[str] = ()
) -> int:
    """Mide el ancho (en píxeles) de una línea de texto según los glifos reales tras el modelado; para ``fallbacks`` ver :func:`fallback_fonts`."""
    left, _, right, _ = khatt.text_bbox(
        text, font=font_path, size_px=int(font_size), fallback_fonts=list(fallbacks)
    )
    return right - left


def text_clip(native, *, shaped_font: str | None = None, **kwargs):
    """Crea el TextClip del subtítulo: el texto en alfabeto árabe lo dibuja khatt y el resto se entrega a ``native``.

    ``native`` es el ``TextClip`` de MoviePy que usa el llamador; el texto que no está en alfabeto árabe se le entrega tal cual
    , exactamente igual que antes del cambio. Los demás parámetros son los mismos que el ``TextClip`` de MoviePy 2.2.
    ``shaped_font`` es la fuente para dibujar el texto en alfabeto árabe (ver :func:`video_font`);
    si se omite se determina según ``font`` y este texto; los caracteres que no pueda dibujar los dibuja ``font``.
    """
    text = kwargs.get("text")
    if not needs_shaping(text):
        return native(**kwargs)
    font = kwargs.get("font") or ""
    if shaped_font is None:
        shaped_font, _ = resolve_font(font, text)
    return ShapedTextClip(
        **{**kwargs, "font": shaped_font},
        fallback_fonts=fallback_fonts(shaped_font, font),
    )
