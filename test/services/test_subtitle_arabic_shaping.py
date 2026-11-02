"""Pruebas de regresión del modelado de subtítulos en alfabeto árabe.

Estas pruebas comprueban los glifos realmente renderizados, no solo las cadenas: los signos vocálicos deben caer sobre su letra,
las letras deben ir enlazadas, el texto debe ir de derecha a izquierda, y tenga o no Pillow Raqm nunca pasa por
Pillow la maquetación de texto (así que no hay modelado doble).
"""

import functools
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from khatt.fonts import font_path as khatt_font_path
from khatt.video import TextClip as ShapedTextClip
from loguru import logger
from moviepy import ColorClip, TextClip
from PIL import ImageDraw, ImageFont

from app.services import video as vd
from app.services.utils import subtitle_shaping
from app.services.video import wrap_text

FONT_DIR = Path(__file__).resolve().parents[2] / "resource/fonts"
# Fuente de subtítulos por defecto del proyecto, sin glifos árabes, usada para cubrir la ruta de fuente de respaldo.
DEFAULT_FONT = str(FONT_DIR / "STHeitiMedium.ttc")
# Fuente que usó el mantenedor para reproducir árabe y chino en el mismo subtítulo.
YAHEI_FONT = str(FONT_DIR / "MicrosoftYaHeiBold.ttc")
MIXED_CUE = "مرحبا بالعالم 中文测试"
ARABIC_FONT = str(khatt_font_path("amiri"))
FALLBACK_FONT = str(khatt_font_path(subtitle_shaping.FALLBACK_FONT))
# khatt Directorio de las fuentes incluidas, para que generate_video elija directamente estas fuentes.
KHATT_FONT_DIR = str(khatt_font_path("amiri").parent)
RLM, LRM = subtitle_shaping.RLM, subtitle_shaping.LRM

# khatt Se emite TextOverflowWarning cuando el texto sobrepasa el lienzo y se recorta el contenido: aquí siempre cuenta como fallo.
pytestmark = pytest.mark.filterwarnings("error::khatt.TextOverflowWarning")


def shaped_clip(text, font=DEFAULT_FONT, **kwargs):
    return subtitle_shaping.text_clip(
        TextClip, text=text, font=font, font_size=100, color="white", **kwargs
    )


def ink_box(clip):
    """Región con tinta real en el clip de subtítulo (left, top, right, bottom)."""
    ink = clip.mask.get_frame(0) > 0.5
    ys, xs = np.nonzero(ink)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def ink_width(clip):
    left, _, right, _ = ink_box(clip)
    return right - left


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("مرحبا بالعالم", True),
        ("سلام دنیا", True),  # persa
        ("ٹھیک ہے", True),  # urdu
        ("Breaking news: عاجل", True),
        ("ﻻ", True),  # formas de presentación (ligadura lam-alef)
        ("﷽", True),  # ligadura de la basmala (tipo símbolo)
        ("Hello world", False),
        ("你好，世界", False),
        ("नमस्ते", False),
        ("", False),
        (None, False),
        ("\ufeffBOM", False),  # la marca de orden de bytes no es una letra árabe
        ("שלום עולם", False),  # hebreo
        ("ދިވެހި، ބަސް", False),  # el dhivehi (Thaana) usa la coma árabe
        ("ܫܠܡܐ؟", False),  # el siríaco usa el signo de interrogación árabe
        ("ߒߞߏ؛", False),  # N'Ko usa el punto y coma árabe
        ("١٢٣ ٤٥٦", False),  # solo dígitos arábigo-índicos
    ],
)
def test_only_arabic_script_text_is_shaped(text, expected):
    assert subtitle_shaping.needs_shaping(text) is expected


def test_font_without_arabic_glyphs_falls_back():
    font, fell_back = subtitle_shaping.resolve_font(DEFAULT_FONT, "مرحبا")
    assert fell_back
    assert Path(font).name == "NotoNaskhArabic-Regular.ttf"


def test_font_with_arabic_glyphs_is_kept():
    assert subtitle_shaping.resolve_font(ARABIC_FONT, "مرحبا") == (ARABIC_FONT, False)


def test_missing_emoji_alone_does_not_switch_font():
    # Si solo faltan emojis se conserva la fuente elegida por el usuario, sin cambiar a la fuente de respaldo.
    assert subtitle_shaping.resolve_font(ARABIC_FONT, "مرحبا 😀") == (ARABIC_FONT, False)


def test_non_arabic_text_goes_to_moviepy_unchanged():
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return "native clip"

    kwargs = dict(text="Hello", font=DEFAULT_FONT, font_size=60, color="white", stroke_width=2)
    assert subtitle_shaping.text_clip(native, **kwargs) == "native clip"
    assert calls == [kwargs]


def test_arabic_text_is_drawn_by_khatt():
    clip = shaped_clip("مرحبا بالعالم", stroke_color="black", stroke_width=3)
    assert isinstance(clip, ShapedTextClip)
    assert ink_width(clip) > 0


def test_never_uses_pillow_text_layout(monkeypatch):
    # Tenga o no Pillow Raqm, los subtítulos en alfabeto árabe no pasan por la maquetación de texto de Pillow,
    # así que no hay modelado doble ni resultados distintos según el entorno.
    def forbidden(*args, **kwargs):
        raise AssertionError("Pillow text layout must not be used for Arabic subtitles")

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", forbidden)
    monkeypatch.setattr(ImageDraw.ImageDraw, "multiline_text", forbidden)
    assert ink_width(shaped_clip("بِتَ مرحبا")) > 0


@pytest.mark.parametrize("font", [DEFAULT_FONT, ARABIC_FONT])
def test_vowel_marks_stay_on_their_letters(font):
    # Caso reproducido por el mantenedor: sin modelar, la Kasra sobresale del glifo más a la derecha (a 100px, ancho 107 → 145).
    plain = ink_box(shaped_clip("بت", font=font))
    marked = ink_box(shaped_clip("بِتَ", font=font))
    assert abs((marked[2] - marked[0]) - (plain[2] - plain[0])) <= 2
    # Los signos vocálicos van encima y debajo de las letras, así que la tinta solo crece en vertical.
    assert marked[1] < plain[1]
    assert marked[3] > plain[3]


def ink_runs(clip):
    """Segmentos de columnas con tinta contiguas de izquierda a derecha: una palabra enlazada es un único segmento."""
    columns = (clip.mask.get_frame(0) > 0.5).any(axis=0)
    starts = columns & ~np.concatenate(([False], columns[:-1]))
    return int(starts.sum())


@pytest.mark.parametrize("word", ["سسس", "ببب"])
def test_letters_join(word):
    # Sin modelar (Pillow sin Raqm) estas letras son tres glifos independientes; tras modelarlas forman un solo segmento.
    assert ink_runs(shaped_clip(word)) == 1
    assert ink_runs(shaped_clip(" ".join(word))) == 3


def test_right_to_left_order():
    # Alef es un trazo vertical alto y Beh un glifo bajo: al ordenar de derecha a izquierda, Alef debe quedar a la derecha.
    ink = shaped_clip("ا ب").mask.get_frame(0) > 0.5
    ys, xs = np.nonzero(ink)
    middle = (xs.min() + xs.max()) / 2
    right_height = np.ptp(ys[xs > middle])
    left_height = np.ptp(ys[xs <= middle])
    assert right_height > left_height


def strip_marks(line):
    return line.lstrip(RLM + LRM)


ARABIC_PARAGRAPH = "تمنح هذه الشهادة إلى الطالب المتميز تقديرا لجهوده الكبيرة في العام الدراسي"


def test_wrap_text_measures_shaped_arabic():
    wrapped, height = wrap_text(ARABIC_PARAGRAPH, 600, font=DEFAULT_FONT, fontsize=60)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    # Solo se salta de línea en los espacios; las palabras no se parten.
    assert " ".join(strip_marks(line) for line in lines).split() == ARABIC_PARAGRAPH.split()
    font, _ = subtitle_shaping.resolve_font(DEFAULT_FONT, ARABIC_PARAGRAPH)
    assert all(subtitle_shaping.text_width(line, font, 60) <= 600 for line in lines)
    assert height > 0


def test_arabic_wraps_by_shaped_width_even_without_raqm(monkeypatch):
    # Windows la versión de Pillow no tiene Raqm: el ancho de texto árabe medido por Pillow es el ancho de letras sueltas
    # e invertidas. El salto de línea debe seguir usando el ancho modelado: ninguna línea se pasa de ancho y se llenan de forma voraz.
    basic = functools.partial(ImageFont.truetype, layout_engine=ImageFont.Layout.BASIC)
    monkeypatch.setattr(vd.ImageFont, "truetype", basic)
    max_width = 420
    wrapped, height = wrap_text(ARABIC_PARAGRAPH, max_width, font=ARABIC_FONT, fontsize=48)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    assert " ".join(strip_marks(line) for line in lines) == ARABIC_PARAGRAPH
    for line, following in zip(lines, lines[1:] + [None]):
        assert subtitle_shaping.text_width(line, ARABIC_FONT, 48) <= max_width
        if following is not None:
            longer = f"{line} {strip_marks(following).split()[0]}"
            assert subtitle_shaping.text_width(longer, ARABIC_FONT, 48) > max_width
    assert height == len(lines) * sum(ImageFont.truetype(ARABIC_FONT, 48).getmetrics())


def test_wrapped_lines_keep_their_paragraphs_direction():
    # Las líneas partidas automáticamente siguen la dirección del párrafo original: cada línea de un párrafo de derecha a izquierda empieza con RLM,
    # y cada línea de un párrafo de izquierda a derecha con árabe intercalado empieza con LRM.
    rtl, _ = wrap_text("مرحبا بالعالم Khatt 2026 جديد " * 4, 300, font=ARABIC_FONT, fontsize=48)
    ltr, _ = wrap_text("Hello مرحبا world بالعالم " * 4, 300, font=ARABIC_FONT, fontsize=48)
    assert len(rtl.split("\n")) > 1 and len(ltr.split("\n")) > 1
    for line in rtl.split("\n"):
        assert line.startswith(RLM) and subtitle_shaping.is_rtl_paragraph(line)
    for line in ltr.split("\n"):
        assert line.startswith(LRM) and not subtitle_shaping.is_rtl_paragraph(line)


def test_hand_broken_lines_keep_their_own_direction():
    # Subtítulos bilingües: una línea en árabe y otra con la traducción al inglés, cada una con su propia dirección.
    wrapped, _ = wrap_text("مرحبا بالعالم\nHello, world!", 900, font=ARABIC_FONT, fontsize=48)
    arabic_line, english_line = wrapped.split("\n")
    assert arabic_line == RLM + "مرحبا بالعالم"
    assert strip_marks(english_line) == "Hello, world!"
    assert not subtitle_shaping.is_rtl_paragraph(english_line)


def test_translation_lines_are_measured_as_khatt_draws_them(monkeypatch):
    # Los subtítulos con letras árabes los dibuja khatt por completo, y la línea traducida al inglés también se mide con khatt.
    measured = []
    original = subtitle_shaping.text_width

    def spy(text, *args):
        measured.append(strip_marks(text))
        return original(text, *args)

    monkeypatch.setattr(subtitle_shaping, "text_width", spy)
    wrap_text("مرحبا\nHello, world!", 900, font=ARABIC_FONT, fontsize=48)
    assert "Hello, world!" in measured


def test_lines_are_measured_in_the_direction_they_render():
    # La segunda línea partida en un párrafo de derecha a izquierda empieza en inglés. El ancho de una línea mixta varía según la dirección,
    # y la medición debe usar la dirección del renderizado (de derecha a izquierda), no la deducida del inglés al inicio de línea.
    word = "الاستـ" + "ـ" * 16 + "ـقبال"  # una palabra alargada con Kashida que ocupa justo toda la primera línea
    line = "Khatt 1.5 مرحبا بكم"
    rtl_width = subtitle_shaping.text_width(RLM + line, ARABIC_FONT, 48)
    ltr_width = subtitle_shaping.text_width(LRM + line, ARABIC_FONT, 48)
    assert rtl_width < ltr_width  # premisa: medida de izquierda a derecha sería más ancha
    wrapped, _ = wrap_text(f"{word} {line}", rtl_width, font=ARABIC_FONT, fontsize=48)
    assert wrapped.split("\n") == [RLM + word, RLM + line]  # la segunda línea cabe justo


def test_arabic_words_are_never_split_across_lines():
    # El ajuste de puntuación a inicio de línea en chino/japonés movería el último carácter de la línea anterior a la siguiente; las palabras árabes
    # no pueden partirse así, o se rompería el enlace entre letras.
    text = "مرحبا بالعالم الجميل ! نعم , هذا اختبار . للتأكد ؟ من النتيجة"
    words = set(text.split())
    for max_width in range(120, 600, 15):
        wrapped, _ = wrap_text(text, max_width, font=ARABIC_FONT, fontsize=48)
        assert {w for line in wrapped.split("\n") for w in strip_marks(line).split()} <= words


def test_non_arabic_wrapping_gets_no_direction_marks():
    for text in ("Hello world from MoneyPrinterTurbo " * 4, "你好世界，这是一个很长的中文字幕句子。" * 3):
        wrapped, _ = wrap_text(text, 300, font=DEFAULT_FONT, fontsize=48)
        assert RLM not in wrapped and LRM not in wrapped


def test_webui_font_check_judges_the_font_that_draws_arabic():
    # Solo árabe: la fuente de respaldo aporta los glifos y ya no se informa por error de que la fuente no lo admite.
    assert vd.subtitle_font_supports_text(DEFAULT_FONT, "مرحبا بالعالم")
    # Árabe con chino: la fuente de respaldo dibuja el árabe y la fuente elegida el chino, y ambas se pueden dibujar.
    assert vd.subtitle_font_supports_text(DEFAULT_FONT, "مرحبا 你好")
    # La detección de otros idiomas no cambia.
    vietnamese_only = str(FONT_DIR / "BeVietnamPro-Medium.ttf")
    assert not vd.subtitle_font_supports_text(vietnamese_only, "你好世界")
    assert vd.subtitle_font_supports_text(DEFAULT_FONT, "你好世界")


class FakeAudio:
    duration = 3

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        pass

    def with_effects(self, _effects):
        return self


def render_subtitled_video(subtitles, frame_at=None, font_dir=None, **param_overrides):
    """Recorre el flujo de subtítulos de generate_video con un ColorClip real y devuelve el fotograma en el instante indicado y las llamadas al renderizador."""
    params = vd.VideoParams(
        video_subject="test",
        video_aspect="9:16",
        subtitle_enabled=True,
        font_name=param_overrides.pop("font_name", "STHeitiMedium.ttc"),
        font_size=60,
        stroke_width=2,
        bgm_type="",
        **param_overrides,
    )
    srt = "".join(
        f"{index}\n00:00:0{index - 1},000 --> 00:00:0{index},000\n{text}\n\n"
        for index, text in enumerate(subtitles, start=1)
    )
    frames = []

    def write(clip, *_args, **_kwargs):
        # generate_video Al volver se cierra el video compuesto, así que el fotograma se captura durante la escritura.
        if frame_at is not None:
            frames.append(clip.get_frame(frame_at))

    with tempfile.TemporaryDirectory() as tmp_dir:
        subtitle_path = Path(tmp_dir) / "subtitle.srt"
        subtitle_path.write_text(srt, encoding="utf-8")
        source = ColorClip((1080, 1920), color=(20, 30, 60), duration=len(subtitles))
        with (
            patch.object(vd, "_open_video_clip_quietly", return_value=source),
            patch.object(vd, "AudioFileClip", return_value=FakeAudio()),
            patch.object(vd, "TextClip", wraps=TextClip) as moviepy_renderer,
            patch.object(
                subtitle_shaping, "ShapedTextClip", wraps=ShapedTextClip
            ) as khatt_renderer,
            patch.object(vd, "_write_videofile_with_codec_fallback", side_effect=write),
            patch.object(vd, "_get_configured_video_codec", return_value="libx264"),
            patch.object(vd.utils, "font_dir", return_value=font_dir or str(FONT_DIR)),
        ):
            vd.generate_video(
                video_path="combined.mp4",
                audio_path="voice.mp3",
                subtitle_path=str(subtitle_path),
                output_file=str(Path(tmp_dir) / "final.mp4"),
                params=params,
            )
    return (frames[0] if frames else None), moviepy_renderer, khatt_renderer


def subtitle_calls(renderer):
    """create_text_clip Llamadas que dibujan realmente el subtítulo (con color y tamaño).

    No incluye las llamadas de sondeo de SubtitlesClip mediante make_textclip, ni las llamadas de medición
    de los estilos sin fondo para obtener la altura natural del subtítulo árabe.
    """
    return [
        call for call in renderer.call_args_list if "size" in call.kwargs and "color" in call.kwargs
    ]


STYLES = [
    {},
    {"text_background_color": "#000000"},
    {"text_background_color": "#000000", "rounded_subtitle_background": True},
]
STYLE_IDS = ["no-background", "background", "rounded-background"]


@pytest.mark.parametrize("overrides", STYLES, ids=STYLE_IDS)
def test_only_arabic_subtitles_go_through_khatt(overrides):
    subtitles = ["Hello world", "مرحبا بالعالم", "你好世界"]
    frame, moviepy_renderer, khatt_renderer = render_subtitled_video(
        subtitles, frame_at=1.5, **overrides
    )
    khatt_calls = subtitle_calls(khatt_renderer)
    moviepy_calls = subtitle_calls(moviepy_renderer)
    assert [call.kwargs["text"] for call in khatt_calls] == [RLM + "مرحبا بالعالم"]
    assert all(call.kwargs["font"] == FALLBACK_FONT for call in khatt_calls)
    # Las llamadas de otros idiomas son idénticas a antes del cambio: fuente original, texto original y sin marcas de dirección.
    # (En Windows generate_video cambia las barras invertidas de la ruta de fuente por barras normales, así que se compara con Path.)
    assert [call.kwargs["text"] for call in moviepy_calls] == ["Hello world", "你好世界"]
    assert all(Path(call.kwargs["font"]) == FONT_DIR / "STHeitiMedium.ttc" for call in moviepy_calls)
    assert (frame >= 250).all(axis=2).any()  # en el segundo del subtítulo árabe se dibujó texto blanco


def test_rounded_background_fits_the_shaped_arabic_text():
    _, _, khatt_renderer = render_subtitled_video(
        ["مرحبا بالعالم"], text_background_color="#000000", rounded_subtitle_background=True
    )
    (call,) = subtitle_calls(khatt_renderer)
    pad_x = int(60 * 0.4)
    expected = subtitle_shaping.text_width(call.kwargs["text"], FALLBACK_FONT, 60) + 2 * pad_x
    assert call.kwargs["size"][0] == expected  # según el ancho modelado, no el ancho de glifos faltantes de STHeiti


@pytest.mark.parametrize("overrides", STYLES, ids=STYLE_IDS)
def test_quranic_pause_marks_are_not_clipped(overrides):
    # Amiri los signos de pausa sobresalen por encima del borde superior de la fuente; la altura fija de los estilos sin fondo no puede recortarlos
    # (al recortarlos khatt emite TextOverflowWarning, que este archivo considera fallo).
    # Este versículo necesita 152px de alto a 60px con contorno de 2px, y según la fórmula solo serían 147px.
    render_subtitled_video(
        ["وَٱلضُّحَىٰ ۖ وَٱلَّيْلِ إِذَا سَجَىٰ ۗ"],
        font_dir=KHATT_FONT_DIR,
        font_name="Amiri-Regular.ttf",
        **overrides,
    )


def test_make_textclip_routes_arabic_through_khatt():
    # SubtitlesClip sondea el primer subtítulo con make_textclip.
    _, moviepy_renderer, khatt_renderer = render_subtitled_video(["مرحبا", "Hello"])
    probes = [call for call in khatt_renderer.call_args_list if "size" not in call.kwargs]
    assert probes and all(call.kwargs["font"] == FALLBACK_FONT for call in probes)
    # MoviePy sondea por sí mismo con "T" si hay máscara transparente; aparte de eso, no debería pasarse árabe a MoviePy.
    moviepy_probes = [c for c in moviepy_renderer.call_args_list if "size" not in c.kwargs]
    assert {call.kwargs["text"] for call in moviepy_probes} <= {"T"}


def test_one_arabic_font_for_the_whole_video(monkeypatch):
    # Rubik Hay letras árabes pero no la ے del urdu: todos los subtítulos árabes del video pasan a usar
    # la fuente de respaldo, en lugar de alternar entre dos fuentes subtítulo a subtítulo; el salto de línea también se mide con esa fuente.
    measured_fonts = set()
    original = subtitle_shaping.text_width

    def spy(text, font, size, *args):
        measured_fonts.add(font)
        return original(text, font, size, *args)

    monkeypatch.setattr(subtitle_shaping, "text_width", spy)
    _, _, khatt_renderer = render_subtitled_video(
        ["مرحبا", "پاکستان ے"], font_dir=KHATT_FONT_DIR, font_name="Rubik-Regular.ttf"
    )
    calls = subtitle_calls(khatt_renderer)
    assert len(calls) == 2
    assert {call.kwargs["font"] for call in calls} == {FALLBACK_FONT}
    assert measured_fonts == {FALLBACK_FONT}


def test_user_font_with_arabic_glyphs_is_used():
    _, _, khatt_renderer = render_subtitled_video(
        ["مرحبا", "سلام"], font_dir=KHATT_FONT_DIR, font_name="Rubik-Regular.ttf"
    )
    assert {Path(call.kwargs["font"]) for call in subtitle_calls(khatt_renderer)} == {
        Path(KHATT_FONT_DIR) / "Rubik-Regular.ttf"
    }


def warnings_while(render):
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        render()
    finally:
        logger.remove(sink)
    return messages


def test_font_fallback_is_logged_once_per_video():
    messages = warnings_while(
        lambda: [render_subtitled_video(["مرحبا", "بالعالم", "سلام"]) for _ in range(2)]
    )
    fallback = f"using {subtitle_shaping.FALLBACK_FONT}"
    assert sum(fallback in message for message in messages) == 2  # una vez por video


@pytest.mark.filterwarnings("ignore::khatt.UnsupportedCharacterWarning")
def test_characters_no_font_can_draw_are_reported():
    # Los caracteres que no tienen ni la fuente de respaldo ni la elegida (aquí un emoji) se omiten: el log debe indicarlo.
    messages = warnings_while(lambda: render_subtitled_video(["مرحبا 😀 你好"]))
    skipped = [message for message in messages if "skipped" in message]
    assert skipped and all("😀" in message and "你" not in message for message in skipped)


@pytest.mark.filterwarnings("error::khatt.UnsupportedCharacterWarning")
def test_mixed_arabic_and_chinese_cue_keeps_the_chinese():
    # El mantenedor informó en #1698: el chino del mismo subtítulo era omitido por la fuente de respaldo. Ahora el chino
    # lo dibuja la fuente elegida, y el chino renderizado es idéntico al dibujado solo con esa fuente.
    clip = shaped_clip(MIXED_CUE, font=YAHEI_FONT)
    chinese = ShapedTextClip(text="中文测试", font=YAHEI_FONT, font_size=100, color="white")
    chinese_width = ink_width(chinese)
    # En un párrafo de derecha a izquierda, el chino queda a la izquierda del árabe: el extremo izquierdo del fotograma son esos caracteres chinos.
    left = clip.mask.get_frame(0)[:, :chinese_width] > 0.5
    alone = chinese.mask.get_frame(0) > 0.5
    rows = np.nonzero(left.any(axis=1))[0]
    alone_rows = np.nonzero(alone.any(axis=1))[0]
    assert np.array_equal(
        left[rows.min() : rows.max() + 1],
        alone[alone_rows.min() : alone_rows.max() + 1, :chinese_width],
    )
    # El árabe sigue modelándose con la fuente de respaldo, con el mismo ancho, justo a la derecha del chino.
    arabic = shaped_clip("مرحبا بالعالم", font=YAHEI_FONT)
    assert ink_width(clip) > chinese_width + ink_width(arabic)


@pytest.mark.filterwarnings("error::khatt.UnsupportedCharacterWarning")
@pytest.mark.parametrize("overrides", STYLES, ids=STYLE_IDS)
def test_mixed_cue_in_a_video_draws_the_chinese_with_the_selected_font(overrides):
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        frame, _, khatt_renderer = render_subtitled_video(
            [MIXED_CUE], frame_at=0.5, font_name="MicrosoftYaHeiBold.ttc", **overrides
        )
    finally:
        logger.remove(sink)
    (call,) = subtitle_calls(khatt_renderer)
    assert call.kwargs["font"] == FALLBACK_FONT
    assert [Path(font) for font in call.kwargs["fallback_fonts"]] == [Path(YAHEI_FONT)]
    assert not any("skipped" in message for message in messages)
    assert (frame >= 250).all(axis=2).any()
    if overrides.get("rounded_subtitle_background"):
        # El ancho del fondo redondeado también incluye el chino.
        text_w = subtitle_shaping.text_width(
            call.kwargs["text"], FALLBACK_FONT, 60, call.kwargs["fallback_fonts"]
        )
        assert call.kwargs["size"][0] == text_w + 2 * int(60 * 0.4)


def test_wrapping_a_mixed_cue_measures_the_chinese():
    wrapped, _ = wrap_text(MIXED_CUE * 3, 500, font=YAHEI_FONT, fontsize=60)
    for line in wrapped.split("\n"):
        width = subtitle_shaping.text_width(line, FALLBACK_FONT, 60, [YAHEI_FONT])
        assert width <= 500


def test_wrap_text_leaves_other_languages_alone(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("non-Arabic subtitles must be measured as before")

    monkeypatch.setattr(subtitle_shaping, "text_width", forbidden)
    wrapped, _ = wrap_text("Hello world " * 20, 600, font=DEFAULT_FONT, fontsize=60)
    assert len(wrapped.split("\n")) > 1
