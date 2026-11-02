import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # MoviePy El SlideIn integrado no es estable con material a pantalla completa en esta cadena de procesamiento,
    # y puede darse que «la transición se aplica lógicamente, pero apenas se nota en la imagen».
    # Aquí se cambia por un fondo negro explícito + animación de desplazamiento, para que la transición sea visible y su comportamiento controlable.
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_duration = min(t, clip.duration)
    transition_start = max(clip.duration - transition_duration, 0)

    # SlideOut También se cambia por un desplazamiento explícito, para que el final del clip salga de la imagen de forma estable.
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(transition_duration, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# Se mantiene la amplitud de zoom del 20% del diseño original, para que incluso clips cortos de unos tres segundos tengan un movimiento Ken Burns claramente visible.
# La estabilidad del zoom la garantiza el muestreo centrado subpíxel de abajo, sin ocultar el parpadeo de codificación del video original reduciendo la amplitud del efecto.
_ZOOM_MAX_SCALE = 1.2


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """Usa un recorte centrado subpíxel para lograr un zoom estable y sin bordes negros.

    No se puede convertir primero el ancho y alto del recorte a enteros: cuando la proporción de zoom cambia de forma continua, los bordes enteros saltan con pasos distintos,
    y al alternar entre tamaños pares e impares cambia la fase de muestreo de medio píxel, lo que se ve como temblor en la imagen. La EXTENT
    transformación de Pillow acepta directamente bordes en coma flotante y hace el muestreo subpíxel sobre un lienzo de salida fijo; los bordes izquierdo/derecho y superior/inferior
    son siempre simétricos alrededor del mismo centro en coma flotante, así que sirve para videos con zoom lento continuo de principio a fin.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero")

    # 1 Con zoom de factor 1 se devuelve el fotograma original, para evitar que un remuestreo innecesario desenfoque ligeramente el primer fotograma.
    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        # El zoom continuo en video da más importancia a la coherencia entre fotogramas adyacentes. BICUBIC/LANCZOS dan fotogramas más nítidos por separado,
        # pero las texturas de alta frecuencia que cruzan la rejilla de muestreo tienden a producir ringing y parpadeo de brillo; BILINEAR es más suave
        # y cambia una pequeña pérdida de nitidez por una sensación dinámica más estable.
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """Amplía suavemente desde la imagen original hasta 1.2 veces a lo largo de todo el clip."""
    # t se mantiene por ahora para conservar la misma firma de llamada que las demás funciones de transición; el zoom debe cubrir todo el clip,
    # o al terminar un zoom breve la imagen se quedaría quieta de golpe, lo que no conviene para material estático o con poco movimiento.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """Reduce suavemente desde 1.2 veces hasta la imagen original a lo largo de todo el clip."""
    # Igual que zoomin_transition, t solo se usa por compatibilidad con la interfaz unificada de llamada de transiciones.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])
