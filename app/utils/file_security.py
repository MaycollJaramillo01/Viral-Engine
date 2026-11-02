import os


def resolve_path_within_directory(
    base_dir: str,
    unsafe_path: str,
    *,
    require_file: bool = True,
) -> str:
    # La ruta que pasa el usuario puede ser un nombre de archivo, una ruta relativa, una ruta absoluta, o incluso incluir `../`.
    # Aquí se resuelve siempre a la ruta real y se usa commonpath para determinar si sigue dentro del directorio permitido.
    # Esto es más fiable que comprobar un simple prefijo de cadena, y cubre enlaces simbólicos, separadores duplicados, rutas relativas
    # y otros casos; sirve para directorios de lista blanca como el de subidas, el de material o el de artefactos de tareas.
    if not unsafe_path:
        raise ValueError("empty path is not allowed")

    base_dir_real = os.path.realpath(base_dir)
    candidate_path = unsafe_path
    if not os.path.isabs(candidate_path):
        candidate_path = os.path.join(base_dir_real, candidate_path)

    resolved_path = os.path.realpath(candidate_path)
    try:
        common_path = os.path.commonpath([base_dir_real, resolved_path])
    except ValueError as exc:
        # Windows En , unidades distintas provocan ValueError, y esas rutas seguro que no pertenecen al directorio permitido.
        raise ValueError("path is outside the allowed directory") from exc

    if common_path != base_dir_real:
        raise ValueError("path is outside the allowed directory")

    if require_file and not os.path.isfile(resolved_path):
        raise ValueError("file does not exist")

    return resolved_path
