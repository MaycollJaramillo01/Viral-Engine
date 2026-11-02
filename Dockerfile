# Use an official Python runtime as a parent image
FROM python:3.11-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.12.12 /uv /usr/local/bin/uv

# Set the working directory in the container
WORKDIR /MoneyPrinterTurbo

# Establecer los permisos del directorio /MoneyPrinterTurbo a777
RUN chmod 777 /MoneyPrinterTurbo

ENV PYTHONPATH="/MoneyPrinterTurbo"

# Por defecto, los usuarios locales siguen priorizando los mirrors de China; al publicar la imagen GHCR desde GitHub Actions se usa default,
# para evitar que los runners fuera de China accedan lentamente a los mirrors locales y bloqueen la publicación durante mucho tiempo.
ARG DOCKER_BUILD_MIRROR=china
ARG PIP_USE_OFFICIAL=0

# La instalación de dependencias del sistema debe cumplir dos condiciones: conservar la capacidad de recurrir a mirrors en entornos de China y, si todos los mirrors
# fallan, hacer que la compilación de Docker falle de inmediato. El último sleep del bucle anterior siempre devolvía 0,
# por lo que se generaba una imagen inservible aunque git/ffmpeg no estuvieran instalados. Aquí se separan «escribir fuentes de software», «instalar»
# y «tres reintentos» en funciones shell con límites claros, y el valor de retorno de cada función decide si se continúa.
# Todas las fuentes de software usan HTTPS, para evitar que algunas redes bloqueen directamente las peticiones HTTP en texto plano.
RUN set -u; \
    write_debian_sources() { \
        main_url="$1"; \
        security_url="$2"; \
        rm -f /etc/apt/sources.list.d/debian.sources; \
        printf 'deb %s trixie main\ndeb %s trixie-updates main\ndeb %s trixie-security main\n' \
            "$main_url" "$main_url" "$security_url" > /etc/apt/sources.list; \
        rm -rf /var/lib/apt/lists/*; \
    }; \
    install_system_dependencies() { \
        apt-get update && \
        apt-get upgrade -y && \
        apt-get install -y --no-install-recommends git ffmpeg; \
    }; \
    retry_system_dependencies() { \
        attempt=1; \
        while [ "$attempt" -le 3 ]; do \
            echo "Attempt $attempt: installing system dependencies"; \
            if install_system_dependencies; then \
                return 0; \
            fi; \
            echo "Attempt $attempt failed" >&2; \
            if [ "$attempt" -lt 3 ]; then \
                echo "Retrying in 5 seconds..." >&2; \
                sleep 5; \
            fi; \
            attempt=$((attempt + 1)); \
        done; \
        return 1; \
    }; \
    if [ "$DOCKER_BUILD_MIRROR" = "china" ]; then \
        write_debian_sources \
            "https://mirrors.aliyun.com/debian" \
            "https://mirrors.aliyun.com/debian-security"; \
        if ! retry_system_dependencies; then \
            echo "Aliyun mirror failed, switching to Tsinghua mirror" >&2; \
            write_debian_sources \
                "https://mirrors.tuna.tsinghua.edu.cn/debian" \
                "https://mirrors.tuna.tsinghua.edu.cn/debian-security"; \
            if ! install_system_dependencies; then \
                echo "Tsinghua mirror failed, switching to default Debian mirror" >&2; \
                write_debian_sources \
                    "https://deb.debian.org/debian" \
                    "https://deb.debian.org/debian-security"; \
                if ! install_system_dependencies; then \
                    echo "Failed to install system dependencies from all configured mirrors" >&2; \
                    exit 1; \
                fi; \
            fi; \
        fi; \
    else \
        echo "Using default Debian mirrors"; \
        write_debian_sources \
            "https://deb.debian.org/debian" \
            "https://deb.debian.org/debian-security"; \
        if ! retry_system_dependencies; then \
            echo "Failed to install system dependencies from the default Debian mirror" >&2; \
            exit 1; \
        fi; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Copy dependency inputs first to reuse the Docker dependency layer
COPY requirements.txt requirements-security.txt ./

# Localmente se prioriza por defecto el mirror PyPI de China; la publicación en GHCR usa el PyPI oficial, para que los runners fuera de China no se ralenticen con mirrors transfronterizos.
RUN export UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=3; \
    install_python_dependencies() { \
        uv pip install --system --no-cache --overrides requirements-security.txt \
            "$@" -r requirements.txt 'setuptools>=82.0.1' 'wheel>=0.46.2'; \
    }; \
    if [ "$PIP_USE_OFFICIAL" = "1" ]; then \
        install_python_dependencies; \
    else \
        install_python_dependencies --default-index https://mirrors.aliyun.com/pypi/simple/ || \
        install_python_dependencies --default-index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ || \
        install_python_dependencies; \
    fi

# Now copy the rest of the codebase into the image
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Dentro del contenedor se debe escuchar en 0.0.0.0; el host sigue limitado a 127.0.0.1 mediante el mapeo de puertos de docker.
# browser.serverAddress solo determina la dirección de acceso mostrada en el navegador; no puede sustituir a server.address.
CMD ["streamlit", "run", "./webui/Main.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.serverAddress=127.0.0.1", "--server.enableCORS=True", "--browser.gatherUsageStats=False", "--client.toolbarMode=minimal", "--logger.hideWelcomeMessage=True", "--server.showEmailPrompt=False"]

# 1. Build the Docker image using the following command
# docker build -t moneyprinterturbo .

# 2. Run the Docker container using the following command
## For Linux or MacOS:
# docker run -v $(pwd)/config.toml:/MoneyPrinterTurbo/config.toml -v $(pwd)/storage:/MoneyPrinterTurbo/storage -p 127.0.0.1:8501:8501 moneyprinterturbo
## For Windows:
# docker run -v ${PWD}/config.toml:/MoneyPrinterTurbo/config.toml -v ${PWD}/storage:/MoneyPrinterTurbo/storage -p 127.0.0.1:8501:8501 moneyprinterturbo
