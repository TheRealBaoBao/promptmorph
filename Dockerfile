FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl

RUN apt-get update \
    && apt-get install --no-install-recommends -y libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/promptmorph

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[sim]"

RUN useradd --create-home --uid 10001 promptmorph
USER promptmorph

ENTRYPOINT ["python", "-m", "promptmorph.cli"]

