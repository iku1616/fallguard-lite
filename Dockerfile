FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG MODEL_SHA256=869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MODEL_PATH=/models/yolo11n-pose.pt \
    MODEL_DEVICE=cpu

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 fallguard \
    && useradd --uid 10001 --gid fallguard --create-home --shell /usr/sbin/nologin fallguard

WORKDIR /opt/fallguard
COPY requirements.txt ./
RUN python -m pip install --index-url "${TORCH_INDEX_URL}" \
        torch==2.14.1 torchvision==0.29.1 \
    && python -m pip install --requirement requirements.txt \
    && python -m pip check

RUN mkdir --parents /models /tmp/Ultralytics \
    && chown --recursive fallguard:fallguard /models /tmp/Ultralytics /opt/fallguard

USER fallguard
WORKDIR /models
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')" \
    && echo "${MODEL_SHA256}  yolo11n-pose.pt" | sha256sum --check \
    && echo "${MODEL_SHA256}  yolo11n-pose.pt" > yolo11n-pose.pt.sha256

WORKDIR /opt/fallguard
COPY --chown=fallguard:fallguard app ./app

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=4s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
