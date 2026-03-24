FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    HF_HOME=/workspace/hf \
    HF_HUB_CACHE=/workspace/hf/hub \
    HF_ASSETS_CACHE=/workspace/hf/assets \
    HF_DATASETS_CACHE=/workspace/hf/datasets \
    TRANSFORMERS_CACHE=/workspace/hf/transformers \
    RUNPOD_MODEL_DIR=/app/models/qwen-image-edit \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt

COPY src /app/src
COPY README.md /app/README.md
COPY input.png /app/input.png
RUN mkdir -p /app/models/qwen-image-edit /workspace/hf/hub /workspace/hf/assets /workspace/hf/transformers /workspace/hf/datasets

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.handler:app", "--host", "0.0.0.0", "--port", "8000"]
