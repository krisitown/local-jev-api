ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e
FROM ${BASE_IMAGE}
# Retain NVIDIA's ARM64 CUDA-enabled PyTorch; do not replace it with CPU wheels.
RUN python3 -m pip install --no-cache-dir \
    transformers==5.17.0 accelerate==1.14.0 huggingface-hub==1.32.0 \
    safetensors==0.8.0 fastapi==0.136.3 uvicorn==0.52.4 pydantic==2.13.5 pillow==12.1.1
WORKDIR /app
COPY runtime/ /app/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "18081"]

COPY NOTICE.md /app/NOTICE.md
COPY licenses/ /app/licenses/
LABEL org.opencontainers.image.source="https://github.com/krisitown/local-jev-api"
LABEL org.opencontainers.image.description="Native local decision APIs for NVIDIA DGX Spark (ARM64); model weights downloaded separately."
