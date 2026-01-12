FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf \
    TRANSFORMERS_CACHE=/runpod-volume/hf \
    TORCH_HOME=/runpod-volume/torch

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py ./
COPY prompt_enhancer_ai_instructions.md ./

CMD ["python", "-u", "handler.py"]
