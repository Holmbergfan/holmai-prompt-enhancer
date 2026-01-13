import base64
import io
import os
import time
import shutil
from typing import Optional

# Configure HuggingFace cache BEFORE importing transformers
# This must happen before any HF imports to take effect
def _configure_hf_cache():
    """Set HF cache to network volume before transformers import."""
    for path in ("/runpod-volume", "/workspace"):
        if os.path.ismount(path) or os.path.exists(path):
            hf_home = os.path.join(path, "hf")
            os.makedirs(hf_home, exist_ok=True)
            os.environ["HF_HOME"] = hf_home
            os.environ["HF_HUB_CACHE"] = hf_home
            os.environ["TRANSFORMERS_CACHE"] = hf_home
            os.environ["TORCH_HOME"] = os.path.join(path, "torch")
            return path
    return "/runpod-volume"

_EARLY_STORAGE_ROOT = _configure_hf_cache()

import requests
import runpod
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-32B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
TOP_K = int(os.getenv("TOP_K", "50"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))
DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
SYSTEM_PROMPT_PATH = os.getenv(
    "SYSTEM_PROMPT_PATH",
    "/app/prompt_enhancer_ai_instructions.md",
)
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto")

MODEL = None
PROCESSOR = None
SYSTEM_PROMPT_CACHE: Optional[str] = None


def log_disk_usage(path: str):
    try:
        total, used, free = shutil.disk_usage(path)
        print(
            "Disk usage for %s: total=%.1fGB used=%.1fGB free=%.1fGB"
            % (path, total / (1024**3), used / (1024**3), free / (1024**3))
        )
    except FileNotFoundError:
        print(f"Disk usage for {path}: path not found")


# Use the storage root configured before transformers import
STORAGE_ROOT = _EARLY_STORAGE_ROOT
print(f"Prompt enhancer storage root: {STORAGE_ROOT}")
print(f"HF_HOME={os.environ.get('HF_HOME', 'NOT SET')}")
log_disk_usage("/workspace")
log_disk_usage("/runpod-volume")


def resolve_torch_dtype():
    value = TORCH_DTYPE.lower()
    if value == "auto":
        return "auto"
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def load_system_prompt() -> str:
    global SYSTEM_PROMPT_CACHE
    if SYSTEM_PROMPT_CACHE is not None:
        return SYSTEM_PROMPT_CACHE

    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as handle:
            SYSTEM_PROMPT_CACHE = handle.read().strip()
    except FileNotFoundError:
        SYSTEM_PROMPT_CACHE = ""
    return SYSTEM_PROMPT_CACHE


def decode_image_from_input(image_input: str) -> Optional[Image.Image]:
    if not image_input:
        return None

    if not isinstance(image_input, str):
        raise ValueError("Image input must be a string.")

    image_data = None
    if image_input.startswith("http://") or image_input.startswith("https://"):
        response = requests.get(image_input, timeout=30)
        response.raise_for_status()
        image_data = response.content
    elif image_input.startswith("data:image"):
        _, encoded = image_input.split(",", 1)
        image_data = base64.b64decode(encoded)
    else:
        try:
            image_data = base64.b64decode(image_input, validate=True)
        except Exception as exc:
            raise ValueError("Unsupported image input format.") from exc

    image = Image.open(io.BytesIO(image_data))
    return image.convert("RGB")


def load_model():
    global MODEL, PROCESSOR
    if MODEL is not None and PROCESSOR is not None:
        return

    torch_dtype = resolve_torch_dtype()
    quantization_config = None
    if LOAD_IN_4BIT or LOAD_IN_8BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=DEVICE_MAP,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)


def build_messages(system_prompt: str, user_prompt: str, has_image: bool):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = []
    if has_image:
        user_content.append({"type": "image"})

    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    else:
        user_content.append({
            "type": "text",
            "text": "Create a prompt package based on the provided image."
        })

    messages.append({"role": "user", "content": user_content})
    return messages


def generate_prompt(system_prompt: str, user_prompt: str, image: Optional[Image.Image]):
    messages = build_messages(system_prompt, user_prompt, image is not None)
    prompt_text = PROCESSOR.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if image is not None:
        inputs = PROCESSOR(text=[prompt_text], images=[image], return_tensors="pt")
    else:
        inputs = PROCESSOR(text=[prompt_text], return_tensors="pt")

    inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
    }

    if TEMPERATURE <= 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
        })

    output_ids = MODEL.generate(**inputs, **gen_kwargs)

    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        trimmed = []
        for in_ids, out_ids in zip(input_ids, output_ids):
            trimmed.append(out_ids[len(in_ids):])
        output_ids = torch.stack(trimmed)

    text = PROCESSOR.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text.strip()


def handler(job):
    load_model()

    job_input = job.get("input", job) or {}
    prompt = (job_input.get("prompt") or "").strip()
    system_prompt = (job_input.get("system_prompt") or load_system_prompt()).strip()

    image_input = (
        job_input.get("image")
        or job_input.get("input_image")
        or job_input.get("input_image_url")
        or job_input.get("image_url")
    )

    if not prompt and not image_input:
        return {"error": "Provide prompt text or an input image."}

    try:
        image = decode_image_from_input(image_input) if image_input else None
    except Exception as exc:
        return {"error": str(exc)}

    start_time = time.time()
    try:
        text = generate_prompt(system_prompt, prompt, image)
    except Exception as exc:
        return {"error": str(exc)}

    elapsed_ms = int((time.time() - start_time) * 1000)
    return {
        "text": text,
        "model": MODEL_ID,
        "latency_ms": elapsed_ms,
    }


runpod.serverless.start({"handler": handler})
