# Holmai Prompt Enhancer (RunPod)

Serverless prompt enhancer for LTX-2 using Qwen2.5-VL with image support.

## Model
Default: `Qwen/Qwen2.5-VL-32B-Instruct` (80GB GPU recommended).

## Build and Run

```bash
docker build -t holmbergfan/holmai-prompt-enhancer:latest .
```

RunPod endpoint should point to `holmbergfan/holmai-prompt-enhancer:latest`.

## Input (RunPod serverless)

```json
{
  "input": {
    "prompt": "A slow dolly-in on a scientist in a neon lab",
    "input_image": "data:image/png;base64,...",
    "system_prompt": "...optional override..."
  }
}
```

Accepted image keys: `input_image`, `input_image_url`, `image`, `image_url`.

## Output

```json
{
  "text": "ENHANCED_PROMPT ...",
  "model": "Qwen/Qwen2.5-VL-32B-Instruct",
  "latency_ms": 1234
}
```

## Environment

- `MODEL_ID` (default: `Qwen/Qwen2.5-VL-32B-Instruct`)
- `SYSTEM_PROMPT_PATH` (default: `/app/prompt_enhancer_ai_instructions.md`)
- `MAX_NEW_TOKENS` (default: `512`)
- `TEMPERATURE` (default: `0.2`)
- `TOP_P` (default: `0.9`)
- `TOP_K` (default: `50`)
- `REPETITION_PENALTY` (default: `1.05`)
- `LOAD_IN_4BIT` / `LOAD_IN_8BIT` (default: `false`)
- `TORCH_DTYPE` (default: `auto`)
- `DEVICE_MAP` (default: `auto`)

## GitHub Actions

Create secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Push to `main` to publish `holmbergfan/holmai-prompt-enhancer:latest`.
