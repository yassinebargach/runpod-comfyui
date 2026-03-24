import base64
import gc
import io
import os
from threading import Lock
from typing import Any, Dict, Optional

import runpod
import torch
from diffusers import DiffusionPipeline
from PIL import Image


MODEL_ID = "Qwen/Qwen-Image-Edit"
MODEL_DIR = os.environ.get("RUNPOD_MODEL_DIR", "/runpod-volume/qwen-image-edit")
MODEL_INDEX_FILE = os.path.join(MODEL_DIR, "model_index.json")

DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_STRENGTH = 0.8

_PIPELINE: Optional[DiffusionPipeline] = None
_PIPELINE_LOCK = Lock()


def error_response(
    message: str,
    error_type: str = "validation_error",
    missing_fields: Optional[list[str]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": message,
        }
    }
    if missing_fields:
        payload["error"]["missing_fields"] = missing_fields
    if details:
        payload["error"]["details"] = details
    return payload


def resolve_torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this worker, but no GPU is available.")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def ensure_model_cached(dtype: torch.dtype) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.isfile(MODEL_INDEX_FILE):
        return

    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        local_files_only=False,
    )
    pipeline.save_pretrained(MODEL_DIR)

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pipeline() -> DiffusionPipeline:
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        dtype = resolve_torch_dtype()
        ensure_model_cached(dtype)

        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipeline.to("cuda")
        _PIPELINE = pipeline

    return _PIPELINE


def decode_base64_image(image_base64: str) -> Image.Image:
    encoded = image_base64.strip()
    if "," in encoded and encoded.lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 image payload.") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.convert("RGB")
    except Exception as exc:
        raise ValueError("Unable to decode the provided image into a valid RGB image.") from exc


def encode_png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_seed(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("'seed' must be an integer when provided.") from exc


def parse_int(value: Any, field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be an integer.") from exc


def parse_float(value: Any, field_name: str, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be a number.") from exc


def build_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    return torch.Generator(device="cuda").manual_seed(seed)


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return error_response(
            "Missing or invalid 'input' object.",
            missing_fields=["input"],
        )

    missing_fields = [
        field for field in ("prompt", "image") if not job_input.get(field)
    ]
    if missing_fields:
        return error_response(
            "Missing required input fields.",
            missing_fields=missing_fields,
        )

    try:
        prompt = str(job_input["prompt"])
        image = decode_base64_image(str(job_input["image"]))
        negative_prompt = job_input.get("negative_prompt")
        num_inference_steps = parse_int(
            job_input.get("num_inference_steps"),
            "num_inference_steps",
            DEFAULT_NUM_INFERENCE_STEPS,
        )
        guidance_scale = parse_float(
            job_input.get("guidance_scale"),
            "guidance_scale",
            DEFAULT_GUIDANCE_SCALE,
        )
        strength = parse_float(
            job_input.get("strength"),
            "strength",
            DEFAULT_STRENGTH,
        )
        seed = parse_seed(job_input.get("seed"))
    except ValueError as exc:
        return error_response(str(exc))

    try:
        pipeline = load_pipeline()
        generator = build_generator(seed)

        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            )

        output_image = result.images[0]
        return {
            "image": encode_png_base64(output_image),
            "cached_model_path": MODEL_DIR,
        }
    except Exception as exc:
        return error_response(
            "Image generation failed.",
            error_type="inference_error",
            details={"reason": str(exc)},
        )


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
