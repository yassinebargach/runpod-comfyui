# RunPod Qwen Image Edit Worker

This project is a custom RunPod Serverless GPU worker built with the official RunPod custom worker pattern. It runs `Qwen/Qwen-Image-Edit` through Hugging Face Diffusers, packages the worker in Docker, and caches the full model on a persistent network volume mounted at `/runpod-volume`.

The worker behavior is:

- First cold start checks for `/runpod-volume/qwen-image-edit/model_index.json`
- If the model is not cached, it downloads `Qwen/Qwen-Image-Edit` from Hugging Face with Diffusers and saves it to `/runpod-volume/qwen-image-edit`
- Later cold starts load only from the local cached directory without re-downloading
- Repeated jobs on the same live worker reuse a global in-memory pipeline singleton instead of reloading the model every request

This implementation does not use ComfyUI, does not use local quantized safetensors, and uses Diffusers installed directly from GitHub.

## Files

- `Dockerfile`: CUDA-enabled RunPod worker image
- `requirements.txt`: Python dependencies, including Diffusers from GitHub
- `src/handler.py`: RunPod Serverless handler with lazy model loading and persistent cache support
- `.dockerignore`: Keeps the Docker build context small

## Build The Docker Image

Build locally:

```bash
docker build -t your-dockerhub-user/qwen-image-edit-runpod:latest .
```

## Push The Image To A Registry

Log in and push to Docker Hub:

```bash
docker login
docker push your-dockerhub-user/qwen-image-edit-runpod:latest
```

You can use any registry RunPod supports, but Docker Hub is the simplest path for most setups.

## Create A RunPod Serverless Endpoint

1. Push the image to your container registry.
2. In RunPod, create a new Serverless endpoint.
3. Select your custom container image, for example `your-dockerhub-user/qwen-image-edit-runpod:latest`.
4. Choose a GPU suitable for image generation.
5. Mount a persistent network volume at `/runpod-volume`.
6. Deploy the endpoint.

The persistent volume is important. The worker stores the model at `/runpod-volume/qwen-image-edit`, so the first cold start downloads and caches the model, and later cold starts reuse the local copy.

## Request Schema

Input JSON:

```json
{
  "input": {
    "prompt": "string, required",
    "image": "base64-encoded input image, required",
    "negative_prompt": "string, optional",
    "num_inference_steps": "int, optional, default 30",
    "guidance_scale": "float, optional, default 4.0",
    "strength": "float, optional, default 0.8",
    "seed": "int, optional"
  }
}
```

Success response JSON:

```json
{
  "image": "base64-encoded PNG output",
  "cached_model_path": "/runpod-volume/qwen-image-edit"
}
```

Validation and inference failures return a structured `error` object.

## Example Request Payload

```json
{
  "input": {
    "prompt": "Turn this product shot into a cinematic luxury ad with warm studio lighting.",
    "image": "BASE64_IMAGE_HERE",
    "negative_prompt": "blurry, distorted, low quality, text, watermark",
    "num_inference_steps": 30,
    "guidance_scale": 4.0,
    "strength": 0.8,
    "seed": 12345
  }
}
```

## Example curl Request

Replace `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`, and `BASE64_IMAGE_HERE` with your values:

```bash
curl -X POST "https://api.runpod.ai/v2/RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Turn this portrait into a polished editorial fashion photo.",
      "image": "BASE64_IMAGE_HERE",
      "negative_prompt": "blurry, deformed, extra limbs, watermark, text",
      "num_inference_steps": 30,
      "guidance_scale": 4.0,
      "strength": 0.8,
      "seed": 7
    }
  }'
```

## Notes

- The worker prefers `torch.bfloat16` when CUDA BF16 is supported, otherwise it uses `torch.float16`.
- The model is loaded lazily in a global singleton so repeated jobs on the same worker reuse the same pipeline.
- The worker starts with `runpod.serverless.start({"handler": handler})`, which matches the standard RunPod custom worker approach.
- This project uses Qwen Image Edit via Hugging Face Diffusers.
