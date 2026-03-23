FROM runpod/worker-comfyui:5.8.5-base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Qwen-Image-Edit core files
# RUN comfy model download \
#   --url https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors \
#   --relative-path models/diffusion_models \
#   --filename qwen_image_edit_fp8_e4m3fn.safetensors

# RUN comfy model download \
#   --url https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors \
#   --relative-path models/loras \
#   --filename Qwen-Image-Lightning-4steps-V1.0.safetensors

# RUN comfy model download \
#   --url https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors \
#   --relative-path models/vae \
#   --filename qwen_image_vae.safetensors

# RUN comfy model download \
#   --url https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
#   --relative-path models/text_encoders \
#   --filename qwen_2.5_vl_7b_fp8_scaled.safetensors