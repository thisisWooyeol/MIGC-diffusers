import random

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from migc_utils import load_migc_from_safetensors
from pipeline_stable_diffusion_migc import StableDiffusionMIGCPipeline
from utils_layout import draw_layout

DEVICE = "cuda"
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    model_path = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionMIGCPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(DEVICE)
    adapter_path = hf_hub_download(repo_id="thisiswooyeol/MIGC-diffusers", filename="migc_adapter_weights.safetensors")
    load_migc_from_safetensors(pipe.unet, adapter_path)

    # Sample inference
    prompt = "bestquality, detailed, 8k.a photo of a black potted plant and a yellow refrigerator and a brown surfboard"
    phrases = [
        "a black potted plant",
        "a brown surfboard",
        "a yellow refrigerator",
    ]
    bboxes = [
        (0.5717187499999999, 0.0, 0.8179531250000001, 0.29807511737089204),
        (0.85775, 0.058755868544600943, 0.9991875, 0.646525821596244),
        (0.6041562500000001, 0.284906103286385, 0.799046875, 0.9898591549295774),
    ]

    seed_everything(SEED)
    image = pipe(
        prompt=prompt,
        phrases=phrases,
        bboxes=bboxes,
        negative_prompt="worst quality, low quality, bad anatomy",
        generator=torch.Generator(DEVICE).manual_seed(SEED),
    ).images[0]
    image.save("image.png")

    # Save image with layout
    image_with_layout = draw_layout(image, phrases, bboxes)
    image_with_layout.save("image_with_layout.png")
