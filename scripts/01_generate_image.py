"""
scripts/01_generate_image.py
Generates anime-style images using local Stable Diffusion (diffusers)
"""

import os
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# ── Load config ─────────────────────────────────────────────
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# ── Prompt builder ──────────────────────────────────────────
POSITIVE_SUFFIX = (
    ", masterpiece, best quality, ultra detailed, anime style, "
    "cinematic composition, Makoto Shinkai style, Studio Ghibli color palette, 4k, sharp focus"
)

NEGATIVE_PROMPT = (
    "lowres, bad anatomy, worst quality, blurry, deformed, ugly, watermark, text, 3d render"
)


def build_prompt(raw_prompt):
    positive = raw_prompt.strip() + POSITIVE_SUFFIX
    return positive, NEGATIVE_PROMPT


# ── Local generation (diffusers) ────────────────────────────
def generate_via_local(prompt, negative, cfg):
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from io import BytesIO

    model_id = cfg["local"]["model_id"]
    device = cfg["local"]["device"]

    dtype = torch.float32 if device == "cpu" else torch.float16

    print(f"   Loading model '{model_id}' on {device}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cpu":
        pipe.enable_attention_slicing()

    generator = torch.Generator(device).manual_seed(cfg["image"]["seed"])

    print("   Generating image...")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=cfg["image"]["width"],
        height=cfg["image"]["height"],
        num_inference_steps=cfg["image"]["steps"],
        guidance_scale=cfg["image"]["guidance_scale"],
        generator=generator,
    )

    buf = BytesIO()
    result.images[0].save(buf, format="PNG")
    return buf.getvalue()


# ── Main function ───────────────────────────────────────────
def generate_image(raw_prompt):
    cfg = load_config()
    backend = cfg.get("inference_backend", "local")

    prompt, negative = build_prompt(raw_prompt)

    print(f"\n🖼️ Generating image via [{backend}] backend...")
    print(f"   Prompt: {prompt[:100]}...")

    if backend == "local":
        img_bytes = generate_via_local(prompt, negative, cfg)
    else:
        raise ValueError("Only local backend supported for now")

    # ── Save with timestamp (NO OVERWRITE) ──────────────────
    output_dir = Path("assets/generated_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Optional: include part of prompt in filename
    safe_prompt = raw_prompt[:20].replace(" ", "_").replace(",", "")

    output_path = output_dir / f"{safe_prompt}_{timestamp}.png"

    output_path.write_bytes(img_bytes)

    print(f"   ✅ Saved → {output_path}")

    return str(output_path)


# ── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "lone samurai standing on misty mountain at sunrise"

    generate_image(prompt)