"""
scripts/01_generate_image.py
Generates anime-style images from a text prompt.
Supports two backends: 'replicate' (API) or 'local' (diffusers, no GPU needed).
"""

import os
import yaml
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)

# ── Prompt builder (your prompt_engine.py output) ───────────────────────────

POSITIVE_SUFFIX = (
    ", masterpiece, best quality, ultra detailed, anime style, "
    "cinematic composition, by Makoto Shinkai, Studio Ghibli palette, 4k"
)
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, worst quality, blurry, deformed, "
    "ugly, watermark, text, 3d render, realistic"
)

def build_prompt(raw: str) -> tuple[str, str]:
    return raw.strip() + POSITIVE_SUFFIX, NEGATIVE_PROMPT


# ── Backend: Replicate API ───────────────────────────────────────────────────

def generate_via_replicate(prompt: str, negative: str, cfg: dict) -> bytes:
    import replicate
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise EnvironmentError("REPLICATE_API_TOKEN not set in .env")
    os.environ["REPLICATE_API_TOKEN"] = token

    output = replicate.run(
        cfg["replicate"]["image_model"],
        input={
            "prompt": prompt,
            "negative_prompt": negative,
            "width": cfg["image"]["width"],
            "height": cfg["image"]["height"],
            "num_inference_steps": cfg["image"]["steps"],
            "guidance_scale": cfg["image"]["guidance_scale"],
            "scheduler": cfg["image"]["scheduler"],
        }
    )
    return requests.get(output[0]).content


# ── Backend: Local diffusers (no GPU, no AUTOMATIC1111) ─────────────────────

def generate_via_local(prompt: str, negative: str, cfg: dict) -> bytes:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from io import BytesIO

    model_id = cfg["local"]["model_id"]
    device   = cfg["local"]["device"]           # "cpu" or "cuda"
    dtype    = torch.float32 if device == "cpu" else torch.float16

    print(f"   Loading model '{model_id}' on {device} (first run downloads ~2GB)...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,   # disable for anime content
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # CPU memory optimization
    if device == "cpu":
        pipe.enable_attention_slicing()

    generator = torch.Generator(device).manual_seed(cfg["image"]["seed"])

    print("   Generating image (CPU mode: expect 3–8 minutes)...")
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


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_image(raw_prompt: str, output_path: str = "assets/generated_images/output.png"):
    cfg     = load_config()
    backend = cfg.get("inference_backend", "replicate")
    prompt, negative = build_prompt(raw_prompt)

    print(f"\n🖼️  Generating image via [{backend}] backend...")
    print(f"   Prompt: {prompt[:80]}...")

    if backend == "replicate":
        img_bytes = generate_via_replicate(prompt, negative, cfg)
    elif backend == "local":
        img_bytes = generate_via_local(prompt, negative, cfg)
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Use 'replicate' or 'local'.")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(img_bytes)
    print(f"   ✅ Saved → {out}")
    return str(out)


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A lone samurai on a misty mountain at dawn"
    generate_image(prompt)