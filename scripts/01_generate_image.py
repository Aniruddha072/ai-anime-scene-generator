# -*- coding: utf-8 -*-
"""
Stage 2 — Image Generation using HuggingFace InferenceClient
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ─── Paths ─────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from prompt_engine import build_prompt

# ─── Load ENV ───────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN = os.getenv("HF_API_TOKEN")

client = InferenceClient(token=HF_TOKEN)

OUTPUT_DIR = PROJECT_ROOT / "assets" / "generated_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Generator ──────────────────────────────────────────────
def generate_image(raw_text, scene_preset=None, filename="output"):
    prompts = build_prompt(raw_text, scene_preset)

    print(f"\n[IMG] Generating image using HF client...")
    print(f"Prompt: {raw_text}")

    start = time.time()

    image = client.text_to_image(
        prompts["positive"],
        model="stabilityai/stable-diffusion-2-1"
    )

    elapsed = time.time() - start
    print(f"[OK] Generated in {elapsed:.1f}s")

    # Save image
    timestamp = int(time.time())
    path = OUTPUT_DIR / f"{filename}_{timestamp}.png"
    image.save(path)

    print(f"[SAVED] {path}")

    return [path]


# ─── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    generate_image(
        raw_text="lone samurai standing on a misty mountain peak at golden hour",
        scene_preset="dawn",
        filename="samurai_dawn"
    )