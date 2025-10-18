#!/usr/bin/env python3
# Minimal: SD 1.x -> Diffusers (lokal, ohne Downloads)
# Erwartet: ./cyberrealistic_v90.safetensors
# Ergebnis: ./models/cyberrealistic_v90/ (unet/, vae/, text_encoder/, tokenizer/, scheduler/)

import os
from pathlib import Path
import torch

CKPT = Path("./models/cyberrealistic_v90.safetensors")     # hart verdrahtet
OUT  = Path("./models/cyberrealistic_v90")          # Zielordner

def main():
    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint fehlt: {CKPT.resolve()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        # vorhandenes Ziel nicht überschreiben (absichtlich minimal)
        print(f"[SKIP] Ziel existiert bereits: {OUT.resolve()}")
    else:
        # --- Konvertierung (SD 1.x) ---
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
            download_from_original_stable_diffusion_ckpt,
        )

        pipe = download_from_original_stable_diffusion_ckpt(
            str(CKPT),  # Pfad zum safetensors-File
            from_safetensors=True,
            extract_ema=True,
            device="cpu",
        )

        pipe.save_pretrained(str(OUT), safe_serialization=True)
        print(f"[OK] Export → {OUT.resolve()}")

    # --- Offline-Ladetest der Komponenten ---
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    base = str(OUT)
    _ = AutoencoderKL.from_pretrained(f"{base}/vae", local_files_only=True)
    _ = UNet2DConditionModel.from_pretrained(f"{base}/unet", local_files_only=True)
    _ = CLIPTextModel.from_pretrained(f"{base}/text_encoder", local_files_only=True)
    _ = CLIPTokenizer.from_pretrained(f"{base}/tokenizer", local_files_only=True)
    _ = DDIMScheduler.from_pretrained(f"{base}/scheduler", local_files_only=True)

    print("[OK] Alle Teilmodelle lassen sich offline laden.")

if __name__ == "__main__":
    main()
