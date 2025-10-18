import os
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer


class StableDiffusionLocal:
    def __init__(self, model_base: str, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base = model_base

        print(f"[INIT] Device: {self.device}")
        print(f"[LOAD] Loading models from '{model_base}' ...")

        self.vae = AutoencoderKL.from_pretrained(f"{model_base}/vae", local_files_only=True).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(f"{model_base}/unet", local_files_only=True).to(self.device)
        self.tok = CLIPTokenizer.from_pretrained(f"{model_base}/tokenizer", local_files_only=True)
        self.text_enc = CLIPTextModel.from_pretrained(f"{model_base}/text_encoder", local_files_only=True).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(f"{model_base}/scheduler", local_files_only=True)

        print("[OK] Models loaded.\n")

    # --------------------------- Embeddings ----------------------------------
    def _encode_text_embeds(self, prompt: str, negative_prompt: str = "") -> torch.Tensor:
        """Return concatenated [uncond, cond] text embeddings for CFG."""
        cond_ids = self.tok([prompt], padding="max_length", max_length=77, return_tensors="pt").to(self.device)
        uncond_ids = self.tok([negative_prompt], padding="max_length", max_length=77, return_tensors="pt").to(self.device)
        cond = self.text_enc(**cond_ids).last_hidden_state
        uncond = self.text_enc(**uncond_ids).last_hidden_state
        return torch.cat([uncond, cond], dim=0)

    # ---------------------------- Latents ------------------------------------
    @staticmethod
    def _shape_to_latent_hw(width: int, height: int) -> tuple[int, int]:
        """Convert pixel size to latent spatial dims (downscale by 8)."""
        return height // 8, width // 8

    def _init_random_latents(self, width: int, height: int) -> torch.Tensor:
        """Create standard-normal latents shaped for the UNet."""
        lh, lw = self._shape_to_latent_hw(width, height)
        return torch.randn((1, 4, lh, lw), device=self.device)

    # --------------------------- Scheduler -----------------------------------
    def _prepare_timesteps(self, steps: int) -> torch.Tensor:
        """Initialize scheduler timesteps and return them."""
        self.scheduler.set_timesteps(steps, device=self.device)
        return self.scheduler.timesteps

    # ----------------------- Noise & Denoising --------------------------------
    def _add_noise_to_latents(self, latents: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Add noise to latents at a given scheduler timestep."""
        noise = noise if noise is not None else torch.randn_like(latents)
        return self.scheduler.add_noise(latents, noise, timestep)

    def _denoise(
        self,
        latents: torch.Tensor,
        text_embeds: torch.Tensor,
        cfg_scale: float,
        timesteps: torch.Tensor,
        start_index: int = 0,
        progress_tag: str = "DENOISE",
        snapshot_dir: Optional[str] = None,
        snapshot_every: int = 0,  # 0 = off
    ) -> torch.Tensor:
        """CFG denoising loop for timesteps[start_index:]."""
        total = len(timesteps[start_index:])
        for i, t in enumerate(timesteps[start_index:]):
            pct = (i + 1) / total * 100
            print(f"\r[{progress_tag}] Step {i + 1}/{total} ({pct:5.1f}%)", end="")
            lat_in = torch.cat([latents, latents], dim=0)
            with torch.inference_mode():
                noise_pred = self.unet(lat_in, t, encoder_hidden_states=text_embeds).sample
            eps_u, eps_c = noise_pred.chunk(2)
            eps = eps_u + cfg_scale * (eps_c - eps_u)
            latents = self.scheduler.step(eps, t, latents).prev_sample

            if snapshot_dir and snapshot_every > 0:
                take = ((i + 1) % snapshot_every == 0) or (i + 1 == total)
                if take:
                    self._decode_and_save_snapshot(latents, snapshot_dir, progress_tag, i + 1, total)
        print()
        return latents

    # ------------------------- VAE encode/decode ------------------------------
    def _encode_image_to_latents(self, pil_img: Image.Image) -> torch.Tensor:
        """Encode a PIL image to latents scaled for SD (factor 0.18215)."""
        arr = np.array(pil_img).astype(np.float32) / 255.0
        arr = (arr * 2.0 - 1.0)
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            latents = self.vae.encode(img_t).latent_dist.sample()
        return latents * 0.18215

    def _decode_latents_to_image(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents to a PIL image [0..255] uint8."""
        latents = latents / 0.18215
        with torch.inference_mode():
            img = self.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img)

    # --------------------------- Public APIs ----------------------------------
    def txt2img(
        self, prompt: str, neg_prompt: str = "", steps: int = 30, cfg: float = 7.5,
        size: tuple[int, int] = (512, 512), snapshot_dir: Optional[str] = None,
        snapshot_every: int = 0
    ) -> Image.Image:
        print(f"[TXT2IMG] '{prompt}'  steps={steps} cfg={cfg} size={size}")
        w, h = size
        text_emb = self._encode_text_embeds(prompt, neg_prompt)
        latents = self._init_random_latents(w, h)
        timesteps = self._prepare_timesteps(steps)
        latents = self._denoise(
            latents, text_emb, cfg, timesteps,
            start_index=0, progress_tag="TXT2IMG",
            snapshot_dir=snapshot_dir, snapshot_every=snapshot_every
        )
        print("[TXT2IMG] Denoising complete. Decoding ...")
        return self._decode_latents_to_image(latents)

    def img2img(
        self, pil_img: Image.Image, prompt: str, scale: float = 1.0, strength: float = 0.4,
        steps: int = 20, cfg: float = 7.5, neg_prompt: str = "",
        snapshot_dir: Optional[str] = None, snapshot_every: int = 0
    ) -> Image.Image:
        """Image-to-image with initial noise injection at timestep determined by `strength`."""
        print(f"[IMG2IMG] scale={scale} strength={strength} steps={steps} cfg={cfg}")
        text_emb = self._encode_text_embeds(prompt, neg_prompt)
        w, h = pil_img.size
        W, H = int(w * scale), int(h * scale)
        up = pil_img.resize((W, H), Image.BICUBIC)

        if snapshot_dir:
            self._ensure_dir(snapshot_dir)
            self._save_image(pil_img, os.path.join(snapshot_dir, "input.png"))
            if scale != 1.0:
                self._save_image(up, os.path.join(snapshot_dir, "upscaled.png"))

        latents = self._encode_image_to_latents(up)
        timesteps = self._prepare_timesteps(steps)
        start_idx = min(int(steps * strength), len(timesteps) - 1)
        t0 = timesteps[start_idx]
        latents = self._add_noise_to_latents(latents, t0)

        latents = self._denoise(
            latents, text_emb, cfg, timesteps,
            start_index=start_idx, progress_tag="IMG2IMG",
            snapshot_dir=snapshot_dir, snapshot_every=snapshot_every
        )
        print("[IMG2IMG] Decoding ...")
        return self._decode_latents_to_image(latents)

    # --------------------- Masked / Edge-guided refine ------------------------
    def _encode_mask(self, mask_pil: Image.Image, target_size: tuple[int, int]) -> torch.Tensor:
        """white(255)=refine, black(0)=keep."""
        m = mask_pil.convert("L").resize(target_size, Image.BILINEAR)
        m = np.array(m).astype(np.float32) / 255.0
        m = torch.from_numpy(m)[None, None, ...].to(self.device)  # (1,1,H,W)
        return m.clamp(0, 1)

    def img2img_masked(
        self,
        pil_img: Image.Image,
        mask: Image.Image,
        prompt: str,
        neg_prompt: str = "",
        strength: float = 0.35,
        steps: int = 30,
        cfg: float = 5.0,
        snapshot_dir: Optional[str] = None,
        snapshot_every: int = 0,
    ) -> Image.Image:
        print(f"[INPAINT] strength={strength} steps={steps} cfg={cfg}")

        if snapshot_dir:
            self._ensure_dir(snapshot_dir)
            self._save_image(pil_img, os.path.join(snapshot_dir, "input.png"))
            self._save_image(mask.convert("L"), os.path.join(snapshot_dir, "mask.png"))

        text_embeds = self._encode_text_embeds(prompt, neg_prompt)
        base_latents = self._encode_image_to_latents(pil_img)
        timesteps = self._prepare_timesteps(steps)
        start_idx = min(int(steps * strength), len(timesteps) - 1)
        t0 = timesteps[start_idx]

        W, H = pil_img.size
        lh, lw = self._shape_to_latent_hw(W, H)
        mask_lat = self._encode_mask(mask, (lw, lh))  # (1,1,lh,lw)

        noise = torch.randn_like(base_latents)
        noised = self.scheduler.add_noise(base_latents, noise, t0)
        latents = base_latents * (1 - mask_lat) + noised * mask_lat

        total = len(timesteps[start_idx:])
        for i, t in enumerate(timesteps[start_idx:]):
            pct = (i + 1) / total * 100
            print(f"\r[INPAINT] Step {i + 1}/{total} ({pct:5.1f}%)", end="")
            lat_in = torch.cat([latents, latents], dim=0)
            with torch.inference_mode():
                noise_pred = self.unet(lat_in, t, encoder_hidden_states=text_embeds).sample
            eps_u, eps_c = noise_pred.chunk(2)
            eps = eps_u + cfg * (eps_c - eps_u)
            latents = self.scheduler.step(eps, t, latents).prev_sample
            # lock outside
            latents = latents * mask_lat + base_latents * (1 - mask_lat)

            if snapshot_dir and snapshot_every > 0:
                take = ((i + 1) % snapshot_every == 0) or (i + 1 == total)
                if take:
                    self._decode_and_save_snapshot(latents, snapshot_dir, "INPAINT", i + 1, total)

        print()
        print("[INPAINT] Decoding ...")
        return self._decode_latents_to_image(latents)

    def _blend_edges_into_image(
        self,
        pil_img: Image.Image,
        edges: Image.Image,
        mask: Image.Image,
        intensity: float = 0.6,
    ) -> Image.Image:
        edges = edges.convert("L").resize(pil_img.size, Image.BILINEAR)
        mask_l = mask.convert("L").resize(pil_img.size, Image.BILINEAR)

        edges_inv = ImageOps.invert(edges)
        edges_rgb = Image.merge("RGB", (edges_inv, edges_inv, edges_inv))
        blended = Image.blend(pil_img, edges_rgb, intensity)

        return Image.composite(blended, pil_img, mask_l)

    def edge_guided_refine(
        self, pil_img: Image.Image, mask: Image.Image, edges: Image.Image,
        prompt: str, neg_prompt: str = "", strength: float = 0.3,
        steps: int = 36, cfg: float = 5.0,
        snapshot_dir: Optional[str] = None, snapshot_every: int = 0,
    ) -> Image.Image:
        guided = self._blend_edges_into_image(pil_img, edges, mask, intensity=0.6)
        if snapshot_dir:
            self._ensure_dir(snapshot_dir)
            self._save_image(guided, os.path.join(snapshot_dir, "guided_input.png"))
            self._save_image(edges.convert("L"), os.path.join(snapshot_dir, "edges.png"))
        return self.img2img_masked(
            guided, mask, prompt, neg_prompt, strength, steps, cfg,
            snapshot_dir=snapshot_dir, snapshot_every=snapshot_every
        )

    # -------------------------- I/O helpers -----------------------------------
    def _ensure_dir(self, d: str):
        os.makedirs(d, exist_ok=True)

    def _save_image(self, pil_img: Image.Image, path: str):
        pil_img.save(path)

    def _decode_and_save_snapshot(self, latents: torch.Tensor, out_dir: str, tag: str, step_idx: int, total: int):
        self._ensure_dir(out_dir)
        img = self._decode_latents_to_image(latents)
        self._save_image(img, os.path.join(out_dir, f"{tag}_{step_idx:03d}_of_{total:03d}.png"))
