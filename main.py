import os
from datetime import datetime
from sd_local import StableDiffusionLocal
from adetailer import run_adetailer
from detectors import build_available_detectors

sd = StableDiffusionLocal("models/cyberrealistic_v90")

prompt = (
    "a beautiful woman in a short red dress sitting gracefully on a mossy stone by a small creek "
    "in an idyllic forest, hands visible, natural finger joints, realistic fingernails, "
    "cinematic composition, soft depth of field, 85mm lens, high detail"
)
neg_prompt = (
    "nsfw, nude, bad anatomy, extra fingers, fused fingers, mangled hands, blurry, low quality, "
    "distorted face, overexposed, underexposed, watermark, signature, text"
)

ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = f"./output/run_{ts}"
os.makedirs(out_dir, exist_ok=True)

# 1) Basispässe mit Snapshots
img = sd.txt2img(
    prompt, neg_prompt, cfg=3.5, steps=15, size=(512, 768),
    snapshot_dir=os.path.join(out_dir, "txt2img"), snapshot_every=3
)
hi = sd.img2img(
    img, prompt, scale=2.0, strength=0.53, steps=20, cfg=1.5,
    snapshot_dir=os.path.join(out_dir, "img2img_upscale"), snapshot_every=4
)
hi.save(os.path.join(out_dir, "pre_adetail.png"))

# 2) Detektor (MediaPipe)
detectors, _ = build_available_detectors(
    mediapipe_kwargs={"min_detection_confidence": 0.5},
)

# 3) ADetailer-Refine mit Debug einschalten (nur Hände)
refined = run_adetailer(
    sd=sd, image=hi, prompt=prompt, neg_prompt=neg_prompt,
    detectors=detectors, targets=["hand"],
    denoise_strength=0.30, steps=36, cfg=5.0,
    expand_px=14, blur_px=10,
    use_edges=False, edges_image=None,
    debug_dir=os.path.join(out_dir, "adetail_debug"),
    snapshot_every=3  # Inpaint-Snapshots
)
refined.save(os.path.join(out_dir, "final_refined.png"))
print(f"[DONE] saved to {out_dir}")
