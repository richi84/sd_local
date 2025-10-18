# adetailer.py
import math
import os, json
from typing import List, Optional, Literal, Tuple

from PIL import Image, ImageDraw

from detectors import BaseDetector, Detection, filter_by_targets, boxes_to_mask

Targets = List[Literal["face","hand","person"]]


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    img_size: Tuple[int, int],
    fraction: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w == 0 and h == 0:
        return bbox

    expand_x = int(math.ceil(w * fraction)) if w > 0 else 0
    expand_y = int(math.ceil(h * fraction)) if h > 0 else 0

    img_w, img_h = img_size
    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y)
    new_x2 = min(img_w - 1, x2 + expand_x)
    new_y2 = min(img_h - 1, y2 + expand_y)

    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return bbox

    return new_x1, new_y1, new_x2, new_y2


def _save_overlay(image: Image.Image, boxes, path: str, *, extra_boxes=None):
    ov = image.copy()
    dr = ImageDraw.Draw(ov)
    for (x1,y1,x2,y2) in boxes:
        dr.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)
    if extra_boxes:
        for (x1, y1, x2, y2) in extra_boxes:
            dr.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=3)
    ov.save(path)


def _save_crops(before: Image.Image, after: Image.Image, boxes, dir_path: str):
    _ensure_dir(dir_path)
    for k, (x1,y1,x2,y2) in enumerate(boxes, start=1):
        b = before.crop((x1,y1,x2,y2))
        a = after.crop((x1,y1,x2,y2))
        b.save(os.path.join(dir_path, f"crop_{k:02d}_before.png"))
        a.save(os.path.join(dir_path, f"crop_{k:02d}_after.png"))


def run_adetailer(
    sd,                         # StableDiffusionLocal instance
    image: Image.Image,
    prompt: str,
    neg_prompt: str = "",
    detectors: List[BaseDetector] = [],
    targets: Targets = ["hand", "face"],
    denoise_strength: float = 0.32,
    steps: int = 32,
    cfg: float = 4.5,
    expand_px: int = 10,
    blur_px: int = 8,
    use_edges: bool = False,
    edges_image: Optional[Image.Image] = None,
    # --- NEW: debugging & snapshots ---
    debug_dir: Optional[str] = None,
    snapshot_every: int = 0,
) -> Image.Image:
    """
    1) run detectors -> boxes
    2) build unified mask
    3) masked img2img (optionally edge-guided)
    Saves all intermediates if debug_dir is provided.
    """
    if debug_dir:
        _ensure_dir(debug_dir)
        image.save(os.path.join(debug_dir, "input_raw.png"))

    # ---- detection
    all_dets: List[Detection] = []
    det_logs = []
    for det in detectors:
        name = det.__class__.__name__
        try:
            ds = det.detect(image, targets)
            all_dets.extend(ds)
            det_logs.append({"detector": name, "count": len(ds)})
        except Exception as e:
            msg = f"Detector {name} failed: {e}"
            print("[ADetailer]", msg)
            det_logs.append({"detector": name, "error": str(e)})

    sel = filter_by_targets(all_dets, targets, min_score=0.2)
    boxes_original = [d.bbox for d in sel]
    expanded_boxes = [
        _expand_bbox(bbox, image.size, 0.10)
        for bbox in boxes_original
    ]
    boxes = expanded_boxes

    if debug_dir:
        # detections.json
        with open(os.path.join(debug_dir, "detections.json"), "w") as f:
            json.dump({
                "logs": det_logs,
                "selected": [
                    {
                        "target": d.target,
                        "score": d.score,
                        "bbox": d.bbox,
                        "expanded_bbox": expanded_boxes[idx] if idx < len(expanded_boxes) else d.bbox,
                    }
                    for idx, d in enumerate(sel)
                ]
            }, f, indent=2)
        # overlay of raw detections
        overlay_path = os.path.join(debug_dir, "input.png")
        if boxes:
            _save_overlay(image, boxes_original, overlay_path, extra_boxes=boxes)
        else:
            image.save(overlay_path)

    if not sel:
        print("[ADetailer] No detections. Skipping refine.")
        return image

    # ---- mask
    mask = boxes_to_mask(size=image.size, boxes=boxes, expand=expand_px, blur=blur_px)
    if debug_dir:
        mask.save(os.path.join(debug_dir, "mask.png"))

    # ---- refine
    if use_edges and edges_image is not None:
        result = sd.edge_guided_refine(
            pil_img=image, mask=mask, edges=edges_image,
            prompt=prompt, neg_prompt=neg_prompt,
            strength=denoise_strength, steps=steps, cfg=cfg,
            snapshot_dir=os.path.join(debug_dir, "inpaint") if debug_dir else None,
            snapshot_every=snapshot_every
        )
    else:
        result = sd.img2img_masked(
            pil_img=image, mask=mask, prompt=prompt, neg_prompt=neg_prompt,
            strength=denoise_strength, steps=steps, cfg=cfg,
            snapshot_dir=os.path.join(debug_dir, "inpaint") if debug_dir else None,
            snapshot_every=snapshot_every
        )

    # ---- crops before/after
    if debug_dir:
        _save_crops(image, result, boxes, os.path.join(debug_dir, "crops"))

    return result
