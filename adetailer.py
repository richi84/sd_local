# adetailer.py
import math
import os, json
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from detectors import BaseDetector, Detection, filter_by_targets, boxes_to_mask

Targets = List[Literal["face","hand","person"]]


@dataclass
class ADetailerPreparation:
    selected: List[Detection]
    boxes_original: List[Tuple[int, int, int, int]]
    boxes_expanded: List[Tuple[int, int, int, int]]
    det_logs: List[dict]
    per_detector: List[Tuple[str, List[Detection]]]
    edges_image: Optional[Image.Image]


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


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


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = float(np.sum(region * kernel))
    return output


def _non_max_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    H, W = magnitude.shape
    result = np.zeros((H, W), dtype=np.float32)
    angle = angle * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            q = 0.0
            r = 0.0
            ang = angle[i, j]

            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= ang < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= ang < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            else:  # 112.5 <= ang < 157.5
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                result[i, j] = magnitude[i, j]
    return result


def _double_threshold(img: np.ndarray, low: float, high: float) -> Tuple[np.ndarray, float, float]:
    res = np.zeros_like(img, dtype=np.uint8)
    strong = 255
    weak = 75

    strong_mask = img >= high
    weak_mask = (img >= low) & (img < high)

    res[strong_mask] = strong
    res[weak_mask] = weak
    return res, weak, strong


def _edge_tracking_by_hysteresis(img: np.ndarray, weak: float, strong: float) -> np.ndarray:
    H, W = img.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if img[i, j] == weak:
                neighborhood = img[i - 1:i + 2, j - 1:j + 2]
                if np.any(neighborhood == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    img[img != strong] = 0
    return img


def _canny_edges(image: Image.Image, low_threshold: float = 60.0, high_threshold: float = 120.0) -> Image.Image:
    gray = image.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1.0))
    arr = np.array(blurred, dtype=np.float32)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    Gx = _convolve2d(arr, Kx)
    Gy = _convolve2d(arr, Ky)

    magnitude = np.hypot(Gx, Gy)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0
    angle = np.arctan2(Gy, Gx)

    suppressed = _non_max_suppression(magnitude, angle)
    thresh, weak, strong = _double_threshold(suppressed, low_threshold, high_threshold)
    edges = _edge_tracking_by_hysteresis(thresh, weak, strong)
    return Image.fromarray(edges.astype(np.uint8), mode="L")


def _build_edges_image(image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Optional[Image.Image]:
    if not boxes:
        return None

    edges_canvas = Image.new("L", image.size, 0)
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        edges_crop = _canny_edges(crop)
        edges_canvas.paste(edges_crop, (x1, y1))

    return edges_canvas


def prepare_adetailer(
    image: Image.Image,
    detectors: List[BaseDetector],
    targets: Targets,
    *,
    expand_fraction: float = 0.10,
) -> ADetailerPreparation:
    all_dets: List[Detection] = []
    det_logs: List[dict] = []
    per_detector: List[Tuple[str, List[Detection]]] = []

    for det in detectors:
        name = det.__class__.__name__
        try:
            detections = det.detect(image, targets)
            all_dets.extend(detections)
            det_logs.append({"detector": name, "count": len(detections)})
            per_detector.append((name, detections))
        except Exception as e:
            msg = f"Detector {name} failed: {e}"
            print("[ADetailer]", msg)
            det_logs.append({"detector": name, "error": str(e)})

    selected = filter_by_targets(all_dets, targets, min_score=0.2)
    boxes_original = [d.bbox for d in selected]
    boxes_expanded = [
        _expand_bbox(bbox, image.size, expand_fraction)
        for bbox in boxes_original
    ]
    edges_image = _build_edges_image(image, boxes_expanded)

    return ADetailerPreparation(
        selected=selected,
        boxes_original=boxes_original,
        boxes_expanded=boxes_expanded,
        det_logs=det_logs,
        per_detector=per_detector,
        edges_image=edges_image,
    )


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
    preparation: Optional[ADetailerPreparation] = None,
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

    if preparation is None:
        preparation = prepare_adetailer(image=image, detectors=detectors, targets=targets)

    sel = preparation.selected
    boxes_original = preparation.boxes_original
    boxes = preparation.boxes_expanded
    det_logs = preparation.det_logs
    per_detector = preparation.per_detector

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
                        "expanded_bbox": boxes[idx] if idx < len(boxes) else d.bbox,
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

        for det_name, det_dets in per_detector:
            det_overlay_path = os.path.join(
                debug_dir, f"input_{_safe_name(det_name)}.png"
            )
            det_filtered = filter_by_targets(det_dets, targets, min_score=0.2)
            det_boxes = [d.bbox for d in det_filtered]
            if det_boxes:
                det_expanded = [
                    _expand_bbox(bbox, image.size, 0.10)
                    for bbox in det_boxes
                ]
                _save_overlay(
                    image,
                    det_boxes,
                    det_overlay_path,
                    extra_boxes=det_expanded,
                )
            else:
                image.save(det_overlay_path)

    if not sel:
        print("[ADetailer] No detections. Skipping refine.")
        return image

    # ---- mask
    mask = boxes_to_mask(size=image.size, boxes=boxes, expand=expand_px, blur=blur_px)
    if debug_dir:
        mask.save(os.path.join(debug_dir, "mask.png"))

    # ---- edges (optional)
    edges_for_guidance: Optional[Image.Image] = edges_image
    if use_edges and edges_for_guidance is None:
        edges_for_guidance = preparation.edges_image
    if debug_dir and edges_for_guidance is not None:
        edges_for_guidance.save(os.path.join(debug_dir, "edges_hand.png"))

    # ---- refine
    if use_edges and edges_for_guidance is not None:
        result = sd.edge_guided_refine(
            pil_img=image, mask=mask, edges=edges_for_guidance,
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
