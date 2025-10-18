from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Tuple
from collections import deque
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

Target = Literal["face", "hand", "person"]

@dataclass
class Detection:
    target: Target
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    score: float

class BaseDetector:
    def detect(self, img: Image.Image, targets: List[Target]) -> List[Detection]:
        raise NotImplementedError

# MediaPipe Hands (optional)
class MediaPipeHandsDetector(BaseDetector):
    def __init__(self, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:
            raise RuntimeError("mediapipe not installed") from e
        self.mp = mp
        self.conf = min_detection_confidence
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=self.conf
        )

    def detect(self, img: Image.Image, targets: List[Target]) -> List[Detection]:
        if "hand" not in targets:
            return []
        rgb = np.array(img.convert("RGB"))
        res = self.hands.process(rgb)
        dets: List[Detection] = []
        if res.multi_hand_landmarks:
            h, w = rgb.shape[:2]
            for lm in res.multi_hand_landmarks:
                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]
                x1, y1, x2, y2 = max(min(xs),0), max(min(ys),0), min(max(xs), w-1), min(max(ys), h-1)
                dets.append(Detection("hand", (x1, y1, x2, y2), 0.9))
        return dets

# YOLOv8 (optional)
class YOLOv8Detector(BaseDetector):
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.3):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("ultralytics not installed") from e
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, img: Image.Image, targets: List[Target]) -> List[Detection]:
        results = self.model.predict(img, conf=self.conf, verbose=False)
        dets: List[Detection] = []
        for r in results:
            boxes = r.boxes
            names = r.names
            for b in boxes:
                cls = int(b.cls.item())
                name = names.get(cls, "")
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                score = float(b.conf.item())
                if "person" in targets and name == "person":
                    dets.append(Detection("person", (x1, y1, x2, y2), score))
        return dets

# Lightweight fallback detector -------------------------------------------
class SimpleSkinDetector(BaseDetector):
    """Naive skin-tone blob detector.

    Provides a dependency-free heuristic so that the refinement pipeline can
    still run when optional detectors (MediaPipe / YOLO) are unavailable.
    The detector looks for skin-colored regions in YCbCr space, clusters them
    via a simple flood fill and returns coarse bounding boxes tagged as
    ``hand`` or ``face`` depending on the available targets.
    """

    def __init__(
        self,
        min_area: int = 600,
        score: float = 0.35,
        max_rel_area: float = 0.35,
        min_fill_ratio: float = 0.25,
    ):
        self.min_area = min_area
        self.score = score
        self.max_rel_area = max_rel_area
        self.min_fill_ratio = min_fill_ratio

    def _classify_target(
        self,
        bbox: Tuple[int, int, int, int],
        img_size: Tuple[int, int],
        targets: List[Target],
    ) -> Target:
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        img_w, img_h = img_size
        rel_area = area / float(img_w * img_h)
        aspect = w / h

        # Prefer ``face`` for roughly square, larger blobs towards the upper half
        if "face" in targets and rel_area > 0.01 and 0.7 <= aspect <= 1.6 and y2 < img_h * 0.85:
            return "face"
        # Otherwise fall back to ``hand`` if requested
        if "hand" in targets:
            return "hand"
        return targets[0]

    def detect(self, img: Image.Image, targets: List[Target]) -> List[Detection]:
        if not targets:
            return []
        if "hand" not in targets and "face" not in targets:
            return []

        # --- basic skin chroma mask (YCbCr)
        ycbcr = np.array(img.convert("YCbCr"), dtype=np.uint8)
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]
        mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)

        # --- additional HSV filter to suppress background colors (e.g. foliage)
        hsv = np.array(img.convert("HSV"), dtype=np.uint8)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        warm_hues = (h <= 50) | (h >= 200)
        hsv_mask = warm_hues & (s >= 40) & (s <= 200) & (v >= 60) & (v <= 245)
        mask &= hsv_mask

        # --- light morphological cleanup to remove isolated noise
        if mask.any():
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.filter(ImageFilter.MaxFilter(size=3))
            mask_img = mask_img.filter(ImageFilter.MinFilter(size=3))
            mask = np.array(mask_img, dtype=bool)

        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        dets: List[Detection] = []

        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue

                queue = deque([(x, y)])
                visited[y, x] = True
                min_x = max_x = x
                min_y = max_y = y
                area = 0

                while queue:
                    cx, cy = queue.popleft()
                    area += 1
                    if cx < min_x:
                        min_x = cx
                    if cx > max_x:
                        max_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cy > max_y:
                        max_y = cy

                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= w:
                            continue
                        for ny in (cy - 1, cy, cy + 1):
                            if ny < 0 or ny >= h:
                                continue
                            if visited[ny, nx] or not mask[ny, nx]:
                                continue
                            visited[ny, nx] = True
                            queue.append((nx, ny))

                if area < self.min_area:
                    continue

                bbox = (min_x, min_y, max_x + 1, max_y + 1)
                bbox_w = max(1, bbox[2] - bbox[0])
                bbox_h = max(1, bbox[3] - bbox[1])
                bbox_area = bbox_w * bbox_h
                rel_area = area / float(w * h)
                fill_ratio = area / float(bbox_area)

                if rel_area > self.max_rel_area:
                    continue
                if fill_ratio < self.min_fill_ratio:
                    continue

                target = self._classify_target(bbox, (w, h), targets)
                dets.append(Detection(target, bbox, self.score))

        return dets

# Mask utilities
def boxes_to_mask(size: Tuple[int,int], boxes: List[Tuple[int,int,int,int]],
                  expand: int = 8, blur: int = 8, min_area: int = 64) -> Image.Image:
    W, H = size
    m = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(m)
    for (x1,y1,x2,y2) in boxes:
        if (x2-x1)*(y2-y1) < min_area:
            continue
        xx1 = max(0, x1 - expand); yy1 = max(0, y1 - expand)
        xx2 = min(W-1, x2 + expand); yy2 = min(H-1, y2 + expand)
        draw.rectangle([xx1, yy1, xx2, yy2], fill=255)
    if blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=blur))
    return m

def filter_by_targets(dets: List[Detection], targets: List[Target], min_score: float = 0.25) -> List[Detection]:
    return [d for d in dets if d.target in targets and d.score >= min_score]
