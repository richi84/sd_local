from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Tuple, Any, Iterable
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

# OpenPose / ControlNet auxiliary detector ---------------------------------
class OpenposeDetector(BaseDetector):
    """Detector that builds bounding boxes from ControlNet's OpenPose keypoints."""

    def __init__(
        self,
        include_body: bool = True,
        include_hands: bool = True,
        include_face: bool = True,
        detect_resolution: int = 512,
        image_resolution: int | None = None,
        confidence_threshold: float = 0.1,
    ) -> None:
        try:
            from controlnet_aux import OpenposeDetector as AuxOpenposeDetector  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("controlnet_aux not installed") from e

        self.include_body = include_body
        self.include_hands = include_hands
        self.include_face = include_face
        self.detect_resolution = detect_resolution
        self.image_resolution = image_resolution
        self.confidence_threshold = confidence_threshold

        if hasattr(AuxOpenposeDetector, "from_pretrained"):
            pretrained_id = "lllyasviel/Annotators"
            try:  # pragma: no cover - best effort optional download
                self.detector = AuxOpenposeDetector.from_pretrained(pretrained_id)
            except Exception:
                self.detector = AuxOpenposeDetector()
        else:
            self.detector = AuxOpenposeDetector()

    def _run_openpose(self, np_img: np.ndarray, hand_and_face: bool) -> dict[str, Any]:
        kwargs = {
            "detect_resolution": self.detect_resolution,
            "hand_and_face": hand_and_face,
        }
        if self.image_resolution is not None:
            kwargs["image_resolution"] = self.image_resolution

        result = None
        try:
            result = self.detector(np_img, return_dict=True, **kwargs)
        except TypeError:
            result = self.detector(np_img, **kwargs)

        if isinstance(result, dict):
            return result

        data: dict[str, Any] = {}
        if isinstance(result, tuple):
            if len(result) == 4:
                _, bodies, hands, faces = result
            elif len(result) == 3:
                bodies, hands, faces = result
            else:
                bodies = result[0] if len(result) > 0 else []
                hands = result[1] if len(result) > 1 else []
                faces = result[2] if len(result) > 2 else []
            data["bodies"] = bodies
            data["hands"] = hands
            data["faces"] = faces
            return data

        # Fallback to attributes exposed on the detector instance
        data["bodies"] = getattr(self.detector, "pose_result", [])
        data["hands"] = getattr(self.detector, "hands_result", [])
        data["faces"] = getattr(self.detector, "face_result", [])
        return data

    def _iter_keypoints(self, entries: Any) -> Iterable[np.ndarray]:
        if entries is None:
            return []
        if isinstance(entries, dict):
            return self._iter_keypoints([entries])
        if isinstance(entries, (list, tuple)):
            arrays = []
            for item in entries:
                if item is None:
                    continue
                if isinstance(item, dict):
                    if "keypoints" in item:
                        kp = np.asarray(item["keypoints"], dtype=float)
                    elif "pose_keypoints_2d" in item:
                        kp = np.asarray(item["pose_keypoints_2d"], dtype=float)
                    else:
                        kp = np.asarray(item, dtype=float)
                else:
                    kp = np.asarray(item, dtype=float)
                if kp.ndim == 1:
                    if kp.size % 3 == 0:
                        kp = kp.reshape(-1, 3)
                    elif kp.size % 2 == 0:
                        xy = kp.reshape(-1, 2)
                        conf = np.ones((xy.shape[0], 1), dtype=float)
                        kp = np.concatenate([xy, conf], axis=1)
                arrays.append(kp)
            return arrays
        kp = np.asarray(entries, dtype=float)
        if kp.ndim == 1 and kp.size % 3 == 0:
            kp = kp.reshape(-1, 3)
        return [kp]

    def _bbox_from_keypoints(
        self, points: np.ndarray, img_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int] | None:
        if points.size == 0:
            return None
        conf_mask = points[:, 2] > self.confidence_threshold
        if not np.any(conf_mask):
            conf_mask = points[:, 2] > 0
            if not np.any(conf_mask):
                return None
        xs = points[conf_mask, 0]
        ys = points[conf_mask, 1]
        if xs.size == 0 or ys.size == 0:
            return None
        img_w, img_h = img_size
        x1 = int(np.clip(xs.min(), 0, img_w - 1))
        y1 = int(np.clip(ys.min(), 0, img_h - 1))
        x2 = int(np.clip(xs.max(), 0, img_w - 1))
        y2 = int(np.clip(ys.max(), 0, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def detect(self, img: Image.Image, targets: List[Target]) -> List[Detection]:
        need_body = self.include_body and "person" in targets
        need_hand = self.include_hands and "hand" in targets
        need_face = self.include_face and "face" in targets
        if not (need_body or need_hand or need_face):
            return []

        np_img = np.array(img.convert("RGB"))
        outputs = self._run_openpose(np_img, hand_and_face=(need_hand or need_face))
        img_size = img.size

        dets: List[Detection] = []

        if need_body:
            for kp in self._iter_keypoints(outputs.get("bodies")):
                bbox = self._bbox_from_keypoints(kp, img_size)
                if bbox is None:
                    continue
                score = float(np.clip(np.mean(kp[:, 2]), 0.0, 1.0)) if kp.size else 0.5
                dets.append(Detection("person", bbox, score or 0.5))

        if need_hand:
            for kp in self._iter_keypoints(outputs.get("hands")):
                bbox = self._bbox_from_keypoints(kp, img_size)
                if bbox is None:
                    continue
                score = float(np.clip(np.mean(kp[:, 2]), 0.0, 1.0)) if kp.size else 0.6
                dets.append(Detection("hand", bbox, score or 0.6))

        if need_face:
            for kp in self._iter_keypoints(outputs.get("faces")):
                bbox = self._bbox_from_keypoints(kp, img_size)
                if bbox is None:
                    continue
                score = float(np.clip(np.mean(kp[:, 2]), 0.0, 1.0)) if kp.size else 0.6
                dets.append(Detection("face", bbox, score or 0.6))

        return dets

# Lightweight fallback detector -------------------------------------------
class SimpleSkinDetector(BaseDetector):
    """Naive skin-tone blob detector with extra heuristics.

    Provides a dependency-free heuristic so that the refinement pipeline can
    still run when optional detectors (MediaPipe / YOLO) are unavailable.  The
    detector looks for skin-colored regions in YCbCr space, clusters them via a
    simple flood fill and returns coarse bounding boxes tagged as ``hand`` or
    ``face`` depending on the available targets.  Additional scoring filters
    keep only the most plausible regions to avoid an excessive number of false
    positives when running with complex scenes.
    """

    def __init__(
        self,
        min_area: int = 600,
        score: float = 0.35,
        min_rel_area: float = 0.002,
        ideal_rel_area: float = 0.015,
        max_rel_area: float = 0.35,
        min_fill_ratio: float = 0.25,
        min_score: float = 0.4,
        max_per_target: int = 3,
        edge_penalty: float = 0.6,
        hand_rel_area: Tuple[float, float, float] | None = None,
        face_rel_area: Tuple[float, float, float] | None = None,
    ):
        self.min_area = min_area
        self.base_score = score
        self.min_rel_area = min_rel_area
        self.ideal_rel_area = max(ideal_rel_area, min_rel_area)
        self.max_rel_area = max_rel_area
        self.min_fill_ratio = min_fill_ratio
        self.min_score = min_score
        self.max_per_target = max_per_target
        self.edge_penalty = edge_penalty
        # target specific relative area tuning (min, ideal, max)
        self.hand_rel_area = hand_rel_area or (0.0015, 0.01, 0.08)
        self.face_rel_area = face_rel_area or (0.004, 0.02, 0.18)

    def _aspect_bounds(self, target: Target) -> Tuple[float, float]:
        if target == "face":
            return 0.7, 1.6
        return 0.35, 2.6

    def _score_detection(
        self,
        target: Target,
        bbox: Tuple[int, int, int, int],
        area: int,
        img_size: Tuple[int, int],
        fill_ratio: float,
    ) -> float:
        img_w, img_h = img_size
        rel_area = area / float(img_w * img_h)

        if target == "face":
            min_rel, ideal_rel, max_rel = self.face_rel_area
        elif target == "hand":
            min_rel, ideal_rel, max_rel = self.hand_rel_area
        else:
            min_rel, ideal_rel, max_rel = (
                self.min_rel_area,
                self.ideal_rel_area,
                self.max_rel_area,
            )

        min_rel = max(self.min_rel_area, min_rel)
        ideal_rel = max(min_rel, ideal_rel)
        max_rel = min(self.max_rel_area, max_rel)

        if rel_area < min_rel or rel_area > max_rel:
            return 0.0

        rel_score = 0.0
        if ideal_rel > min_rel:
            rel_score = min(
                1.0,
                max(0.0, (rel_area - min_rel) / (ideal_rel - min_rel)),
            )

        fill_score = min(1.0, max(0.0, (fill_ratio - self.min_fill_ratio) / max(1e-6, 1.0 - self.min_fill_ratio)))

        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect = w / h
        lo, hi = self._aspect_bounds(target)
        if aspect < lo:
            aspect_score = max(0.0, aspect / lo)
        elif aspect > hi:
            aspect_score = max(0.0, hi / aspect)
        else:
            aspect_score = 1.0

        touches_border = x1 <= 2 or y1 <= 2 or x2 >= img_w - 3 or y2 >= img_h - 3

        score = 0.45 * fill_score + 0.35 * rel_score + 0.20 * aspect_score
        if touches_border:
            score *= self.edge_penalty

        score = max(0.0, min(score, 0.99))

        if self.base_score:
            score = max(score, min(0.99, self.base_score))

        return float(score)

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

        # --- additional RGB-domain heuristics (helps suppress saturated reds)
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        rgb_mask = (
            (r > 80)
            & (g > 35)
            & (b > 15)
            & ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15)
            & (np.abs(r - g) > 7)
            & (r > g)
            & (r > b)
        )
        sum_rgb = np.maximum(r + g + b, 1)
        norm_r = r / sum_rgb
        norm_g = g / sum_rgb
        rgb_mask &= (
            (norm_r >= 0.32)
            & (norm_r <= 0.52)
            & (norm_g >= 0.24)
            & (norm_g <= 0.40)
        )
        mask &= rgb_mask

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

                target = self._classify_target(bbox, (w, h), targets)
                score = self._score_detection(target, bbox, area, (w, h), fill_ratio)
                if score < self.min_score:
                    continue
                dets.append(Detection(target, bbox, score))

        if self.max_per_target and dets:
            grouped: dict[Target, List[Detection]] = {}
            for det in dets:
                grouped.setdefault(det.target, []).append(det)
            limited: List[Detection] = []
            for det_list in grouped.values():
                det_list.sort(key=lambda d: d.score, reverse=True)
                limited.extend(det_list[: self.max_per_target])
            dets = limited

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
