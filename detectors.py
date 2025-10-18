from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Target = Literal["face", "hand", "person"]


@dataclass
class DetectorStatus:
    """Small helper for reporting detector initialization status."""

    name: str
    available: bool
    error: Optional[str] = None


@dataclass
class Detection:
    target: Target
    bbox: Tuple[int, int, int, int]  # x1,y1,x2,y2
    score: float


class BaseDetector:
    def detect(self, img: Image.Image, targets: List[Target]) -> List[Detection]:
        raise NotImplementedError


class MediaPipeHandsDetector(BaseDetector):
    """Detector backed by MediaPipe Hands."""

    def __init__(self, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("mediapipe not installed") from e

        self.mp = mp
        self.conf = min_detection_confidence
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=self.conf,
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
                x1, y1, x2, y2 = (
                    max(min(xs), 0),
                    max(min(ys), 0),
                    min(max(xs), w - 1),
                    min(max(ys), h - 1),
                )
                dets.append(Detection("hand", (x1, y1, x2, y2), 0.9))
        return dets


def build_available_detectors(
    *,
    mediapipe_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[List[BaseDetector], List[DetectorStatus]]:
    """Return initialized MediaPipe-based detectors together with availability logs."""

    detectors: List[BaseDetector] = []
    statuses: List[DetectorStatus] = []

    try:
        detector = MediaPipeHandsDetector(**(mediapipe_kwargs or {}))
    except Exception as exc:  # pragma: no cover - optional dependency
        statuses.append(
            DetectorStatus(name="MediaPipeHandsDetector", available=False, error=str(exc))
        )
    else:
        detectors.append(detector)
        statuses.append(DetectorStatus(name="MediaPipeHandsDetector", available=True))

    if verbose:
        for status in statuses:
            if status.available:
                print(f"[INFO] {status.name} ready")
            else:
                print(f"[INFO] {status.name} not available: {status.error}")

    return detectors, statuses


# Mask utilities

def boxes_to_mask(
    size: Tuple[int, int],
    boxes: List[Tuple[int, int, int, int]],
    expand: int = 8,
    blur: int = 8,
    min_area: int = 64,
) -> Image.Image:
    W, H = size
    m = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(m)
    for (x1, y1, x2, y2) in boxes:
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        xx1 = max(0, x1 - expand)
        yy1 = max(0, y1 - expand)
        xx2 = min(W - 1, x2 + expand)
        yy2 = min(H - 1, y2 + expand)
        draw.rectangle([xx1, yy1, xx2, yy2], fill=255)
    if blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=blur))
    return m


def filter_by_targets(
    dets: List[Detection],
    targets: List[Target],
    min_score: float = 0.25,
) -> List[Detection]:
    return [d for d in dets if d.target in targets and d.score >= min_score]
