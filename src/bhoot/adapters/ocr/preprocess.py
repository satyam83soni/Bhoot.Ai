"""Scan clean-up before OCR (OpenCV, installed with ``bhoot[ocr]``).

``auto``: grayscale, straighten a slightly rotated page, even out contrast.
``aggressive``: also binarise (faxes, faint photocopies). ``off``: the image as rendered.

Straightening uses a projection-profile search (the angle at which text rows give the
sharpest row histogram), which behaves the same on every OpenCV version. The transform from
prepared pixels back to original pixels is returned, so OCR boxes land on the real page.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from bhoot.ports.ocr import IDENTITY_3X3, PreparedImage

MAX_SKEW_DEG = 5.0
SKEW_STEP_DEG = 0.25
MIN_SKEW_DEG = 0.3  # smaller tilts don't hurt OCR; leave the image untouched
ESTIMATE_SIDE_PX = 1000
PARAMS_VERSION = "1"


def _cv() -> tuple[Any, Any]:
    import cv2
    import numpy as np

    return cv2, np


class OpenCvScanPreprocessor:
    def params_hash(self) -> str:
        key = f"{PARAMS_VERSION}:{MAX_SKEW_DEG}:{SKEW_STEP_DEG}:{MIN_SKEW_DEG}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def prepare(self, png: bytes, *, mode: Literal["auto", "off", "aggressive"]) -> PreparedImage:
        if mode == "off":
            return PreparedImage(png)
        cv2, np = _cv()
        gray = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("page image could not be decoded")
        steps = ["grayscale"]
        transform: tuple[float, ...] = IDENTITY_3X3
        angle = self.estimate_skew(gray)
        if abs(angle) >= MIN_SKEW_DEG:
            h, w = gray.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(
                gray,
                matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            inverse = cv2.invertAffineTransform(matrix)  # prepared -> original
            transform = (*map(float, inverse[0]), *map(float, inverse[1]), 0.0, 0.0, 1.0)
            steps.append(f"deskew {angle:+.2f}deg")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        steps.append("contrast")
        if mode == "aggressive":
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )
            steps.append("binarize")
        ok, encoded = cv2.imencode(".png", gray)
        if not ok:
            raise ValueError("prepared page image could not be encoded")
        return PreparedImage(encoded.tobytes(), rotation=0, transform=transform, steps=tuple(steps))

    @staticmethod
    def estimate_skew(gray: Any) -> float:
        """The rotation (degrees, counter-clockwise) that makes text rows horizontal."""
        cv2, np = _cv()
        h, w = gray.shape
        scale = min(1.0, ESTIMATE_SIDE_PX / max(h, w))
        small = cv2.resize(
            gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
        _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if cv2.countNonZero(ink) < 50:
            return 0.0
        sh, sw = ink.shape
        center = (sw / 2, sh / 2)
        best_angle, best_score = 0.0, -1.0
        steps = int(MAX_SKEW_DEG / SKEW_STEP_DEG)
        for k in range(-steps, steps + 1):
            angle = k * SKEW_STEP_DEG
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(ink, matrix, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
            rows = rotated.sum(axis=1, dtype=np.float64)
            score = float(np.sum(np.diff(rows) ** 2))
            if score > best_score:
                best_angle, best_score = angle, score
        return best_angle
