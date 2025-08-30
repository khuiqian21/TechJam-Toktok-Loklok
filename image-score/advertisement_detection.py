import os
import re
import time
from typing import List

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pil_to_cv(img: Image.Image) -> np.ndarray:
    import numpy as _np
    import cv2 as _cv2
    return _cv2.cvtColor(_np.array(img), _cv2.COLOR_RGB2BGR)


def _quad_plausible(points: np.ndarray, img_shape: tuple,
                    min_area_ratio: float = 0.005,
                    max_aspect_deviation: float = 0.35) -> bool:
    """Validate that detected quad is large and roughly square.
    - min_area_ratio: minimum area relative to image area to consider valid.
    - max_aspect_deviation: allowed deviation from square (0.0 means perfect square).
    """
    try:
        if points is None:
            return False
        # points can be shape (4,2) or (1,4,2)
        pts = np.array(points, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 4:
            return False
        area = cv2.contourArea(pts)
        H, W = img_shape[:2]
        img_area = float(H * W)
        if img_area <= 0:
            return False
        if area / img_area < min_area_ratio:
            return False
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), angle = rect
        w, h = float(max(w, 1e-6)), float(max(h, 1e-6))
        ar = max(w, h) / min(w, h)
        # ar close to 1.0 for square; allow small deviation
        if ar - 1.0 > max_aspect_deviation:
            return False
        return True
    except Exception:
        return False


def detect_qr_codes(img: Image.Image, strict: bool = True) -> bool:
    """Detect QR codes using OpenCV QRCodeDetector.
    - strict=True (default): only flag when decoding succeeds.
    - strict=False: also accept plausible quads when decode fails (may add false positives).
    """
    cv_img = pil_to_cv(img)
    detector = cv2.QRCodeDetector()

    variants: List[np.ndarray] = [cv_img]
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        variants.append(gray)
        h, w = gray.shape[:2]
        if min(h, w) < 512:
            up = cv2.resize(cv_img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)
            variants.append(up)
    except Exception:
        pass

    for v in variants:
        try:
            data, points, _ = detector.detectAndDecode(v)
            if data:
                log("QR detected via decode")
                return True
            if not strict and _quad_plausible(points, v.shape):
                log("QR plausible quad detected (non-strict)")
                return True
            if hasattr(detector, "detectAndDecodeMulti"):
                try:
                    retval, decoded_info, mpoints, _ = detector.detectAndDecodeMulti(v)  # type: ignore
                    if retval and decoded_info and any(bool(s) for s in decoded_info):
                        log("QR detected via multi-decode")
                        return True
                    if not strict and _quad_plausible(mpoints, v.shape):
                        log("QR plausible multi-quad detected (non-strict)")
                        return True
                except Exception:
                    pass
        except Exception:
            continue
    return False


STRONG_COUPON_PATTERNS = [
    r"\b\d{1,2}\s*%\s*off\b",
    r"\b(?:rm|\$)\s?\d+\s*off\b",
    r"promo\s*code",
    r"use\s*code",
    r"buy\s*one\s*get\s*one|\bbogo\b",
    r"scan\s*to\s*(redeem|order|pay)",
]

WEAK_COUPON_KEYWORDS = [
    r"\bcoupon\b", r"\bvoucher\b", r"\bdiscount\b", r"\bdeal\b",
    r"special\s*offer", r"\boffer\b", r"cash\s*back|cashback",
    r"\bsave\b", r"\bfree\b", r"\bsale\b", r"limited\s*time|limited\s*period",
    r"redeem\s*(at|now)", r"member\s*price|member\s*exclusive", r"\bpromo\b",
    r"valid\s*until|expires\s*on",
    # Chinese / Japanese common variants
    r"特价", r"优惠", r"折扣", r"买一送一", r"优惠券", r"促销", r"クーポン",
]


def ocr_coupon_hits(img: Image.Image, lang: str = "eng", min_conf: int = 60,
                    weak_threshold: int = 2, fast: bool = False) -> bool:
    """Run Tesseract OCR and search for coupon-like keywords with multiple variants.
    Uses word-level confidences; requires either a strong pattern or multiple weak keywords.
    """
    tess_cmd = os.environ.get("TESSERACT_CMD", None)
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    cv_img = pil_to_cv(img)
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    variants: List[np.ndarray] = []
    try:
        variants.append(gray)
        if not fast:
            _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(th_otsu)
            th_gauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 31, 11)
            variants.append(th_gauss)
            th_mean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 31, 11)
            variants.append(th_mean)
            variants.extend([255 - v for v in [th_otsu, th_gauss, th_mean] if v is not None])
            blur = cv2.medianBlur(gray, 3)
            variants.append(blur)
            h, w = gray.shape[:2]
            if min(h, w) < 700:
                up = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_LINEAR)
                variants.append(up)
    except Exception:
        variants.append(gray)

    expanded: List[np.ndarray] = []
    for v in variants:
        expanded.append(v)
        try:
            expanded.append(cv2.rotate(v, cv2.ROTATE_90_CLOCKWISE))
            expanded.append(cv2.rotate(v, cv2.ROTATE_90_COUNTERCLOCKWISE))
        except Exception:
            pass

    psm_modes = [6] if fast else [11, 6, 3]
    langs = lang or "eng"

    limit = 3 if fast else 10
    for v in expanded[:limit]:
        for psm in psm_modes:
            cfg = f"--oem 3 --psm {psm}"
            try:
                data = pytesseract.image_to_data(v, lang=langs, config=cfg, output_type=Output.DICT)
            except Exception:
                continue
            words = data.get("text", []) or []
            confs = data.get("conf", []) or []
            tokens = []
            for w, c in zip(words, confs):
                try:
                    c = float(c)
                except Exception:
                    c = -1.0
                if c >= float(min_conf):
                    w = (w or "").strip().lower()
                    if w:
                        tokens.append(w)
            if len(tokens) < 2:
                continue
            tl = " ".join(tokens)
            # Strong patterns first
            for pat in STRONG_COUPON_PATTERNS:
                if re.search(pat, tl):
                    log(f"Coupon strong pattern via OCR (psm={psm})")
                    return True
            # Count weak matches
            weak_hits = 0
            for kw in WEAK_COUPON_KEYWORDS:
                if re.search(kw, tl):
                    weak_hits += 1
            if weak_hits >= weak_threshold:
                log(f"Coupon weak keywords via OCR (psm={psm}, hits={weak_hits})")
                return True

    return False
