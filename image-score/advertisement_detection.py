import os
import re
import time
from typing import List

import cv2
import numpy as np
import pytesseract
from PIL import Image


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pil_to_cv(img: Image.Image) -> np.ndarray:
    import numpy as _np
    import cv2 as _cv2
    return _cv2.cvtColor(_np.array(img), _cv2.COLOR_RGB2BGR)


def detect_qr_codes(img: Image.Image) -> bool:
    """Detect presence of QR codes using OpenCV QRCodeDetector.
    More permissive: treat detected points as QR presence even if decode fails.
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
                return True
            if points is not None and len(points) > 0:
                return True
            if hasattr(detector, "detectAndDecodeMulti"):
                try:
                    retval, decoded_info, mpoints, _ = detector.detectAndDecodeMulti(v)  # type: ignore
                    if retval:
                        if decoded_info and any(bool(s) for s in decoded_info):
                            return True
                        if mpoints is not None and len(mpoints) > 0:
                            return True
                except Exception:
                    pass
        except Exception:
            continue
    return False


COUPON_KEYWORDS = [
    # English core keywords
    r"\bcoupon\b", r"\bvoucher\b", r"promo\s*code", r"\bdiscount\b",
    r"\bdeal\b", r"special\s*offer", r"\boffer\b", r"cash\s*back|cashback",
    r"\bsave\s*(up\s*to\s*)?\$?\d+%?\b", r"\bfree\b(\s*gift|\s*drink|\s*dessert)?",
    r"\b\d{1,2}\s*%\s*off\b", r"\b\$\s?\d+\s*off\b", r"\brm\s?\d+\s*off\b",
    r"flash\s*deal", r"\bsale\b", r"scan\s*to\s*(redeem|order|pay)",
    r"use\s*code", r"\bqr\s*code\b", r"limited\s*time|limited\s*period",
    r"buy\s*one\s*get\s*one|\bbogo\b", r"redeem\s*(at|now)",
    r"member\s*price|member\s*exclusive", r"\bpromo\b",
    r"valid\s*until|expires\s*on",
    # Chinese / Japanese common variants
    r"特价", r"优惠", r"折扣", r"买一送一", r"优惠券", r"促销", r"クーポン",
]


def ocr_coupon_hits(img: Image.Image, lang: str = "eng") -> bool:
    """Run Tesseract OCR and search for coupon-like keywords with multiple variants."""
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

    psm_modes = [11, 6, 3]
    langs = lang or "eng"

    for v in expanded[:10]:
        for psm in psm_modes:
            cfg = f"--oem 3 --psm {psm}"
            try:
                text = pytesseract.image_to_string(v, lang=langs, config=cfg)
            except Exception:
                continue
            tl = text.lower()
            if len(tl.strip()) < 5:
                continue
            for kw in COUPON_KEYWORDS:
                if re.search(kw, tl):
                    log(f"Coupon keyword detected via OCR (psm={psm})")
                    return True

    return False

