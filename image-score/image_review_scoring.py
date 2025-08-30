import io
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps

# OpenCV (use headless build for servers)
import cv2

# OCR
from advertisement_detection import detect_qr_codes, ocr_coupon_hits

# CLIP
import torch
import open_clip
from tqdm import tqdm

# -------------------------------
# Utilities
# -------------------------------

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_split_urls(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    # Try JSON-style list first, e.g. ["url1", "url2"]
    try:
        if (s.startswith('[') and s.endswith(']')):
            import json
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    # Split on common delimiters: comma, pipe, semicolon, newline, tabs
    tokens = [t.strip() for t in re.split(r'[\,\|;\n\t]+', s) if t.strip()]
    # Deduplicate preserving order
    seen = set()
    uniq: List[str] = []
    for x in tokens:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

# to handle google drive links

def normalize_image_url(url: str) -> str:
    if not isinstance(url, str):
        return url
    u = url.strip()
    # Google Drive patterns
    m = re.search(r"https?://drive\.google\.com/file/d/([^/]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    m = re.search(r"https?://drive\.google\.com/open\?id=([^&]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    m = re.search(r"https?://drive\.google\.com/uc\?id=([^&]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return u


# download image and return PIL Image or nan on failure
def download_image(url: str, timeout: float = 10.0, max_bytes: int = 15_000_000) -> Optional[Image.Image]:
    try:
        # Normalize sharable links (e.g., Google Drive) to direct download
        norm_url = normalize_image_url(url)
        if norm_url != url:
            log("Normalized sharing URL to direct download format")

        if re.match(r'^https?://', norm_url, flags=re.I):
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ReviewImageScorer/1.0)",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            }
            with requests.get(norm_url, headers=headers, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                # Limit bytes to avoid huge downloads
                content = io.BytesIO()
                bytes_read = 0
                for chunk in r.iter_content(chunk_size=16384):
                    if chunk:
                        content.write(chunk)
                        bytes_read += len(chunk)
                        if bytes_read > max_bytes:
                            raise ValueError("Image too large")
                content.seek(0)
                img = Image.open(content)
        else:
            # Local file path
            img = Image.open(url)
        img = img.convert("RGB")
        return img
    except Exception as e:
        return None


def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def normalize_blur_score(var_lap: float, scale: float = 400.0) -> float:
    """
    Map Laplacian variance to [0,1] clarity score (higher = sharper).
    Heuristic: saturates near ~1 for very sharp images; near 0 for very blurry.
    """
    # Avoid negatives
    var_lap = max(0.0, float(var_lap))
    # Exponential saturating mapping
    scale = max(1e-6, float(scale))
    clarity = 1.0 - math.exp(-var_lap / scale)
    return float(np.clip(clarity, 0.0, 1.0))


def apply_gamma(x: float, gamma: float) -> float:
    """Gamma-adjust a [0,1] score. gamma>1 increases contrast around mid values."""
    g = max(1e-6, float(gamma))
    x = float(np.clip(x, 0.0, 1.0))
    return float(np.power(x, g))


def apply_contrast_sigmoid(x: float, k: float) -> float:
    """Apply a centered sigmoid with slope k to [0,1] score to broaden spread."""
    k = float(k)
    x = float(np.clip(x, 0.0, 1.0))
    if abs(k) < 1e-6 or k == 1.0:
        return x
    y = 1.0 / (1.0 + math.exp(-k * (x - 0.5)))
    return float(np.clip(y, 0.0, 1.0))


def compute_blur_variance(img: Image.Image) -> float:
    """Compute Laplacian variance as a measure of blur (higher = sharper)."""
    cv_img = pil_to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()



# -------------------------------
# CLIP helpers (open-clip)
# -------------------------------

@dataclass
class CLIPContext:
    model: torch.nn.Module
    preprocess: torch.nn.Module
    tokenizer: callable
    device: str
    static_text_embeds: Dict[str, torch.Tensor]  # cache for static prompts
    dynamic_text_cache: Dict[str, torch.Tensor]  # cache for ad-hoc prompts


def build_clip(device: Optional[str] = None) -> CLIPContext:
    """
    Build an open-clip model (ViT-B-32) with OpenAI weights.
    Selects a safe device if requested backend isn't available.
    """
    def _mps_available() -> bool:
        try:
            return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and torch.backends.mps.is_built())
        except Exception:
            return False

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif _mps_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        req = str(device).lower()
        if req == "cuda" and not torch.cuda.is_available():
            log("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        elif req in ("mps", "metal") and not _mps_available():
            log("MPS requested but not available. Falling back to CPU.")
            device = "cpu"
        elif req not in ("cuda", "cpu", "mps"):
            # Unknown device string; choose safely
            device = "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)
    model.eval()

    # Precompute static class prompts
    static_prompts = {
        # Generally “good” review images for any location
        "storefront": [
            "the exterior of a shop or venue storefront",
            "a building exterior with the venue entrance visible",
            "a shopfront sign on a street"
        ],
        "interior": [
            "the interior of a venue with rooms, tables, or counters",
            "an indoor scene of a shop or service counter",
            "a lobby, waiting area, or seating area"
        ],
        "signage": [
            "a sign with the venue name",
            "a signboard or nameplate",
            "a storefront sign"
        ],
        "product": [
            "a product or item on a shelf or counter",
            "merchandise displayed in a store",
            "an equipment or tool used in a service"
        ],
        "landmark": [
            "a recognizable landmark or facade of a place",
            "the outside of a building of interest",
        ],
        # Keep “food” so restaurants still score well without specializing
        "food": [
            "a close-up photo of a dish",
            "a plate of food on a table",
            "a bowl of noodles",
            "a burger and fries",
            "sushi on a plate"
        ],
        # Negative/low-value
        "random": [
            "a selfie photo",
            "a random screenshot",
            "a blank image",
            "a plain QR code poster",
            "an unrelated picture"
        ]
    }   

    static_text_embeds: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k, prompts in static_prompts.items():
            toks = tokenizer(prompts).to(device)
            feats = model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            static_text_embeds[k] = feats  # [n_prompts, d]

    return CLIPContext(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        static_text_embeds=static_text_embeds,
        dynamic_text_cache={},
    )


@torch.no_grad()
def image_features(ctx: CLIPContext, img: Image.Image) -> torch.Tensor:
    x = ctx.preprocess(img).unsqueeze(0).to(ctx.device)
    feats = ctx.model.encode_image(x)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [1, d]


@torch.no_grad()
def image_features_batch(ctx: CLIPContext, imgs: List[Image.Image]) -> torch.Tensor:
    if not imgs:
        return torch.empty((0, ctx.model.visual.output_dim), device=ctx.device)
    batch = torch.stack([ctx.preprocess(im) for im in imgs], dim=0).to(ctx.device)
    feats = ctx.model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [n, d]


@torch.no_grad()
def text_features(ctx: CLIPContext, prompts: List[str]) -> torch.Tensor:
    key = "||".join(prompts).lower()
    if key in ctx.dynamic_text_cache:
        return ctx.dynamic_text_cache[key]
    toks = ctx.tokenizer(prompts).to(ctx.device)
    feats = ctx.model.encode_text(toks)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    ctx.dynamic_text_cache[key] = feats
    return feats


def cosine_sim(img_feats: torch.Tensor, txt_feats: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix [n_img x n_txt]. img_feats and txt_feats must be L2-normalized."""
    return img_feats @ txt_feats.T


def precompute_review_text_feature(ctx: CLIPContext, review_text: str) -> Optional[torch.Tensor]:
    if not isinstance(review_text, str) or not review_text.strip():
        return None
    return text_features(ctx, [review_text])  # [1, d]


def precompute_relevance_text_features(ctx: CLIPContext, location_name: str, category: str) -> torch.Tensor:
    prompts = []
    if isinstance(location_name, str) and location_name.strip():
        ln = location_name.strip()
        prompts.extend([
            f"a photo of {ln} storefront sign",
            f"the exterior of {ln} restaurant or shop",
            f"the interior of {ln}",
        ])
    if isinstance(category, str) and category.strip():
        cat = category.strip()
        prompts.extend([
            f"{cat} food",
            f"a {cat} restaurant dish",
            f"a {cat} restaurant interior",
        ])
    if not prompts:
        prompts = ["a restaurant", "a cafe", "a plate of food"]
    return text_features(ctx, prompts)


def group_max_similarity(img_feats: torch.Tensor, group_txt: torch.Tensor) -> float:
    """Return max cosine similarity (as float) between the image and a group of prompts."""
    sims = cosine_sim(img_feats, group_txt)  # [1, n_group]
    return float(sims.max().item())


def relevance_score(ctx: CLIPContext, img: Image.Image, location_name: str, category: str) -> float:
    """
    Compute relevance of the image to the given locationName and category using CLIP.
    Returns a score in [0,1].
    """
    prompts = []
    if isinstance(location_name, str) and location_name.strip():
        ln = location_name.strip()
        prompts.extend([
            f"a photo of {ln} storefront sign",
            f"the exterior of {ln} restaurant or shop",
            f"the interior of {ln}",
        ])
    if isinstance(category, str) and category.strip():
        cat = category.strip()
        prompts.extend([
            f"{cat} food",
            f"a {cat} restaurant dish",
            f"a {cat} restaurant interior",
        ])
    # Fallback prompts if nothing provided
    if not prompts:
        prompts = ["a restaurant", "a cafe", "a plate of food"]

    img_feats = image_features(ctx, img)
    txt_feats = text_features(ctx, prompts)
    sims = cosine_sim(img_feats, txt_feats)  # [1, n_prompts]
    sim_max = float(sims.max().item())  # cosine in [-1,1]
    # Map cosine [-1,1] -> [0,1]
    return float((sim_max + 1.0) / 2.0)


def review_text_similarity(ctx: CLIPContext, img: Image.Image, review_text: str) -> float:
    """
    Compute similarity between the review text and the image using CLIP.
    Returns a score in [0,1].
    """
    if not review_text.strip():
        return 0.0  # No meaningful text to compare

    # Encode the review text
    text_feats = text_features(ctx, [review_text])

    # Encode the image
    img_feats = image_features(ctx, img)

    # Compute cosine similarity
    sim = cosine_sim(img_feats, text_feats)  # [1, 1]
    sim_score = float(sim.item())  # Extract scalar value

    # Map cosine similarity from [-1, 1] to [0, 1]
    return (sim_score + 1.0) / 2.0


# -------------------------------
# Scoring logic
# -------------------------------

def score_single_image(ctx: CLIPContext, img: Image.Image,
                       location_name: str, category: str,
                       review_text: str,
                       run_ocr_lang: str = "eng",
                       text_gamma: float = 1.5,
                       rel_gamma: float = 1.2,
                       clarity_scale: float = 400.0,
                       qr_strict: bool = True,
                       fast_ocr: bool = False,
                       ocr_min_conf: int = 60,
                       ocr_weak_threshold: int = 2,
                       img_feat: Optional[torch.Tensor] = None,
                       review_text_feat: Optional[torch.Tensor] = None,
                       rel_txt_feats: Optional[torch.Tensor] = None) -> Tuple[float, float, float, bool]:
    # 1) Text-image similarity using review content
    if img_feat is None:
        img_feat = image_features(ctx, img)  # [1, d]
    if review_text_feat is None:
        txt_sim_sc = review_text_similarity(ctx, img, review_text)
    else:
        sim = cosine_sim(img_feat, review_text_feat)
        txt_sim_sc = float((sim.item() + 1.0) / 2.0)
    txt_sim_sc = apply_gamma(txt_sim_sc, text_gamma)

    # 2) Relevance score (to location/category)
    if rel_txt_feats is None:
        rel_sc = relevance_score(ctx, img, location_name, category)
    else:
        sims = cosine_sim(img_feat, rel_txt_feats)  # [1, n]
        sim_max = float(sims.max().item())
        rel_sc = float((sim_max + 1.0) / 2.0)
    rel_sc = apply_gamma(rel_sc, rel_gamma)

    # 3) Clarity/blur score
    var_lap = compute_blur_variance(img)
    clarity_sc = normalize_blur_score(var_lap, scale=clarity_scale)

    # 4) Advertisement detection (QR or OCR coupon keywords)
    has_qr = detect_qr_codes(img, strict=qr_strict)
    from advertisement_detection import ocr_coupon_hits as _ocr
    has_coupon = _ocr(img, lang=run_ocr_lang, min_conf=ocr_min_conf, weak_threshold=ocr_weak_threshold, fast=fast_ocr)
    is_ad = bool(has_qr or has_coupon)

    # Return component scores for aggregation at review level
    return float(txt_sim_sc), float(rel_sc), float(clarity_sc), is_ad


def combine_scores(img_scores: List[float]) -> float:
    """
    Average per-image combined scores into a single review-level score.
    Returns combined_image_score (float).
    """
    if not img_scores:
        return float('nan')
    return float(np.mean(img_scores))

