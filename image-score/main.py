import argparse
from typing import List, Optional
import pandas as pd
from tqdm import tqdm
from image_review_scoring import (
    log,
    build_clip,
    safe_split_urls,
    download_image,
    score_single_image,
    combine_scores,
)


def process_dataframe(df: pd.DataFrame,
                      image_col: str = "reviewImage",
                      location_col: str = "locationName",
                      category_col: str = "category",
                      review_text_col: str = "reviewText",
                      max_images_per_review: int = 5,
                      device: Optional[str] = None,
                      ocr_lang: str = "eng",
                      text_gamma: float = 1.5,
                      rel_gamma: float = 1.2,
                      clarity_scale: float = 400.0,
                      combine_contrast: float = 1.0) -> pd.DataFrame:
    """
    Process a DataFrame and append the two requested columns:
      - combined_image_score (float)
      - isAdvertistment (int flag)
    """
    # Prepare CLIP (once)
    log("Loading CLIP model (ViT-B-32, openai weights)...")
    ctx = build_clip(device=device)

    combined_scores: List[float] = []
    ad_flags: List[int] = []

    # Log input schema and attempt to auto-detect image column if missing
    cols = list(df.columns)
    log(f"Input columns: {cols}")
    if image_col not in df.columns:
        candidates = [c for c in cols if any(k in c.lower() for k in ["image", "images", "photo", "picture", "img"])]
        if candidates:
            log(f"Image column '{image_col}' not found. Using '{candidates[0]}' instead.")
            image_col = candidates[0]
        else:
            log(f"Image column '{image_col}' not found and no obvious alternative detected.")
    if image_col in df.columns:
        non_empty = int(df[image_col].astype(str).str.contains(r"https?://", case=False, na=False).sum())
        log(f"Detected {non_empty} row(s) with URL(s) in '{image_col}'.")

    n = len(df)
    for idx in tqdm(range(n), desc="Scoring reviews"):
        row = df.iloc[idx]
        raw_val = row.get(image_col, "")
        urls = safe_split_urls(raw_val)
        if max_images_per_review > 0:
            urls = urls[:max_images_per_review]

        per_image_scores: List[float] = []
        is_ad_any = False
        log(f"Review {idx+1}/{n}: processing {len(urls)} image(s)")

        for u in urls:
            img = download_image(u)
            if img is None:
                short_u = (u[:100] + '...') if len(u) > 100 else u
                log(f"Failed to load image from URL: {short_u}")
                continue
            # Log successful image fetch for this URL
            short_u = (u[:100] + '...') if len(u) > 100 else u
            try:
                w, h = img.size
                log(f"Loaded image from URL: {short_u} (size={w}x{h})")
            except Exception:
                log(f"Loaded image from URL: {short_u}")
            try:
                sc, is_ad = score_single_image(
                    ctx, img,
                    location_name=str(row.get(location_col, "")),
                    category=str(row.get(category_col, "")),
                    review_text=str(row.get(review_text_col, "")),
                    run_ocr_lang=ocr_lang,
                    text_gamma=text_gamma,
                    rel_gamma=rel_gamma,
                    clarity_scale=clarity_scale,
                    combine_contrast=combine_contrast
                )
                per_image_scores.append(sc)
                is_ad_any = is_ad_any or is_ad
            except Exception as e:
                # Skip image on error
                log(f"Error scoring image from URL: {short_u} ({type(e).__name__}: {str(e)[:120]})")
                continue

        if len(urls) == 0:
            rv = str(raw_val)
            rv_short = (rv[:120] + '...') if len(rv) > 120 else rv
            log(f"No image URLs parsed for review {idx+1}. Raw field value: {rv_short!r}")

        combined = combine_scores(per_image_scores)
        combined_scores.append(combined)
        ad_flags.append(int(is_ad_any))
        log(f"Review {idx+1}/{n} combined score: {combined if combined==combined else float('nan'):.3f}, isAdvertistment={int(is_ad_any)}")

    df = df.copy()
    df["combined_image_score"] = combined_scores
    # Keep user's requested column name spellings
    df["isAdvertistment"] = ad_flags
    return df

def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str):
    if path.lower().endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Compute image-based scores for review datasets.")
    parser.add_argument("--input", required=True, help="Path to input CSV/XLSX with review data.")
    parser.add_argument("--output", required=True, help="Path to output CSV/XLSX with new columns.")
    parser.add_argument("--image_col", default="reviewImage", help="Column with comma-separated image URLs.")
    parser.add_argument("--location_col", default="locationName", help="Column for location name.")
    parser.add_argument("--category_col", default="category", help="Column for category (e.g., cuisine).")
    parser.add_argument("--review_text_col", default="reviewText", help="Column for review text used for image-text similarity.")
    parser.add_argument("--max_images_per_review", type=int, default=5, help="Limit images processed per review.")
    parser.add_argument("--device", default=None, help="Force device: 'cuda' or 'cpu'. Default: auto.")
    parser.add_argument("--ocr_lang", default="eng", help="Tesseract language code, e.g., 'eng', 'chi_sim'.")
    parser.add_argument("--text_gamma", type=float, default=1.5, help="Gamma for text-image similarity to increase contrast.")
    parser.add_argument("--rel_gamma", type=float, default=1.2, help="Gamma for relevance score to increase contrast.")
    parser.add_argument("--clarity_scale", type=float, default=400.0, help="Scale for clarity mapping; lower spreads scores more.")
    parser.add_argument("--combine_contrast", type=float, default=1.0, help="Sigmoid slope for combined score; >1 broadens spread.")
    parser.add_argument("--post_norm", default="none", choices=["none", "minmax", "zscore", "rank"],
                        help="Dataset-level normalization applied to combined_image_score.")

    args = parser.parse_args()

    df = read_table(args.input)
    out_df = process_dataframe(
        df,
        image_col=args.image_col,
        location_col=args.location_col,
        category_col=args.category_col,
        review_text_col=args.review_text_col,
        max_images_per_review=args.max_images_per_review,
        device=args.device,
        ocr_lang=args.ocr_lang,
        text_gamma=args.text_gamma,
        rel_gamma=args.rel_gamma,
        clarity_scale=args.clarity_scale,
        combine_contrast=args.combine_contrast,
    )
    # Optional post-normalization for broader spread
    scores = out_df["combined_image_score"].to_numpy(dtype=float)
    mask = ~np.isnan(scores)
    method = (args.post_norm or "none").lower()
    # Allow env override IMG_SCORE_POST_NORM
    import os as _os
    env_method = (_os.environ.get("IMG_SCORE_POST_NORM", "") or "").lower()
    if env_method in {"minmax", "zscore", "rank", "none"} and env_method != "":
        method = env_method
    if mask.any() and method in {"minmax", "zscore", "rank"}:
        vals = scores[mask]
        if method == "minmax":
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmax > vmin:
                scores[mask] = (vals - vmin) / (vmax - vmin)
                log(f"Post-norm minmax applied (min={vmin:.4f}, max={vmax:.4f}).")
        elif method == "zscore":
            mu, sigma = float(np.mean(vals)), float(np.std(vals) + 1e-8)
            z = (vals - mu) / sigma
            scores[mask] = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
            log(f"Post-norm zscore->CDF applied (mean={mu:.4f}, std={sigma:.4f}).")
        elif method == "rank":
            order = np.argsort(vals)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(vals), dtype=float)
            denom = max(1.0, float(len(vals) - 1))
            scores[mask] = ranks / denom
            log("Post-norm rank applied (uniform 0..1 spread).")
        out_df["combined_image_score"] = scores
    write_table(out_df, args.output)
    log(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()
