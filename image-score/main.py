import argparse
from typing import List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from image_review_scoring import (
    log,
    build_clip,
    safe_split_urls,
    download_image,
    score_single_image,
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
    Process a DataFrame and append the requested columns:
      - combined_score_relevance: avg of per-image (review_text_similarity and relevance)
      - combined_score_quality: avg of per-image clarity scores
      - isAdvertisement: OR over per-image ad flags (QR or coupon keyword)
    """
    # Prepare CLIP (once)
    log("Loading CLIP model (ViT-B-32, openai weights)...")
    ctx = build_clip(device=device)

    rel_scores_out: List[float] = []
    qual_scores_out: List[float] = []
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

        text_sims: List[float] = []
        rel_sims: List[float] = []
        clarities: List[float] = []
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
                txt_sc, rel_sc, clarity_sc, is_ad = score_single_image(
                    ctx, img,
                    location_name=str(row.get(location_col, "")),
                    category=str(row.get(category_col, "")),
                    review_text=str(row.get(review_text_col, "")),
                    run_ocr_lang=ocr_lang,
                    text_gamma=text_gamma,
                    rel_gamma=rel_gamma,
                    clarity_scale=clarity_scale,
                )
                text_sims.append(float(txt_sc))
                rel_sims.append(float(rel_sc))
                clarities.append(float(clarity_sc))
                is_ad_any = is_ad_any or is_ad
            except Exception as e:
                # Skip image on error
                log(f"Error scoring image from URL: {short_u} ({type(e).__name__}: {str(e)[:120]})")
                continue

        if len(urls) == 0:
            rv = str(raw_val)
            rv_short = (rv[:120] + '...') if len(rv) > 120 else rv
            log(f"No image URLs parsed for review {idx+1}. Raw field value: {rv_short!r}")

        if len(text_sims) > 0:
            mean_txt = float(np.mean(text_sims))
            mean_rel = float(np.mean(rel_sims)) if len(rel_sims) > 0 else float('nan')
            rel_scores_out.append(float(np.mean([mean_txt, mean_rel])))
            qual_scores_out.append(float(np.mean(clarities)) if len(clarities) > 0 else float('nan'))
        else:
            rel_scores_out.append(float('nan'))
            qual_scores_out.append(float('nan'))
        ad_flags.append(int(is_ad_any))
        log(f"Review {idx+1}/{n} combined_score_relevance: {rel_scores_out[-1] if rel_scores_out[-1]==rel_scores_out[-1] else float('nan'):.3f}, combined_score_quality: {qual_scores_out[-1] if qual_scores_out[-1]==qual_scores_out[-1] else float('nan'):.3f}, isAdvertisement={int(is_ad_any)}")

    df = df.copy()
    df["combined_score_relevance"] = rel_scores_out
    df["combined_score_quality"] = qual_scores_out
    df["isAdvertisement"] = ad_flags
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
    )

    write_table(out_df, args.output)
    log(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()
