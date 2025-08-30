# Filtering the Noise: ML for Trustworthy Location Reviews

> **Group:** Toktok Loklok
> **Members:**
>
> - Cai Yu Shi — [caiyushi2019@gmail.com](mailto:caiyushi2019@gmail.com)
> - Pang Zi Ying, Petrine — [petrinepang09@gmail.com](mailto:petrinepang09@gmail.com)
> - Khoo Hui Qian — [khuiqian@gmail.com](mailto:khuiqian@gmail.com)
> - Chong Zhen Xi — [chongzhenxi@gmail.com](mailto:chongzhenxi@gmail.com)

---

## 1) Project Overview

**Goal.** Design and implement an ML system to evaluate **quality** and **relevancy** of Google location reviews so platforms and users can filter spam, promos, and off‑topic content.

**Data.** We combine:

- **Scraped Google reviews** (Apify) from \~50 locations across categories (restaurants, attractions, clinics, salons, …; multi‑country).
- **Kaggle Google Maps reviews** dataset.
- **Synthetic irrelevant reviews** (generated via LLM prompts) to stress‑test relevance detection.

All sources are normalized into a single dataframe with consistent columns and cleaned text.

**Features.** We engineer signals spanning:

- **Text quality:** word count, readability (Flesch), grammar/spelling issue rate.
- **Sentiment:** DistilBERT polarity mapped to \[0,1], plus **sentiment–rating gap**.
- **Links/ads:** URL regex + image ad/QR detection (OpenCV + OCR).
- **Images:** CLIP text–image similarity to review/category, plus image **clarity** (Laplacian variance).

**Models.** A **hybrid** pipeline for robustness + interpretability:

- **Policy & heuristics:** hard rules for links/ads, rapid repeats, duplicates → auto‑flag **low quality**.
- **Quality (Low/Medium/High):**

  - **Random Forest** on structured features.
  - **Dual‑head contrastive model** (MiniLM encoder + SupCon warm‑up) with a quality head (3‑class) and a relevance head (binary).
  - **Final label:** rule‑based overrides; otherwise **probability averaging** (RF ⊕ dual‑head).

- **Relevance (Relevant/Irrelevant):**

  - **Category–review semantic relevance:** sentence‑transformers embeddings + **LLM‑generated keywords** per category (Qwen 4B‑Instruct) → combined relevance score.
  - **Logistic Regression** on (category relevance, image relevance).
  - **Final label:** ensemble of dual‑head relevance head ⊕ logistic regression (avg prob, threshold 0.5).

**Outputs.** For each review we produce:

- `qualityLevel ∈ {0,1,2}` (Low, Medium, High) **and** its class probabilities.
- `isRelevant ∈ {0,1}` **and** its probability.
- Rich diagnostics: `readabilityScore`, `grammarSpellingScore`, `sentimentScore`, `sentimentRatingDiff`, `hasLink`, `combined_score_relevance`, `combined_score_quality`, `isAdvertisement`, and more.

---

## 2) Setup Instructions

### 2.1. Requirements

- **Python** ≥ 3.10
- OS: Linux / macOS / Windows
- (Optional) **GPU** (CUDA) or Apple Silicon (MPS) for faster CLIP/transformers

### 2.2. System packages

- **Tesseract OCR** (for coupon/offer OCR)

  - macOS (Homebrew): `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
  - Windows: Install the Tesseract MSI and ensure `tesseract.exe` is on PATH.

> If Tesseract is installed to a non‑standard path, set `TESSERACT_CMD` env var or point `pytesseract.pytesseract.tesseract_cmd` to it in code.

### 2.3. Python environment

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate        # Windows PowerShell

# install python deps
pip install -r requirements.txt
```

**Key Python packages** (already in `requirements.txt`):

- `pandas`, `numpy`, `matplotlib`, `seaborn`, `openpyxl`
- `textstat`, `language-tool-python`
- `transformers`, `torch`, `sentence-transformers`, `huggingface-hub`
- `pillow`, `requests`, `opencv-python-headless`, `pytesseract`
- `open-clip-torch`, `ftfy`, `regex`

### 2.4. Hugging Face access

- Create a **Read** token at _Hugging Face → Settings → Access Tokens_.
- Export it so scripts can pull models (e.g., Qwen, MiniLM):

```bash
export HF_TOKEN=***your_hf_token***   # macOS/Linux
# setx HF_TOKEN your_hf_token         # Windows (new shell needed)
```

### 2.5. Apple Silicon / CUDA notes (optional)

- **Apple Silicon (M‑series):** PyTorch MPS backend works out‑of‑the‑box in recent versions. You can enable it in code with `device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")`.
- **CUDA:** install a Torch build matching your CUDA toolchain (see pytorch.org). Scripts will auto‑use `cuda` if available.

---

## 3) How to Reproduce Results

The pipeline is modular. You can **(A)** use our released artifacts to reproduce predictions quickly, or **(B)** run the full process end‑to‑end.

### A) Quick reproduction (use pre‑trained artifacts)

1. Place the provided `artifacts/` directory (encoder, tokenizer, heads, config) at repo root.
2. Ensure you have a cleaned dataset (see §B.1–B.3) or use the sample provided by your team.
3. In Python, load the predictor and score reviews:

```python
from reviewclassifier import ReviewMTLPredictor
import pandas as pd

ARTIFACT_DIR = "artifacts"
clf = ReviewMTLPredictor.from_dir(ARTIFACT_DIR)

# df has a column `reviewText` (and other optional columns used for diagnostics)
df = pd.read_excel("scraped_dataset_cleaned_scored.xlsx")
probs = clf.predict(list(df["reviewText"]))  # returns dict with per‑head probabilities
# attach decisions
# quality: argmax over [Low, Medium, High]; relevance: probability > 0.5
```

4. For relevance ensembling, combine `dual_probs` with logistic regression probabilities (see §B.5).

### B) Full pipeline (end‑to‑end)

#### B.1. Gather raw data

- **Scraped reviews** (Apify export) → CSV/Parquet with fields like: `categoryName`, `name`, `publishedAtDate`, `title`, `text`, `textTranslated`, `stars`/`rating`, `reviewImageUrls/0..49`, `reviewerNumberOfReviews`, `temporarilyClosed`.
- **Kaggle dataset** (Restaurants).
- **Synthetic irrelevant** reviews (from your LLM prompt).

> Place files under `data/raw/` (or your chosen path).

#### B.2. Preprocess & unify schema

Apply the following (script/notebook of your choice):

- **Column selection/rename:**
  `title→locationName`, `categoryName→category`, `name→reviewerName`.
- **Image URLs:** concatenate `reviewImageUrls/0..49` → `imageURLs` (pipe‑separated or JSON list).
- **Review text:** `reviewText = textTranslated if available else text`.
- **Drop** rows with missing/empty `reviewText`.
- **Normalize ratings** to **\[0,1]**:
  `stars` or integer: `(rating-1)/4`;
  fraction `num/den`: `num/den`.
- **Reorder** columns for consistency:
  `locationName, category, reviewerName, publishedAtDate, rating, reviewText, imageURLs, reviewerNumberOfReviews, temporarilyClosed`.
- **Clean text:** remove emojis/special symbols.
- **Kaggle integration:** map to the same schema; set `category="Restaurant"`; normalize rating; add missing columns (empty defaults).
- **Concatenate** scraped + Kaggle + synthetic into one dataframe.
- **Standardize restaurant labels** to `"Restaurant"`.

**Output:** `scraped_dataset_cleaned.xlsx`

#### B.3. Feature engineering (text/sentiment/readability/links)

Augment the dataframe with columns:

- `reviewWordCount`
- `sentimentScore` and `sentimentRatingDiff` (DistilBERT sentiment vs normalized rating)
- `readabilityScore` (Flesch → \[0,1])
- `grammarSpellingScore` (LanguageTool → \[0,1])
- `hasLink` (URL regex)

**Output:** `scraped_dataset_cleaned_scored.xlsx`

#### B.4. Policy enforcement & image scoring

Run the image/policy scorer to add image relevance/quality and ad flags.

```bash
python main.py \
  --input scraped_dataset_cleaned_scored.xlsx \
  --output reviews_scored_with_images.xlsx
```

This computes, per review:

- `combined_score_relevance` = mean of (text–image similarity) and (image–place/category relevance) across images
- `combined_score_quality` = mean image clarity across images
- `isAdvertisement` based on QR/offer OCR

#### B.5. Category ↔ review semantic relevance

Generate category keywords (Qwen‑Instruct via HF) and compute semantic relevance with MiniLM embeddings.

- Set paths & `HF_TOKEN` inside `review_category_relevance.py`.
- Run:

```bash
python review_category_relevance.py
```

**Output:** adds `categoryRelevanceScore` (and stores partial keyword JSON for resuming).

#### B.6. Train models & produce final labels

**Quality (Low/Medium/High).**

1. **Rule‑based overrides** (low‑quality if: `hasLink==1` OR `isAdvertisement==1` OR spam duplicates/rapid repeats).
2. **Random Forest** with features: `readabilityScore, grammarSpellingScore, sentimentRatingDiff, reviewWordCount, imageQualityScore` (or `combined_score_quality`).
3. **Dual‑head contrastive model**

   - Stage‑1 SupCon warm‑up (balanced batches by quality)
   - Stage‑2 joint loss: `CE(Quality) + CE(Relevance) + λ·SupCon (λ≈0.1)`
   - Inference returns per‑class probabilities for **both heads**

4. **Final quality label**: if no rule override, **average** RF and dual‑head **probabilities** → argmax.

**Relevance (Relevant/Irrelevant).**

- Train **Logistic Regression** on `[categoryRelevanceScore, imageRelevanceScore]` (the latter can be derived from image text‑match/CLIP relevance).
- **Final relevance**: average `P(relevant)` from dual‑head relevance head and logistic regression; label = `1` if avg ≥ 0.5.

**Repro tips.** Fix seeds for determinism:

```python
import random, os, numpy as np, torch
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

#### B.7. Batch inference (using released artifacts)

```python
from reviewclassifier import ReviewMTLPredictor
import pandas as pd

clf = ReviewMTLPredictor.from_dir("artifacts")
df = pd.read_excel("reviews_scored_with_images.xlsx")

pred = clf.predict(df["reviewText"].tolist())
# pred["quality_probs"]: Nx3; pred["relevance_probs"]: Nx2
# attach to df and compute final labels per §B.6
```

---

## 4) Key Files

- `requirements.txt` — Python dependencies.
- `main.py` — Image scoring + ad/QR detection; writes combined image scores.
- `review_category_relevance.py` — Category keyword generation and semantic relevance scoring.
- `reviewclassifier.py` — Dual‑head contrastive model + `ReviewMTLPredictor` loader/inference.
- `artifacts/` — Pretrained encoder/tokenizer/heads and config for quick reproduction.
- `scraped_dataset_cleaned_scored.xlsx` — Example post‑feature‑engineering dataset (name may vary on your machine).

> Your repository may include additional notebooks or scripts for scraping, cleaning, model training, and evaluation. Align paths and filenames in the commands to your local layout.

---

## 5) Evaluation & Baselines (summary)

- **Dual‑head contrastive** model outperforms:

  - **Frozen embedding → Logistic Regression**
  - **TF‑IDF → Logistic Regression**

- Use class‑weighted losses + label smoothing for imbalance; report F1 (macro), precision/recall per class, and AUC for relevance.

---

## 6) Troubleshooting

- **Tesseract not found:** ensure it’s installed and on PATH; verify `tesseract --version`.
- **HF model download blocked:** set `HF_TOKEN` and ensure internet access; try `huggingface-cli login`.
- **Torch backend:** if GPU not used, force CPU by setting `CUDA_VISIBLE_DEVICES=""`; on Apple Silicon, verify `torch.backends.mps.is_available()`.
- **LanguageTool init time:** first run downloads language models; allow a short warm‑up on initial execution.

---

## 7) License & Acknowledgments

- Data sources: Google Maps reviews (scraped), Kaggle dataset (see dataset license), synthetic content generated via LLM.
- Models and libraries: Hugging Face Transformers, Sentence-Transformers, OpenCLIP, OpenCV, Tesseract OCR, LanguageTool, scikit‑learn, PyTorch.
