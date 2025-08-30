# ===============================
# Review Category Relevance  
# ===============================

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from google.colab import drive
import torch
import os
import json
import math

# -------- SETTINGS --------
TOP_N_WORDS = 50
WEIGHT_CAT = 0.5
WEIGHT_KEYEMB = 0.5
MODEL_NAME_EMB = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
HF_TOKEN = "<token>" #replace with Hugging Face token

# -------- MOUNT DRIVE --------
#drive.mount('/content/drive')
FILE_PATH = "/content/drive/MyDrive/toktoktest/scraped_dataset_cleaned_scored.xlsx"
OUTPUT_FILE = "/content/drive/MyDrive/toktoktest/dataset_with_category_relevance_score.xlsx"
PARTIAL_FILE = "/content/drive/MyDrive/toktoktest/category_keywords_partial.json"

# -------- LOAD DATA --------
print("Loading dataset...")
df = pd.read_excel(FILE_PATH)

# -------- CLEAN COLUMNS TO STRINGS --------
df["category"] = df["category"].fillna("").astype(str)
df["reviewText"] = df["reviewText"].fillna("").astype(str)

# -------- LOAD SENTENCE TRANSFORMER --------
print(f"Loading sentence transformer: {MODEL_NAME_EMB} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME_EMB, device=device)

# -------- SETUP HF INFERENCE API --------
print(f"Connecting to Hugging Face Inference API: {HF_MODEL} ...")
client = InferenceClient(HF_MODEL, token=HF_TOKEN)

# -------- LOAD OR INITIALIZE PARTIAL RESULTS --------
if os.path.exists(PARTIAL_FILE):
    with open(PARTIAL_FILE, "r") as f:
        category_to_words = json.load(f)
else:
    category_to_words = {}

# -------- FUNCTION TO GENERATE KEYWORDS FOR BATCH --------
def generate_keywords_batch(categories, top_n=TOP_N_WORDS):
    """
    Generate keywords for multiple categories in a single API call.
    Returns a dict: {category: [keyword1, keyword2, ...]}
    """
    prompt = (
        f"For each of the following categories, generate {top_n} keywords that "
        "will appear in reviews, as a comma-separated list, no explanation. "
        "Respond in JSON format with category as key and list of keywords as value.\n"
    )
    prompt += json.dumps([str(c) for c in categories])
    
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    
    text = response.choices[0].message["content"]
    
    try:
        result = json.loads(text)  # parse JSON directly
    except json.JSONDecodeError:
        print("Error parsing JSON from response:")
        print(text)
        result = {}

    # Ensure each list contains top_n keywords and only whole words
    for k, v in result.items():
        if isinstance(v, list):
            result[k] = [w.strip().lower() for w in v][:top_n]
        else:
            # fallback: split string by commas
            result[k] = [w.strip().lower() for w in str(v).split(",") if w.strip()][:top_n]
        print(f"    {k} → {result[k]}")
    
    return result

# -------- GENERATE KEYWORDS IN BATCHES WITH RETRY --------
unique_categories = df["category"].unique()
print(unique_categories)
# Remove empty strings
unique_categories = [c for c in unique_categories if c.strip()]
remaining_categories = [c for c in unique_categories if c not in category_to_words]
print(f"{len(remaining_categories)} categories remaining to generate keywords...")

if remaining_categories:
    batch_size = max(1, math.ceil(len(remaining_categories) / 5))
    
    for i in range(0, len(remaining_categories), batch_size):
        batch_cats = remaining_categories[i:i+batch_size]
        batch_results = generate_keywords_batch(batch_cats, TOP_N_WORDS)
        
        # Update partial results
        category_to_words.update(batch_results)
        
        # Retry any missing categories individually
        missing = [c for c in batch_cats if c not in batch_results]
        if missing:
            print("Retrying missing categories individually:", missing)
            for cat in missing:
                retry_result = generate_keywords_batch([cat], TOP_N_WORDS)
                category_to_words.update(retry_result)
        
        # Save partial progress after each batch
        with open(PARTIAL_FILE, "w") as f:
            json.dump(category_to_words, f)
else:
    print("No categories left to process.")

# -------- MAP KEYWORDS TO DF --------
df["relevantWords"] = df["category"].map(category_to_words)

# -------- SIMILARITY SCORE (Category ↔ Review) --------
print("Computing embeddings similarity (category vs review)...")
cat_embs = model.encode(df["category"].tolist(), convert_to_tensor=True)
rev_embs = model.encode(df["reviewText"].tolist(), convert_to_tensor=True)
df["similarity_catReview"] = [util.cos_sim(c, r).item() for c, r in zip(cat_embs, rev_embs)]

# -------- KEYWORD EMBEDDING SIMILARITY --------
print("Computing similarity to keyword embeddings...")
def review_keyword_similarity(review, keywords, model):
    if not isinstance(keywords, list) or not keywords:  # Check if keywords is a list and not empty
        return 0.0
    rev_emb = model.encode(review, convert_to_tensor=True)
    kw_embs = model.encode(keywords, convert_to_tensor=True)
    return util.cos_sim(rev_emb, kw_embs).max().item()

df["similarity_keywords"] = [
    review_keyword_similarity(r, kws, model) for r, kws in zip(df["reviewText"], df["relevantWords"])
]

# -------- FINAL RELEVANCE SCORE --------
df["relevanceScore"] = (
    WEIGHT_CAT * df["similarity_catReview"] +
    WEIGHT_KEYEMB * df["similarity_keywords"]
)

# -------- SAVE RESULTS --------
df.to_excel(OUTPUT_FILE, index=False)
print(f"\nResults saved to {OUTPUT_FILE}")
print(f"Partial keyword progress saved to {PARTIAL_FILE}")
