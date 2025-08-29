# Data cleaning of scraped dataset and Kaggle dataset
import pandas as pd
import numpy as np
import re

# ----------------------------
# 0. Define file paths
# ----------------------------
input_scraped_path = "C:/Users/Asus/Desktop/Y4S1/techjam/dataset_scraped_all_fields.csv"  
input_kaggle_path = "C:/Users/Asus/Desktop/Y4S1/techjam/reviews.csv"                     
output_path = "C:/Users/Asus/Desktop/Y4S1/techjam/cleaned_reviews_dataset.csv"           

# ----------------------------
# 1. Load original scraped dataset
# ----------------------------
uncleaned_df = pd.read_csv(input_scraped_path, encoding='latin1')

# ----------------------------
# 2. Feature selection
# ----------------------------
columns_to_keep = ["categoryName",
                   "name",
                   "publishedAtDate",
                   "rating"] + [f"reviewImageUrls/{i}" for i in range(50)] + [
                   "reviewsCount",
                   "stars",
                   "temporarilyClosed",
                   "text",
                   "textTranslated",
                   "title",
                   "reviewerNumberOfReviews"]

cleaned_df = uncleaned_df[columns_to_keep].copy()

# ----------------------------
# 3. Concatenate review image URLs
# ----------------------------
image_columns = [f"reviewImageUrls/{i}" for i in range(50)]
def combine_images(row):
    return ",".join([str(x) for x in row if pd.notna(x) and str(x).strip() not in ["", "None", "nan"]])

cleaned_df["imageURLs"] = cleaned_df[image_columns].apply(combine_images, axis=1)
cleaned_df.drop(columns=image_columns, inplace=True)

# ----------------------------
# 4. Create reviewText
# ----------------------------
cleaned_df["reviewText"] = np.where(
    cleaned_df["textTranslated"].notna() & (cleaned_df["textTranslated"].str.strip() != ""),
    cleaned_df["textTranslated"],
    cleaned_df["text"]
)

# Drop empty reviewText rows
cleaned_df = cleaned_df[cleaned_df["reviewText"].notna() & (cleaned_df["reviewText"].str.strip() != "")]

# ----------------------------
# 5. Create rating column
# ----------------------------
def extract_rating(row):
    if pd.notna(row["stars"]) and str(row["stars"]).strip() != "":
        return round(float(row["stars"]))
    elif pd.notna(row["rating"]) and "/" in str(row["rating"]):
        return round(float(str(row["rating"]).split("/")[0]))
    elif pd.notna(row["rating"]):
        return round(float(row["rating"]))
    else:
        return np.nan

cleaned_df["rating"] = cleaned_df.apply(extract_rating, axis=1)

# ----------------------------
# 6. Rename columns
# ----------------------------
cleaned_df.rename(columns={
    "title": "locationName",
    "categoryName": "category",
    "name": "reviewerName"
}, inplace=True)

# ----------------------------
# 7. Reorder columns
# ----------------------------
new_column_order = [
    "locationName",
    "category",
    "reviewerName",
    "publishedAtDate",
    "rating",
    "reviewText",
    "imageURLs",
    "reviewerNumberOfReviews",
    "temporarilyClosed"
]

cleaned_df = cleaned_df[new_column_order]

# ----------------------------
# 8. Remove emoticons / non-text characters
# ----------------------------
def remove_emojis(text):
    if pd.isna(text):
        return text
    return re.sub(r"[^\u0000-\uFFFFa-zA-Z0-9\s.,!?;:()\-']", "", text)

cleaned_df["reviewText"] = cleaned_df["reviewText"].apply(remove_emojis)

# ----------------------------
# 9. Load Kaggle dataset and map columns
# ----------------------------
kaggle_df = pd.read_csv(input_kaggle_path, encoding='latin1')

kaggle_df.rename(columns={
    "business_name": "locationName",
    "author_name": "reviewerName",
    "text": "reviewText",
    "photo": "imageURLs",
    "rating_category": "category"
}, inplace=True)

# Set all categories to "Restaurant"
kaggle_df["category"] = "Restaurant"

# Add missing columns
for col_name in new_column_order:
    if col_name not in kaggle_df.columns:
        if col_name == "temporarilyClosed":
            kaggle_df[col_name] = ""
        elif col_name == "reviewerNumberOfReviews":
            kaggle_df[col_name] = ""
        elif col_name == "rating":
            kaggle_df[col_name] = np.nan
        else:
            kaggle_df[col_name] = pd.NaT

# Ensure same column order
kaggle_df = kaggle_df[new_column_order]

# Remove emoticons from Kaggle reviewText
kaggle_df["reviewText"] = kaggle_df["reviewText"].apply(remove_emojis)

# ----------------------------
# 10. Combine datasets
# ----------------------------
combined_df = pd.concat([cleaned_df, kaggle_df], ignore_index=True)

# Normalize restaurant categories in combined data
combined_df["category"] = combined_df["category"].apply(
    lambda x: "Restaurant" if isinstance(x,str) and "restaurant" in x.lower() else x
)

# ----------------------------
# 11. Export to CSV
# ----------------------------
combined_df.to_csv(output_path, index=False)
print(f"Combined dataset exported to {output_path}")
