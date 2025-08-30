# ===============================
# Logistic Regression 
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -------- LOAD DATA --------
df = df.copy()

# Ensure target column exists and is numeric
df["isRelevant"] = df["isRelevant"].astype(int)

# -------- FEATURES & TARGET --------
X = df[["categoryrelevanceScore", "similarity_catReview", "similarity_keywords"]]
y = df["isRelevant"]

# -------- TRAIN-TEST SPLIT (stratified) --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- TRAIN LOGISTIC REGRESSION --------
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# -------- PREDICT PROBABILITIES & DEFAULT THRESHOLD --------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)  # default threshold

# -------- EVALUATION --------
print("\nClassification Report (threshold = 0.5):")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

# -------- INSPECT COEFFICIENTS --------
for feat, coef in zip(X.columns, model.coef_[0]):
    print(f"Coefficient for {feat}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")
