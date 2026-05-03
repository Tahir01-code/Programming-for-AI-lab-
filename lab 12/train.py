"""
train.py — Medical Symptom Checker
====================================
Run this script ONCE before starting the Flask app.
It reads the Kaggle CSVs, trains the NLP + ML model,
and saves model.pkl to the current directory.

Usage:
    python train.py
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Step 1: Load dataset ─────────────────────────────────────
print("Loading dataset...")
df      = pd.read_csv(os.path.join(BASE_DIR, 'dataset.csv'))
sev_df  = pd.read_csv(os.path.join(BASE_DIR, 'Symptom-severity.csv'))

df['Disease'] = df['Disease'].str.strip()

print(f"  Rows: {len(df)} | Diseases: {df['Disease'].nunique()}")

# ── Step 2: Build symptom text per row ───────────────────────
print("Processing symptoms...")
symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]

def row_to_text(row):
    syms = [
        str(row[c]).strip().replace('_', ' ')
        for c in symptom_cols
        if pd.notna(row[c]) and str(row[c]).strip() not in ('', 'nan')
    ]
    return ' '.join(syms)

df['symptom_text'] = df.apply(row_to_text, axis=1)

# ── Step 3: Calculate average severity score per disease ─────
print("Calculating severity scores...")
sev_dict = dict(zip(sev_df['Symptom'].str.strip(), sev_df['weight']))

def row_severity_score(row):
    total = 0
    for c in symptom_cols:
        sym = str(row[c]).strip() if pd.notna(row[c]) else ''
        total += sev_dict.get(sym, 0)
    return total

df['severity_score']  = df.apply(row_severity_score, axis=1)
disease_severity      = df.groupby('Disease')['severity_score'].mean().to_dict()

# ── Step 4: TF-IDF Vectorization ─────────────────────────────
print("Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)
X     = tfidf.fit_transform(df['symptom_text'])
y     = df['Disease']

# ── Step 5: Train / Test Split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# ── Step 6: Train Random Forest ──────────────────────────────
print("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ── Step 7: Evaluate ─────────────────────────────────────────
print("\nEvaluating model...")
y_pred   = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"  Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Step 8: Save model ───────────────────────────────────────
model_path = os.path.join(BASE_DIR, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({
        'tfidf':            tfidf,
        'clf':              clf,
        'disease_severity': disease_severity,
    }, f)

print(f"Model saved → {model_path}")
print("\nDone! You can now run:  python app.py")