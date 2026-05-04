"""
train.py — Medical Symptom Checker (NLP Pipeline)
===================================================
Run this ONCE before starting the Flask app.

Steps:
  1. Load & clean dataset
  2. NLP preprocessing (lemmatization, stopword removal)
  3. TF-IDF feature extraction (unigrams + bigrams)
  4. Train multiple classifiers & pick the best
  5. Save model.pkl

Usage:
    pip install -r requirements.txt
    python train.py
"""

import os, re, pickle, warnings
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Built-in stopwords (no NLTK download needed) ────────────
STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then',
    'once','here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own',
    'same','so','than','too','very','s','t','can','will','just','don',
    'should','now','feel','feeling','also','since','get','got','please',
    'past','last','days','day','ago','back','little','much','well','really',
}

# ── Simple rule-based lemmatizer ─────────────────────────────
LEMMA_RULES = [
    (r'ings$',''), (r'ing$',''), (r'tion$',''), (r'tions$',''),
    (r'nesses$',''), (r'ness$',''), (r'ies$','y'), (r'edly$',''),
    (r'ed$',''), (r'es$',''), (r's$',''),
]

def simple_lemmatize(word):
    if len(word) <= 4:
        return word
    for pattern, repl in LEMMA_RULES:
        result = re.sub(pattern, repl, word)
        if result != word and len(result) >= 3:
            return result
    return word

# ── NLP Preprocessing ────────────────────────────────────────
def nlp_preprocess(text: str) -> str:
    """
    Full NLP pipeline:
      - Lowercase
      - Remove special characters / digits
      - Replace underscores with spaces
      - Tokenize
      - Remove stopwords
      - Lemmatize
    """
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z\s_]', ' ', text)
    text = text.replace('_', ' ')
    tokens = text.split()
    tokens = [simple_lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

# ── Step 1: Load Dataset ─────────────────────────────────────
print("\n[1/7] Loading datasets...")
df     = pd.read_csv(os.path.join(BASE_DIR, 'dataset.csv'))
sev_df = pd.read_csv(os.path.join(BASE_DIR, 'Symptom-severity.csv'))

df['Disease'] = df['Disease'].str.strip()
print(f"  Rows: {len(df)} | Unique Diseases: {df['Disease'].nunique()}")

# ── Step 2: Build symptom text per row ───────────────────────
print("\n[2/7] Building symptom text with NLP preprocessing...")
symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]

def row_to_text(row):
    syms = []
    for c in symptom_cols:
        val = str(row[c]).strip() if pd.notna(row[c]) else ''
        if val and val.lower() != 'nan':
            syms.append(val)
    return nlp_preprocess(' '.join(syms))

df['symptom_text'] = df.apply(row_to_text, axis=1)
print(f"  Sample: '{df['symptom_text'].iloc[0]}'")

# ── Step 3: Severity scoring ─────────────────────────────────
print("\n[3/7] Computing severity scores per disease...")
sev_dict = dict(zip(sev_df['Symptom'].str.strip(), sev_df['weight']))

def row_severity(row):
    total = 0
    for c in symptom_cols:
        sym = str(row[c]).strip() if pd.notna(row[c]) else ''
        total += sev_dict.get(sym, 0)
    return total

df['severity_score'] = df.apply(row_severity, axis=1)
disease_severity = df.groupby('Disease')['severity_score'].mean().to_dict()
print(f"  Severity computed for {len(disease_severity)} diseases.")

# ── Step 4: TF-IDF Vectorization ─────────────────────────────
print("\n[4/7] TF-IDF Vectorization (unigrams + bigrams)...")
tfidf = TfidfVectorizer(
    ngram_range  = (1, 2),
    max_features = 2000,
    sublinear_tf = True,
    min_df       = 2,
)

X = tfidf.fit_transform(df['symptom_text'])
y = df['Disease']
print(f"  Feature matrix: {X.shape}")

# ── Step 5: Train/Test Split ─────────────────────────────────
print("\n[5/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Step 6: Train & Compare Models ───────────────────────────
print("\n[6/7] Training & evaluating classifiers...")

candidates = {
    'Random Forest':       RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=5.0, random_state=42),
    'LinearSVC (calib.)':  CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42)),
}

best_name, best_clf, best_acc = None, None, 0.0

for name, model in candidates.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  {name:25s} -> Accuracy: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc  = acc
        best_clf  = model
        best_name = name

print(f"\n  Best model: {best_name} ({best_acc*100:.2f}%)")

print("\nClassification Report (Best Model):")
print(classification_report(y_test, best_clf.predict(X_test)))

# ── Step 7: Save model bundle ────────────────────────────────
print("\n[7/7] Saving model bundle...")
model_path = os.path.join(BASE_DIR, 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({
        'tfidf':            tfidf,
        'clf':              best_clf,
        'disease_severity': disease_severity,
        'best_model_name':  best_name,
        'accuracy':         best_acc,
        'nlp_stopwords':    STOP_WORDS,
        'lemma_rules':      LEMMA_RULES,
    }, f)

print(f"  Saved -> {model_path}")
print("\nDone! Run:  python app.py")