"""
Hadith QnA Bot – PAI Lab 12 Task 2
Pipeline: Git Clone → Load CSVs → Clean → MiniLM Embed → FAISS Index → Flask UI
"""

import os
import re
import glob
import subprocess
import numpy as np
import pandas as pd
import faiss
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

CORPUS_DIR = "LK-Hadith-Corpus"
CORPUS_URL = "https://github.com/ShathaTm/LK-Hadith-Corpus.git"
CSV_OUT    = "cleaned_hadith.csv"
EMBED_OUT  = "hadith_embeddings.npy"
INDEX_OUT  = "faiss_index.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K      = 5

COLUMNS = [
    'Chapter_Number', 'Chapter_English', 'Chapter_Arabic',
    'Section_Number', 'Section_English', 'Section_Arabic',
    'Hadith_Number',
    'English_Hadith', 'English_Isnad', 'English_Matn',
    'Arabic_Hadith', 'Arabic_Isnad', 'Arabic_Matn',
    'Arabic_Grade', 'English_Grade', 'Cleaned_Hadith'
]

def clone_dataset():
    global CORPUS_DIR
    if os.path.exists("LK-Hadith-Corpus-main") and not os.path.exists(CORPUS_DIR):
        os.rename("LK-Hadith-Corpus-main", CORPUS_DIR)
        print("Renamed LK-Hadith-Corpus-main to LK-Hadith-Corpus")
    if os.path.exists(CORPUS_DIR):
        print("Dataset already present.")
        return
    try:
        print("Cloning LK-Hadith-Corpus ...")
        subprocess.run(["git", "clone", CORPUS_URL], check=True)
        print("Clone complete.")
    except FileNotFoundError:
        print("\nERROR: Git not found. Download dataset manually:")
        print(f"  1. Go to: {CORPUS_URL}")
        print("  2. Click Code > Download ZIP")
        print("  3. Extract and place 'LK-Hadith-Corpus-main' next to app.py")
        print("  4. Re-run app.py")
        raise SystemExit(1)

# FIX: Keep spaces so MiniLM understands sentence meaning.
# Old code: re.sub(r'[^a-zA-Z]','',text) removed ALL spaces -> broken embeddings
# New code: re.sub(r'[^a-zA-Z\s]',' ',text) keeps spaces -> correct semantics
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_hadith():
    if os.path.exists(CSV_OUT):
        df_check = pd.read_csv(CSV_OUT, nrows=3)
        sample = str(df_check['Cleaned_Hadith'].dropna().iloc[0]) if len(df_check) > 0 else ''
        # Detect old broken cache (no spaces) and rebuild
        if ' ' not in sample and len(sample) > 20:
            print("Old cache detected (no spaces) - rebuilding ...")
            os.remove(CSV_OUT)
            if os.path.exists(EMBED_OUT): os.remove(EMBED_OUT)
            if os.path.exists(INDEX_OUT): os.remove(INDEX_OUT)
        else:
            print("Loading cached cleaned_hadith.csv ...")
            return pd.read_csv(CSV_OUT)

    print("Loading CSV files ...")
    files = sorted(glob.glob(CORPUS_DIR + '/**/*.csv', recursive=True))
    print(f"Found {len(files)} CSV file(s)")

    all_hadith = []
    for f in files:
        try:
            df = pd.read_csv(f, names=COLUMNS, skiprows=1)
            df['Cleaned_Hadith'] = df['English_Hadith'].astype(str).apply(clean_text)
            all_hadith.extend(df[COLUMNS].values.tolist())
        except Exception as e:
            print(f"Skipping {f}: {e}")

    hadith_df = pd.DataFrame(all_hadith, columns=COLUMNS)
    hadith_df = hadith_df[hadith_df['English_Hadith'].astype(str).str.strip() != '']
    hadith_df = hadith_df[hadith_df['English_Hadith'].astype(str) != 'nan']
    hadith_df = hadith_df.reset_index(drop=True)
    hadith_df.to_csv(CSV_OUT, index=False)
    print(f"Loaded {len(hadith_df)} hadiths.")
    return hadith_df

def build_embeddings(hadith_df, model):
    if os.path.exists(EMBED_OUT):
        print("Loading cached embeddings ...")
        return np.load(EMBED_OUT)
    print("Encoding hadiths with MiniLM (may take a few minutes) ...")
    embeddings = model.encode(hadith_df['Cleaned_Hadith'].tolist(), batch_size=128, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype='float32')
    np.save(EMBED_OUT, embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def build_faiss_index(embeddings):
    if os.path.exists(INDEX_OUT):
        print("Loading cached FAISS index ...")
        return faiss.read_index(INDEX_OUT)
    print("Building FAISS IndexFlatL2 ...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, INDEX_OUT)
    print(f"FAISS index ready: {index.ntotal} vectors, dim={d}")
    return index

def get_similar_hadith(query, count=TOP_K):
    query_embedding = MODEL.encode([clean_text(query)]).astype('float32')
    distances, indices = FAISS_INDEX.search(query_embedding, count)
    results = []
    for i in range(count):
        idx = indices[0][i]
        if idx < 0 or idx >= len(HADITH_DF):
            continue
        row = HADITH_DF.iloc[idx]
        results.append({
            "rank":          i + 1,
            "hadith":        str(row['English_Hadith']),
            "chapter":       str(row.get('Chapter_English', '')),
            "section":       str(row.get('Section_English', '')),
            "grade":         str(row.get('English_Grade', '')),
            "hadith_number": str(row.get('Hadith_Number', '')),
            "distance":      float(round(distances[0][i], 4)),
        })
    return results

print("Starting Hadith Bot pipeline ...")
clone_dataset()
HADITH_DF   = load_hadith()
MODEL       = SentenceTransformer(MODEL_NAME)
EMBEDDINGS  = build_embeddings(HADITH_DF, MODEL)
FAISS_INDEX = build_faiss_index(EMBEDDINGS)
print("Bot is ready!\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data  = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Please enter a search query."}), 400
    results = get_similar_hadith(query)
    return jsonify({"query": query, "results": results})

if __name__ == "__main__":
    app.run(debug=True)