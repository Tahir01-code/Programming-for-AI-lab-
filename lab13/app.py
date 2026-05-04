

import os, re, pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app      = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model bundle ────────────────────────────────────────
print("Loading model...")
with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    bundle = pickle.load(f)

tfidf            = bundle['tfidf']
clf              = bundle['clf']
disease_severity = bundle['disease_severity']
STOP_WORDS       = bundle.get('nlp_stopwords', set())
LEMMA_RULES      = bundle.get('lemma_rules', [])
print(f"  Model: {bundle.get('best_model_name','Unknown')} "
      f"(accuracy {bundle.get('accuracy',0)*100:.1f}%)")

# ── Load disease info ────────────────────────────────────────
desc_df = pd.read_csv(os.path.join(BASE_DIR, 'symptom_Description.csv'))
prec_df = pd.read_csv(os.path.join(BASE_DIR, 'symptom_precaution.csv'))
sev_df  = pd.read_csv(os.path.join(BASE_DIR, 'Symptom-severity.csv'))

desc_df['Disease'] = desc_df['Disease'].str.strip()
prec_df['Disease'] = prec_df['Disease'].str.strip()

DISEASE_INFO = {}
for _, row in desc_df.iterrows():
    DISEASE_INFO[row['Disease']] = {
        'description': row['Description'],
        'precautions': []
    }
for _, row in prec_df.iterrows():
    d     = row['Disease']
    precs = [str(row[c]).strip()
             for c in ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
             if pd.notna(row[c]) and str(row[c]).strip() not in ('', 'nan')]
    if d in DISEASE_INFO:
        DISEASE_INFO[d]['precautions'] = precs

ALL_SYMPTOMS = sorted(
    sev_df['Symptom'].str.strip().str.replace('_', ' ').tolist()
)

# ── NLP helpers ──────────────────────────────────────────────
def simple_lemmatize(word):
    if len(word) <= 4:
        return word
    for pattern, repl in LEMMA_RULES:
        result = re.sub(pattern, repl, word)
        if result != word and len(result) >= 3:
            return result
    return word

def nlp_preprocess(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z\s_]', ' ', text)
    text = text.replace('_', ' ')
    tokens = text.split()
    tokens = [simple_lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

# ── Synonym map ──────────────────────────────────────────────
SYNONYM_MAP = {
    'stomach ache': 'stomach pain', 'tummy ache': 'stomach pain',
    'throwing up':  'vomiting',     'puke':        'vomiting',
    'runny nose':   'continuous sneezing',
    'feel tired':   'fatigue',      'tired':       'fatigue',
    'short of breath': 'breathlessness',
    'can t breathe':   'breathlessness',
    'high temperature':'high fever', 'loose motion': 'diarrhoea',
    'loose stool':     'diarrhoea', 'yellow skin':  'yellowing of skin',
    'hair fall':       'hair loss', 'back ache':    'back pain',
}

def apply_synonyms(text: str) -> str:
    for phrase, replacement in SYNONYM_MAP.items():
        text = text.replace(phrase, replacement)
    return text

# ── Severity label ───────────────────────────────────────────
def severity_label(score):
    if score >= 18:   return 'Critical', '#c0392b'
    elif score >= 13: return 'High',     '#e74c3c'
    elif score >= 8:  return 'Moderate', '#f39c12'
    else:             return 'Mild',     '#27ae60'

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw  = data.get('symptoms', '').strip()
    if not raw or len(raw) < 3:
        return jsonify({'error': 'Please enter at least one symptom.'}), 400

    text    = apply_synonyms(raw.lower())
    cleaned = nlp_preprocess(text)

    if not cleaned:
        return jsonify({'error': 'Could not extract meaningful symptoms. Please rephrase.'}), 400

    vec     = tfidf.transform([cleaned])
    proba   = clf.predict_proba(vec)[0]
    classes = clf.classes_
    results = []

    for idx in proba.argsort()[-5:][::-1]:
        disease    = classes[idx]
        confidence = float(proba[idx])
        if confidence < 0.03:
            continue
        info              = DISEASE_INFO.get(disease, {})
        sev_score         = disease_severity.get(disease, 0)
        sev_lbl, sev_col  = severity_label(sev_score)
        results.append({
            'disease':     disease,
            'confidence':  round(confidence * 100, 1),
            'description': info.get('description', 'No description available.'),
            'precautions': info.get('precautions', []),
            'severity':    sev_lbl,
            'sev_color':   sev_col,
            'sev_score':   round(sev_score, 1),
        })

    if not results:
        return jsonify({'error': 'Could not determine a condition. Please add more symptoms.'}), 400

    return jsonify({'results': results, 'symptoms_processed': cleaned, 'original_input': raw})

@app.route('/symptoms-list')
def symptoms_list():
    return jsonify({'symptoms': ALL_SYMPTOMS})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'diseases': len(DISEASE_INFO)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)