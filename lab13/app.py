from flask import Flask, render_template, request, jsonify
import pickle, re, os, pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load trained model ───────────────────────────────────────
with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    bundle = pickle.load(f)
tfidf            = bundle['tfidf']
clf              = bundle['clf']
disease_severity = bundle['disease_severity']   # avg severity score per disease

# ── Load disease info from CSVs ──────────────────────────────
desc_df = pd.read_csv(os.path.join(BASE_DIR, 'symptom_Description.csv'))
prec_df = pd.read_csv(os.path.join(BASE_DIR, 'symptom_precaution.csv'))
desc_df['Disease'] = desc_df['Disease'].str.strip()
prec_df['Disease'] = prec_df['Disease'].str.strip()

DISEASE_INFO = {}
for _, row in desc_df.iterrows():
    DISEASE_INFO[row['Disease']] = {'description': row['Description'], 'precautions': []}

for _, row in prec_df.iterrows():
    d = row['Disease']
    precs = [str(row[c]).strip() for c in ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
             if pd.notna(row[c]) and str(row[c]).strip() not in ('', 'nan')]
    if d in DISEASE_INFO:
        DISEASE_INFO[d]['precautions'] = precs

# ── Severity label from score ────────────────────────────────
def severity_label(score):
    if score >= 18:   return 'Critical', '#c0392b'
    elif score >= 13: return 'High',     '#e74c3c'
    elif score >= 8:  return 'Moderate', '#f39c12'
    else:             return 'Mild',     '#2ecc71'

# ── Text cleaning ────────────────────────────────────────────
def clean_input(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(text.split())

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

    cleaned = clean_input(raw)
    vec     = tfidf.transform([cleaned])
    proba   = clf.predict_proba(vec)[0]
    classes = clf.classes_

    top3_idx = proba.argsort()[-3:][::-1]
    results  = []

    for idx in top3_idx:
        disease    = classes[idx]
        confidence = float(proba[idx])
        if confidence < 0.03:
            continue

        info        = DISEASE_INFO.get(disease, {})
        sev_score   = disease_severity.get(disease, 0)
        sev_lbl, sev_color = severity_label(sev_score)

        results.append({
            'disease':     disease,
            'confidence':  round(confidence * 100, 1),
            'description': info.get('description', 'No description available.'),
            'precautions': info.get('precautions', []),
            'severity':    sev_lbl,
            'sev_color':   sev_color,
            'sev_score':   round(sev_score, 1),
        })

    if not results:
        return jsonify({'error': 'Could not determine a condition. Please describe your symptoms in more detail.'}), 400

    return jsonify({'results': results, 'symptoms_processed': cleaned})

@app.route('/symptoms-list')
def symptoms_list():
    """Returns list of known symptoms — useful for autocomplete."""
    sev_df = pd.read_csv(os.path.join(BASE_DIR, 'Symptom-severity.csv'))
    syms   = sorted(sev_df['Symptom'].str.strip().str.replace('_', ' ').tolist())
    return jsonify({'symptoms': syms})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
