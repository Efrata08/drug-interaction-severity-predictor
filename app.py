import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from scipy.sparse import hstack, csr_matrix

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MedCheck AI — Drug Interaction Predictor",
    page_icon="💊",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f1f5f9;
}

.block-container {
    max-width: 700px;
    padding: 2rem 1rem;
}

/* Hide streamlit default elements */
#MainMenu, footer, header {visibility: hidden;}

/* NAV */
.nav-bar {
    background: #1a3a5c;
    border-radius: 16px 16px 0 0;
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
}
.nav-left { display: flex; align-items: center; gap: 10px; }
.nav-cross {
    width: 30px; height: 30px;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: #e05252; font-weight: 700;
}
.nav-title { font-size: 13px; font-weight: 500; color: #fff; margin: 0; }
.nav-sub { font-size: 10px; color: #90b8d8; margin: 0; }
.nav-badge {
    background: rgba(255,255,255,.1);
    border: 1px solid rgba(255,255,255,.15);
    color: #cbd5e1; font-size: 10px;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: .05em;
}

/* MAIN CARD */
.main-card {
    background: #fff;
    border-radius: 0 0 16px 16px;
    border: 1px solid #e2e8f0;
    border-top: none;
    padding: 1.75rem 1.5rem 1.25rem;
    margin-bottom: 1rem;
}

/* HERO */
.hero-row { display: flex; align-items: flex-start; gap: 12px; margin-bottom: 1.25rem; }
.rx-symbol {
    font-size: 52px; font-weight: 300;
    color: #1a3a5c; line-height: 1;
    font-family: Georgia, serif; opacity: .85;
}
.hero-eyebrow { font-size: 9px; letter-spacing: .15em; color: #94a3b8; margin-bottom: .3rem; }
.hero-title { font-size: 20px; font-weight: 600; color: #1a3a5c; margin-bottom: .25rem; line-height: 1.25; }
.hero-desc { font-size: 11px; color: #64748b; line-height: 1.7; }
.hero-divider { border: none; border-top: 1px dashed #e2e8f0; margin: 0 0 1.25rem; }

/* DRUG CARDS */
.drug-cards-row {
    display: grid;
    grid-template-columns: 1fr 36px 1fr;
    gap: 10px;
    align-items: center;
    margin-bottom: 1rem;
}
.drug-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    position: relative;
    overflow: hidden;
}
.drug-card-bar-a {
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #e05252, #f97316);
}
.drug-card-bar-b {
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
}
.drug-card-type {
    font-size: 9px; letter-spacing: .1em; color: #94a3b8;
    margin-bottom: .35rem; display: flex; align-items: center; gap: 5px;
}
.dot-a { width: 5px; height: 5px; border-radius: 50%; background: #e05252; display: inline-block; }
.dot-b { width: 5px; height: 5px; border-radius: 50%; background: #6366f1; display: inline-block; }
.drug-card-name { font-size: 15px; font-weight: 500; color: #1a3a5c; margin-bottom: .2rem; }
.drug-card-hint { font-size: 10px; color: #94a3b8; }
.vs-circle {
    width: 36px; height: 36px; border-radius: 50%;
    background: #f1f5f9; border: 1px solid #e2e8f0;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 500; color: #94a3b8;
    margin: 0 auto;
}

/* RESULT */
.result-major {
    background: #fff; border: 1px solid #fca5a5;
    border-radius: 12px; overflow: hidden; margin-bottom: 1rem;
}
.result-moderate {
    background: #fff; border: 1px solid #fde68a;
    border-radius: 12px; overflow: hidden; margin-bottom: 1rem;
}
.result-minor {
    background: #fff; border: 1px solid #bbf7d0;
    border-radius: 12px; overflow: hidden; margin-bottom: 1rem;
}
.result-top-major { background: #fef2f2; padding: 1.25rem; border-bottom: 1px dashed #fca5a5; }
.result-top-moderate { background: #fffbeb; padding: 1.25rem; border-bottom: 1px dashed #fde68a; }
.result-top-minor { background: #f0fdf4; padding: 1.25rem; border-bottom: 1px dashed #bbf7d0; }
.result-inner { display: flex; align-items: center; gap: 14px; }
.severity-ring-major {
    width: 44px; height: 44px; border-radius: 50%;
    border: 2px solid #e05252; background: #fff;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 18px;
}
.severity-ring-moderate {
    width: 44px; height: 44px; border-radius: 50%;
    border: 2px solid #fbbf24; background: #fff;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 18px;
}
.severity-ring-minor {
    width: 44px; height: 44px; border-radius: 50%;
    border: 2px solid #22c55e; background: #fff;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 18px;
}
.result-label-major { font-size: 14px; font-weight: 600; color: #991b1b; margin-bottom: 3px; }
.result-label-moderate { font-size: 14px; font-weight: 600; color: #92400e; margin-bottom: 3px; }
.result-label-minor { font-size: 14px; font-weight: 600; color: #166534; margin-bottom: 3px; }
.result-sub { font-size: 11px; color: #64748b; line-height: 1.5; }
.result-conf { margin-left: auto; text-align: right; }
.conf-val { font-size: 22px; font-weight: 600; color: #1a3a5c; }
.conf-lbl { font-size: 9px; color: #94a3b8; letter-spacing: .05em; }

/* SCALE */
.scale-row {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 6px; padding: 1rem 1.25rem;
}
.scale-minor { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 8px; text-align: center; }
.scale-moderate { background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 8px; text-align: center; }
.scale-major { background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 8px; text-align: center; }
.scale-dot-minor { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; margin: 0 auto 5px; }
.scale-dot-moderate { width: 6px; height: 6px; border-radius: 50%; background: #fbbf24; margin: 0 auto 5px; }
.scale-dot-major { width: 6px; height: 6px; border-radius: 50%; background: #e05252; margin: 0 auto 5px; }
.scale-name-minor { font-size: 11px; font-weight: 500; color: #166534; margin-bottom: 2px; }
.scale-name-moderate { font-size: 11px; font-weight: 500; color: #92400e; margin-bottom: 2px; }
.scale-name-major { font-size: 11px; font-weight: 500; color: #991b1b; margin-bottom: 2px; }
.scale-desc { font-size: 9px; color: #94a3b8; }
.scale-active-border { outline: 2px solid #1a3a5c; outline-offset: 2px; border-radius: 8px; }

/* STATS FOOTER */
.stats-footer {
    background: #f8fafc; border-top: 1px dashed #e2e8f0;
    display: grid; grid-template-columns: repeat(4,1fr);
    border-radius: 0 0 16px 16px;
}
.stat-item { text-align: center; padding: .85rem .5rem; border-right: 1px solid #e2e8f0; }
.stat-item:last-child { border-right: none; }
.stat-val { font-size: 15px; font-weight: 600; color: #1a3a5c; margin-bottom: 2px; }
.stat-lbl { font-size: 9px; color: #94a3b8; }

/* DISCLAIMER */
.disclaimer {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 10px; padding: .75rem 1rem;
    font-size: 10px; color: #78716c; line-height: 1.6;
    margin-top: 1rem;
}
.disclaimer b { color: #92400e; }

/* Streamlit button override */
.stButton > button {
    background: #1a3a5c !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    letter-spacing: .02em !important;
}
.stButton > button:hover {
    background: #0f2640 !important;
}

/* Input override */
.stTextInput > div > div > input {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    color: #1a3a5c !important;
    padding: .6rem .9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a3a5c !important;
    box-shadow: 0 0 0 2px rgba(26,58,92,.1) !important;
}
label { font-size: 10px !important; letter-spacing: .08em !important; color: #94a3b8 !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
root_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(root_dir, 'models')

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(models_dir, 'drug_interaction_model.pkl'))
    le_level = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    le_a_atc = joblib.load(os.path.join(models_dir, 'le_a_atc.pkl'))
    le_b_atc = joblib.load(os.path.join(models_dir, 'le_b_atc.pkl'))
    le_a_groups = joblib.load(os.path.join(models_dir, 'le_a_groups.pkl'))
    le_b_groups = joblib.load(os.path.join(models_dir, 'le_b_groups.pkl'))
    tfidf_a = joblib.load(os.path.join(models_dir, 'tfidf_a.pkl'))
    tfidf_b = joblib.load(os.path.join(models_dir, 'tfidf_b.pkl'))
    return model, le_level, le_a_atc, le_b_atc, le_a_groups, le_b_groups, tfidf_a, tfidf_b

@st.cache_data
def load_drugbank():
    return pd.read_csv(os.path.join(root_dir, 'drugbank_lookup.csv'))

model, le_level, le_a_atc, le_b_atc, le_a_groups, le_b_groups, tfidf_a, tfidf_b = load_model()
drugbank = load_drugbank()

# ── Helper functions ──────────────────────────────────────────
def get_drug_features(drug_name):
    name = drug_name.lower().strip()
    match = drugbank[drugbank['name'] == name]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        'atc': str(row['atc-codes']) if pd.notna(row['atc-codes']) else '',
        'groups': str(row['groups']) if pd.notna(row['groups']) else '',
        'mechanism': str(row['mechanism-of-action']) if pd.notna(row['mechanism-of-action']) else '',
        'metabolism': str(row['metabolism']) if pd.notna(row['metabolism']) else '',
        'targets': str(row['targets']) if pd.notna(row['targets']) else '',
        'cyp': str(row['cyp_enzymes']) if pd.notna(row['cyp_enzymes']) else ''
    }

def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except:
        return 0

def predict(drug_a_name, drug_b_name):
    feat_a = get_drug_features(drug_a_name)
    feat_b = get_drug_features(drug_b_name)
    if feat_a is None:
        return None, None, f"'{drug_a_name}' not found in DrugBank. Try the generic name (e.g. 'acetylsalicylic acid' for aspirin)."
    if feat_b is None:
        return None, None, f"'{drug_b_name}' not found in DrugBank. Try the generic name."
    a_atc_enc = safe_encode(le_a_atc, feat_a['atc'])
    b_atc_enc = safe_encode(le_b_atc, feat_b['atc'])
    a_groups_enc = safe_encode(le_a_groups, feat_a['groups'])
    b_groups_enc = safe_encode(le_b_groups, feat_b['groups'])
    cyp_a = set(feat_a['cyp'].split()) if feat_a['cyp'] else set()
    cyp_b = set(feat_b['cyp'].split()) if feat_b['cyp'] else set()
    shared = 1 if cyp_a and cyp_b and cyp_a.intersection(cyp_b) else 0
    X_structured = csr_matrix([[a_atc_enc, b_atc_enc, a_groups_enc, b_groups_enc, shared]])
    text_a = feat_a['mechanism'] + ' ' + feat_a['metabolism'] + ' ' + feat_a['targets'] + ' ' + feat_a['cyp']
    text_b = feat_b['mechanism'] + ' ' + feat_b['metabolism'] + ' ' + feat_b['targets'] + ' ' + feat_b['cyp']
    X_text_a = tfidf_a.transform([text_a])
    X_text_b = tfidf_b.transform([text_b])
    X = hstack([X_structured, X_text_a, X_text_b])
    prediction_enc = model.predict(X)[0]
    prediction = le_level.inverse_transform([prediction_enc])[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities) * 100
    return prediction, confidence, None

# ── UI ────────────────────────────────────────────────────────

# NAV
st.markdown("""
<div class="nav-bar">
  <div class="nav-left">
    <div class="nav-cross">✚</div>
    <div>
      <div class="nav-title">MedCheck AI</div>
      <div class="nav-sub">Drug Interaction Severity Predictor</div>
    </div>
  </div>
  <div class="nav-badge">AI4ALL · Group 3C · 2026</div>
</div>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="main-card">
  <div class="hero-row">
    <div class="rx-symbol">℞</div>
    <div>
      <div class="hero-eyebrow">CLINICAL DECISION SUPPORT</div>
      <div class="hero-title">Drug Interaction Analysis</div>
      <div class="hero-desc">Enter two medications to assess their combined severity using pharmacological features from DDInter and DrugBank.</div>
    </div>
  </div>
  <hr class="hero-divider">
</div>
""", unsafe_allow_html=True)

# INPUTS
col1, col2 = st.columns([1, 1])
with col1:
    drug_a = st.text_input("MEDICATION A", placeholder="e.g. warfarin")
with col2:
    drug_b = st.text_input("MEDICATION B", placeholder="e.g. sertraline")

# Drug search expander
with st.expander("Browse available drug names"):
    search = st.text_input("Search", placeholder="Type to search...", key="search")
    if search:
        matches = drugbank[drugbank['name'].str.contains(search.lower().strip(), na=False)]['name'].head(20).tolist()
        if matches:
            st.write(", ".join(matches))
        else:
            st.write("No matches found")

st.markdown("")
predict_btn = st.button("℞  Analyze interaction")

# RESULT
if predict_btn:
    if not drug_a or not drug_b:
        st.warning("Please enter both drug names.")
    elif drug_a.lower().strip() == drug_b.lower().strip():
        st.warning("Please enter two different drugs.")
    else:
        with st.spinner("Analyzing..."):
            prediction, confidence, error = predict(drug_a, drug_b)

        if error:
            st.markdown(f"""
            <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;padding:1rem 1.25rem;font-size:12px;color:#991b1b;margin-top:1rem">
              ⚠️ {error}
            </div>""", unsafe_allow_html=True)
        else:
            conf_str = f"{confidence:.1f}%"

            if prediction == "Major":
                icon, label, sub = "⚠️", "Major interaction detected", "High risk — avoid this combination or monitor very closely. Consult provider before prescribing."
                top_class, card_class, ring_class, label_class = "result-top-major", "result-major", "severity-ring-major", "result-label-major"
                minor_active, moderate_active, major_active = "", "", "scale-active-border"
            elif prediction == "Moderate":
                icon, label, sub = "🟡", "Moderate interaction detected", "Significant risk — may require dose adjustment or closer monitoring."
                top_class, card_class, ring_class, label_class = "result-top-moderate", "result-moderate", "severity-ring-moderate", "result-label-moderate"
                minor_active, moderate_active, major_active = "", "scale-active-border", ""
            else:
                icon, label, sub = "✅", "Minor interaction detected", "Low risk — generally manageable with standard care."
                top_class, card_class, ring_class, label_class = "result-top-minor", "result-minor", "severity-ring-minor", "result-label-minor"
                minor_active, moderate_active, major_active = "scale-active-border", "", ""

            st.markdown(f"""
            <div class="{card_class}" style="margin-top:1rem">
              <div class="{top_class}">
                <div class="result-inner">
                  <div class="{ring_class}">{icon}</div>
                  <div>
                    <div class="{label_class}">{label}</div>
                    <div class="result-sub">{sub}</div>
                  </div>
                  <div class="result-conf">
                    <div class="conf-val">{conf_str}</div>
                    <div class="conf-lbl">CONFIDENCE</div>
                  </div>
                </div>
              </div>
              <div class="scale-row">
                <div class="scale-minor {minor_active}">
                  <div class="scale-dot-minor"></div>
                  <div class="scale-name-minor">Minor</div>
                  <div class="scale-desc">Standard care</div>
                </div>
                <div class="scale-moderate {moderate_active}">
                  <div class="scale-dot-moderate"></div>
                  <div class="scale-name-moderate">Moderate</div>
                  <div class="scale-desc">Monitor closely</div>
                </div>
                <div class="scale-major {major_active}">
                  <div class="scale-dot-major"></div>
                  <div class="scale-name-major">Major</div>
                  <div class="scale-desc">High risk</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# STATS FOOTER
st.markdown("""
<div class="stats-footer" style="margin-top:1.5rem">
  <div class="stat-item"><div class="stat-val">79%</div><div class="stat-lbl">Accuracy</div></div>
  <div class="stat-item"><div class="stat-val">82%</div><div class="stat-lbl">Major recall</div></div>
  <div class="stat-item"><div class="stat-val">4,800</div><div class="stat-lbl">Training pairs</div></div>
  <div class="stat-item"><div class="stat-val">8</div><div class="stat-lbl">Drug categories</div></div>
</div>
""", unsafe_allow_html=True)

# DISCLAIMER
st.markdown("""
<div class="disclaimer">
  <b>⚠ Disclaimer:</b> This tool is for educational purposes only and does not replace professional medical judgment.
  Always consult a licensed healthcare provider before making prescribing decisions.
</div>
""", unsafe_allow_html=True)
