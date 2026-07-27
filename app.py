import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack, csr_matrix

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Interaction Severity Predictor",
    page_icon="💊",
    layout="centered"
)

# ── Load model and encoders ───────────────────────────────────
import os
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
        'atc': str(row['atc-codes']),
        'groups': str(row['groups']),
        'mechanism': str(row['mechanism-of-action']),
        'metabolism': str(row['metabolism']),
        'targets': str(row['targets']),
        'cyp': str(row['cyp_enzymes'])
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
        return None, None, f"'{drug_a_name}' not found in DrugBank. Try the generic name."
    if feat_b is None:
        return None, None, f"'{drug_b_name}' not found in DrugBank. Try the generic name."

    # Structured features
    a_atc_enc = safe_encode(le_a_atc, feat_a['atc'])
    b_atc_enc = safe_encode(le_b_atc, feat_b['atc'])
    a_groups_enc = safe_encode(le_a_groups, feat_a['groups'])
    b_groups_enc = safe_encode(le_b_groups, feat_b['groups'])

    # Shared enzyme
    cyp_a = set(feat_a['cyp'].split()) if feat_a['cyp'] else set()
    cyp_b = set(feat_b['cyp'].split()) if feat_b['cyp'] else set()
    shared = 1 if cyp_a and cyp_b and cyp_a.intersection(cyp_b) else 0

    X_structured = csr_matrix([[
        a_atc_enc, b_atc_enc,
        a_groups_enc, b_groups_enc,
        shared
    ]])

    # Text features
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
st.title("💊 Drug Interaction Severity Predictor")
st.markdown("*Built by Group 3C — AI4ALL Ignite 2026*")
st.markdown("---")

st.markdown("""
This tool predicts whether a combination of two medications is 
**Minor**, **Moderate**, or **Major** risk based on pharmacological 
features including drug category, metabolizing enzymes, and 
molecular targets.
""")

st.markdown("### Enter two drug names")

col1, col2 = st.columns(2)
with col1:
    drug_a = st.text_input("Drug A", placeholder="e.g. warfarin")
with col2:
    drug_b = st.text_input("Drug B", placeholder="e.g. aspirin")

# Show available drugs hint
with st.expander("Need help? Browse available drug names"):
    search = st.text_input("Search drugs", placeholder="Type to search...")
    if search:
        matches = drugbank[drugbank['name'].str.contains(
            search.lower(), na=False)]['name'].head(20).tolist()
        if matches:
            st.write(matches)
        else:
            st.write("No matches found")

st.markdown("")

if st.button("Predict Interaction", type="primary", use_container_width=True):
    if not drug_a or not drug_b:
        st.warning("Please enter both drug names")
    elif drug_a.lower().strip() == drug_b.lower().strip():
        st.warning("Please enter two different drugs")
    else:
        with st.spinner("Analyzing interaction..."):
            prediction, confidence, error = predict(drug_a, drug_b)

        if error:
            st.error(f"⚠️ {error}")
        else:
            st.markdown("---")
            st.markdown("### Result")

            if prediction == "Major":
                st.error(f"🔴 **MAJOR** interaction detected")
                st.markdown("""
                **Clinical significance:** High risk. This combination 
                should typically be avoided or require close monitoring 
                and possible dose adjustment.
                """)
            elif prediction == "Moderate":
                st.warning(f"🟡 **MODERATE** interaction detected")
                st.markdown("""
                **Clinical significance:** Significant risk. The combination 
                may require dose adjustment, increased monitoring, or 
                alternative medication consideration.
                """)
            else:
                st.success(f"🟢 **MINOR** interaction detected")
                st.markdown("""
                **Clinical significance:** Low risk. The combination is 
                generally manageable with standard care.
                """)

            st.metric("Model confidence", f"{confidence:.1f}%")

            st.markdown("---")
            st.caption("""
            ⚠️ **Disclaimer:** This tool is for educational purposes only 
            and does not replace professional medical judgment. Always 
            consult a licensed healthcare provider before making 
            prescribing decisions.
            """)

st.markdown("---")
st.markdown("""
**Model:** Random Forest Classifier | **Accuracy:** 79% | 
**Major Recall:** 82% | **Training data:** DDInter + DrugBank
""")
