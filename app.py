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

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f1f5f9; }
.block-container { max-width: 720px; padding: 2rem 1rem 3rem; }
#MainMenu, footer, header { visibility: hidden; }

/* NAV */
.nav-bar {
    background: #1a3a5c; border-radius: 16px 16px 0 0;
    padding: 1rem 1.5rem; display: flex; align-items: center;
    justify-content: space-between;
}
.nav-left { display: flex; align-items: center; gap: 10px; }
.nav-cross {
    width: 30px; height: 30px; background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.2); border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: #e05252; font-weight: 700;
}
.nav-title { font-size: 13px; font-weight: 500; color: #fff; margin: 0; }
.nav-sub { font-size: 10px; color: #90b8d8; margin: 0; }
.nav-badge {
    background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.15);
    color: #cbd5e1; font-size: 10px; padding: 3px 10px;
    border-radius: 20px; letter-spacing: .05em;
}

/* HERO */
.hero-card {
    background: #fff; border-radius: 0 0 16px 16px;
    border: 1px solid #e2e8f0; border-top: none;
    padding: 1.6rem 1.5rem 1.4rem; margin-bottom: 1.25rem;
}
.hero-row { display: flex; align-items: flex-start; gap: 14px; }
.rx-symbol {
    font-size: 50px; font-weight: 300; color: #1a3a5c;
    line-height: .95; font-family: Georgia, serif; opacity: .85;
}
.hero-eyebrow { font-size: 9px; letter-spacing: .15em; color: #94a3b8; margin-bottom: .3rem; }
.hero-title { font-size: 20px; font-weight: 600; color: #1a3a5c; margin-bottom: .3rem; }
.hero-desc { font-size: 11.5px; color: #64748b; line-height: 1.65; }

/* SECTION LABEL */
.sec-label {
    font-size: 10px; letter-spacing: .12em; color: #94a3b8;
    font-weight: 600; margin: 0 0 .5rem;
}

/* PICKER PANEL */
.picker {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 1.1rem 1.25rem .5rem;
    margin-bottom: 1rem;
}

/* RESULT */
.res-wrap { border-radius: 14px; overflow: hidden; margin-top: 1.1rem; border: 1px solid; }
.res-major    { border-color: #fca5a5; }
.res-moderate { border-color: #fde68a; }
.res-minor    { border-color: #bbf7d0; }
.res-top { padding: 1.2rem 1.35rem; display: flex; align-items: center; gap: 15px; }
.res-top-major    { background: #fef2f2; border-bottom: 1px dashed #fca5a5; }
.res-top-moderate { background: #fffbeb; border-bottom: 1px dashed #fde68a; }
.res-top-minor    { background: #f0fdf4; border-bottom: 1px dashed #bbf7d0; }
.ring {
    width: 46px; height: 46px; border-radius: 50%; background: #fff;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 19px; border: 2px solid;
}
.ring-major { border-color: #e05252; }
.ring-moderate { border-color: #fbbf24; }
.ring-minor { border-color: #22c55e; }
.res-label { font-size: 14.5px; font-weight: 600; margin-bottom: 3px; }
.lab-major { color: #991b1b; } .lab-moderate { color: #92400e; } .lab-minor { color: #166534; }
.res-sub { font-size: 11.5px; color: #64748b; line-height: 1.5; }
.res-conf { margin-left: auto; text-align: right; }
.conf-val { font-size: 23px; font-weight: 600; color: #1a3a5c; }
.conf-lbl { font-size: 9px; color: #94a3b8; letter-spacing: .05em; }

/* WHY PANEL */
.why { background: #fff; padding: .95rem 1.35rem 1.1rem; }
.why-lbl { font-size: 9.5px; letter-spacing: .12em; color: #94a3b8; font-weight: 600; margin-bottom: .5rem; }
.why-row { font-size: 11.5px; color: #334155; line-height: 1.75; }
.why-row b { color: #1a3a5c; }
.chip {
    display: inline-block; background: #eef2f7; color: #1a3a5c;
    font-size: 10px; padding: 2px 8px; border-radius: 20px; margin-right: 4px;
}
.chip-hot { background: #fef2f2; color: #991b1b; }

/* SEVERITY SCALE */
.scale { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;
         padding: 0 1.35rem 1.2rem; background: #fff; }
.sc { border-radius: 8px; padding: 8px; text-align: center; border: 1px solid; }
.sc-minor { background: #f0fdf4; border-color: #bbf7d0; }
.sc-moderate { background: #fffbeb; border-color: #fde68a; }
.sc-major { background: #fef2f2; border-color: #fca5a5; }
.sc-dot { width: 6px; height: 6px; border-radius: 50%; margin: 0 auto 5px; }
.sc-name { font-size: 11px; font-weight: 600; margin-bottom: 2px; }
.sc-desc { font-size: 9px; color: #94a3b8; }
.sc-on { outline: 2px solid #1a3a5c; outline-offset: 2px; }

/* STATS */
.stats { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px;
         display: grid; grid-template-columns: repeat(4,1fr); margin-top: 1.25rem; }
.stat { text-align: center; padding: .85rem .5rem; border-right: 1px solid #e2e8f0; }
.stat:last-child { border-right: none; }
.stat-val { font-size: 15px; font-weight: 600; color: #1a3a5c; }
.stat-lbl { font-size: 9px; color: #94a3b8; }

/* DISCLAIMER */
.disc { background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px;
        padding: .75rem 1rem; font-size: 10px; color: #78716c;
        line-height: 1.6; margin-top: 1rem; }
.disc b { color: #92400e; }

/* ── Streamlit widget overrides ───────────────────────────── */
.stButton > button {
    background: #1a3a5c !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    padding: .7rem 1.5rem !important; font-size: 13px !important;
    font-weight: 600 !important; width: 100% !important;
    letter-spacing: .02em !important;
}
.stButton > button:hover { background: #0f2640 !important; }

/* selectbox: white field, navy text, clear border */
div[data-baseweb="select"] > div {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
    color: #1a3a5c !important;
    min-height: 42px !important;
}
div[data-baseweb="select"] > div:hover { border-color: #1a3a5c !important; }
div[data-baseweb="select"] svg { color: #64748b !important; }
/* dropdown menu */
ul[role="listbox"] { background: #fff !important; border: 1px solid #cbd5e1 !important; }
li[role="option"] { font-size: 13px !important; color: #1a3a5c !important; }
li[role="option"]:hover { background: #eef2f7 !important; }

label { font-size: 10px !important; letter-spacing: .1em !important;
        color: #94a3b8 !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────
root_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(root_dir, 'models')

@st.cache_resource
def load_model():
    return (
        joblib.load(os.path.join(models_dir, 'drug_interaction_model.pkl')),
        joblib.load(os.path.join(models_dir, 'label_encoder.pkl')),
        joblib.load(os.path.join(models_dir, 'le_a_atc.pkl')),
        joblib.load(os.path.join(models_dir, 'le_b_atc.pkl')),
        joblib.load(os.path.join(models_dir, 'le_a_groups.pkl')),
        joblib.load(os.path.join(models_dir, 'le_b_groups.pkl')),
        joblib.load(os.path.join(models_dir, 'tfidf_a.pkl')),
        joblib.load(os.path.join(models_dir, 'tfidf_b.pkl')),
    )

# ATC level-1 categories, as they appear in DrugBank's atc-codes text
ATC1 = {
    'ALIMENTARY TRACT AND METABOLISM':            'Alimentary tract & metabolism',
    'BLOOD AND BLOOD FORMING ORGANS':             'Blood & blood-forming organs',
    'CARDIOVASCULAR SYSTEM':                      'Cardiovascular system',
    'DERMATOLOGICALS':                            'Dermatologicals',
    'GENITO URINARY SYSTEM AND SEX HORMONES':     'Genitourinary & sex hormones',
    'SYSTEMIC HORMONAL PREPARATIONS':             'Systemic hormonal preparations',
    'ANTIINFECTIVES FOR SYSTEMIC USE':            'Antiinfectives (systemic)',
    'ANTINEOPLASTIC AND IMMUNOMODULATING AGENTS': 'Antineoplastic & immunomodulating',
    'MUSCULO-SKELETAL SYSTEM':                    'Musculo-skeletal system',
    'NERVOUS SYSTEM':                             'Nervous system',
    'ANTIPARASITIC PRODUCTS':                     'Antiparasitic products',
    'RESPIRATORY SYSTEM':                         'Respiratory system',
    'SENSORY ORGANS':                             'Sensory organs',
    'VARIOUS':                                    'Various',
}

@st.cache_data
def load_drugbank():
    db = pd.read_csv(os.path.join(root_dir, 'drugbank_lookup.csv'))
    # Only drugs with an ATC code can produce a valid prediction —
    # ATC class is a required model feature.
    db = db[db['atc-codes'].notna() & db['name'].notna()].copy()
    db['name'] = db['name'].str.lower().str.strip()
    db = db.drop_duplicates(subset='name')

    def cats(text):
        up = str(text).upper()
        return [label for key, label in ATC1.items() if key in up]

    db['categories'] = db['atc-codes'].apply(cats)
    db['display'] = db['name'].str.title()
    return db.sort_values('display').reset_index(drop=True)

model, le_level, le_a_atc, le_b_atc, le_a_groups, le_b_groups, tfidf_a, tfidf_b = load_model()
drugbank = load_drugbank()


# ── Prediction helpers ────────────────────────────────────────
def features(name):
    row = drugbank[drugbank['name'] == name]
    if row.empty:
        return None
    r = row.iloc[0]
    g = lambda c: str(r[c]) if pd.notna(r[c]) else ''
    return {'atc': g('atc-codes'), 'groups': g('groups'),
            'mechanism': g('mechanism-of-action'), 'metabolism': g('metabolism'),
            'targets': g('targets'), 'cyp': g('cyp_enzymes')}


def safe_encode(enc, val):
    try:
        return enc.transform([val])[0]
    except Exception:
        return 0


def predict(a_name, b_name):
    fa, fb = features(a_name), features(b_name)
    if fa is None or fb is None:
        return None, None, None

    cyp_a = set(fa['cyp'].split()) if fa['cyp'] else set()
    cyp_b = set(fb['cyp'].split()) if fb['cyp'] else set()
    shared_set = cyp_a & cyp_b
    shared = 1 if shared_set else 0

    X_struct = csr_matrix([[
        safe_encode(le_a_atc, fa['atc']),
        safe_encode(le_b_atc, fb['atc']),
        safe_encode(le_a_groups, fa['groups']),
        safe_encode(le_b_groups, fb['groups']),
        shared,
    ]])
    ta = ' '.join([fa['mechanism'], fa['metabolism'], fa['targets'], fa['cyp']])
    tb_ = ' '.join([fb['mechanism'], fb['metabolism'], fb['targets'], fb['cyp']])
    X = hstack([X_struct, tfidf_a.transform([ta]), tfidf_b.transform([tb_])])

    pred = le_level.inverse_transform([model.predict(X)[0]])[0]
    conf = max(model.predict_proba(X)[0]) * 100
    return pred, conf, {'shared': sorted(shared_set),
                        'cyp_a': sorted(cyp_a), 'cyp_b': sorted(cyp_b)}


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
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
<div class="hero-card">
  <div class="hero-row">
    <div class="rx-symbol">℞</div>
    <div>
      <div class="hero-eyebrow">CLINICAL DECISION SUPPORT</div>
      <div class="hero-title">Drug Interaction Analysis</div>
      <div class="hero-desc">Select two medications to assess their combined severity
      using pharmacological features from DDInter and DrugBank.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Category filter ───────────────────────────────────────────
st.markdown('<div class="sec-label">FILTER BY DRUG CATEGORY</div>', unsafe_allow_html=True)
all_cats = sorted({c for lst in drugbank['categories'] for c in lst})
choice = st.selectbox(
    "Category",
    ["All categories"] + all_cats,
    label_visibility="collapsed",
)

if choice == "All categories":
    pool = drugbank
else:
    pool = drugbank[drugbank['categories'].apply(lambda l: choice in l)]

st.caption(f"{len(pool):,} medications available in this category")

# ── Drug pickers ──────────────────────────────────────────────
options = pool['display'].tolist()
name_of = dict(zip(pool['display'], pool['name']))

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="sec-label">MEDICATION A</div>', unsafe_allow_html=True)
    a_disp = st.selectbox("Medication A", options, index=None,
                          placeholder="Type to search…",
                          label_visibility="collapsed", key="a")
with c2:
    st.markdown('<div class="sec-label">MEDICATION B</div>', unsafe_allow_html=True)
    b_disp = st.selectbox("Medication B", options, index=None,
                          placeholder="Type to search…",
                          label_visibility="collapsed", key="b")

st.markdown("")
go = st.button("℞   Analyze interaction")

# ── Result ────────────────────────────────────────────────────
if go:
    if not a_disp or not b_disp:
        st.warning("Please select both medications.")
    elif a_disp == b_disp:
        st.warning("Please select two different medications.")
    else:
        with st.spinner("Analyzing…"):
            pred, conf, detail = predict(name_of[a_disp], name_of[b_disp])

        if pred is None:
            st.error("Could not build features for that pair.")
        else:
            cfg = {
                'Major':    ('⚠️', 'Major interaction detected',
                             'High risk — avoid this combination or monitor very closely. '
                             'Consult a provider before prescribing.', 'major'),
                'Moderate': ('🟡', 'Moderate interaction detected',
                             'Significant risk — may require dose adjustment or closer monitoring.',
                             'moderate'),
                'Minor':    ('✅', 'Minor interaction detected',
                             'Low risk — generally manageable with standard care.', 'minor'),
            }[pred]
            icon, label, sub, k = cfg
            on = {'minor': '', 'moderate': '', 'major': ''}
            on[k] = 'sc-on'

            # why-panel content
            if detail['shared']:
                chips = ' '.join(f'<span class="chip chip-hot">{e}</span>'
                                 for e in detail['shared'])
                why = (f'<div class="why-row"><b>Shared metabolic pathway.</b> Both drugs are '
                       f'processed by {chips} — the most common cause of dangerous '
                       f'interactions.</div>')
            elif detail['cyp_a'] or detail['cyp_b']:
                ca = ' '.join(f'<span class="chip">{e}</span>' for e in detail['cyp_a']) or '<span class="chip">not documented</span>'
                cb = ' '.join(f'<span class="chip">{e}</span>' for e in detail['cyp_b']) or '<span class="chip">not documented</span>'
                why = (f'<div class="why-row"><b>No shared CYP enzyme detected.</b><br>'
                       f'{a_disp}: {ca}<br>{b_disp}: {cb}</div>')
            else:
                why = ('<div class="why-row"><b>Limited enzyme data.</b> Neither drug has '
                       'documented CYP metabolism in DrugBank — this prediction relies on '
                       'drug class and mechanism text alone, so treat it with extra caution.</div>')

            st.markdown(f"""
            <div class="res-wrap res-{k}">
              <div class="res-top res-top-{k}">
                <div class="ring ring-{k}">{icon}</div>
                <div>
                  <div class="res-label lab-{k}">{label}</div>
                  <div class="res-sub">{sub}</div>
                </div>
                <div class="res-conf">
                  <div class="conf-val">{conf:.1f}%</div>
                  <div class="conf-lbl">CONFIDENCE</div>
                </div>
              </div>
              <div class="why">
                <div class="why-lbl">WHY THIS PREDICTION</div>
                {why}
              </div>
              <div class="scale">
                <div class="sc sc-minor {on['minor']}">
                  <div class="sc-dot" style="background:#22c55e"></div>
                  <div class="sc-name" style="color:#166534">Minor</div>
                  <div class="sc-desc">Standard care</div>
                </div>
                <div class="sc sc-moderate {on['moderate']}">
                  <div class="sc-dot" style="background:#fbbf24"></div>
                  <div class="sc-name" style="color:#92400e">Moderate</div>
                  <div class="sc-desc">Monitor closely</div>
                </div>
                <div class="sc sc-major {on['major']}">
                  <div class="sc-dot" style="background:#e05252"></div>
                  <div class="sc-name" style="color:#991b1b">Major</div>
                  <div class="sc-desc">High risk</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="stats">
  <div class="stat"><div class="stat-val">79%</div><div class="stat-lbl">Accuracy</div></div>
  <div class="stat"><div class="stat-val">82%</div><div class="stat-lbl">Major recall</div></div>
  <div class="stat"><div class="stat-val">4,800</div><div class="stat-lbl">Training pairs</div></div>
  <div class="stat"><div class="stat-val">8</div><div class="stat-lbl">Drug categories</div></div>
</div>
<div class="disc">
  <b>⚠ Disclaimer:</b> This tool is for educational purposes only and does not replace
  professional medical judgment. Always consult a licensed healthcare provider before
  making prescribing decisions.
</div>
""", unsafe_allow_html=True)
