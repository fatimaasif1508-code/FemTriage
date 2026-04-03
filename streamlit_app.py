"""
FemTriage — Streamlit App
Harvard HSIL Hackathon 2026

Loads pre-trained PCOS and Endometriosis models and runs inference.
Place this file at your repo root alongside the models/ folder.
"""

import os
import numpy as np
import joblib
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FemTriage",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #blue;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Hero title */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #1a1a2e;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #7a6f6a;
    margin-bottom: 2rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* Section headers */
.section-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #c4604a;
    margin-top: 1.8rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #e8ddd8;
    padding-bottom: 0.3rem;
}

/* Result cards */
.result-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    border-left: 5px solid #c4604a;
}
.result-card.low  { border-left-color: #4caf87; }
.result-card.mod  { border-left-color: #e8a034; }
.result-card.high { border-left-color: #c4604a; }

.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    margin-bottom: 0.2rem;
}
.result-label.low  { color: #2e7d5a; }
.result-label.mod  { color: #b97a20; }
.result-label.high { color: #c4604a; }

.result-sub {
    color: #7a6f6a;
    font-size: 0.9rem;
    font-weight: 300;
}

/* Prob bars */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0.6rem 0;
}
.prob-name {
    width: 130px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #3a3535;
}
.prob-bar-bg {
    flex: 1;
    background: #f0e8e4;
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}
.prob-pct {
    width: 44px;
    text-align: right;
    font-size: 0.85rem;
    font-weight: 600;
    color: #1a1a2e;
}

/* Disclaimer */
.disclaimer {
    background: #f5eeea;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.8rem;
    color: #7a6f6a;
    margin-top: 2rem;
    line-height: 1.6;
}

/* Submit button */
div.stButton > button {
    background: #1a1a2e;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    cursor: pointer;
    width: 100%;
    margin-top: 1rem;
    transition: background 0.2s;
}
div.stButton > button:hover {
    background: #c4604a;
}

/* Number inputs */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    font-size: 0.88rem;
    font-weight: 500;
    color: #3a3535;
}
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
MODELS_DIR = "models"

@st.cache_resource
def load_models():
    try:
        pcos_model  = joblib.load(os.path.join(MODELS_DIR, "pcos_model.pkl"))
        pcos_scaler = joblib.load(os.path.join(MODELS_DIR, "pcos_scaler.pkl"))
        pcos_cols   = joblib.load(os.path.join(MODELS_DIR, "pcos_feature_cols.pkl"))
        endo_model  = joblib.load(os.path.join(MODELS_DIR, "endo_model.pkl"))
        endo_scaler = joblib.load(os.path.join(MODELS_DIR, "endo_scaler.pkl"))
        endo_cols   = joblib.load(os.path.join(MODELS_DIR, "endo_feature_cols.pkl"))
        return pcos_model, pcos_scaler, pcos_cols, endo_model, endo_scaler, endo_cols, None
    except FileNotFoundError as e:
        return None, None, None, None, None, None, str(e)

pcos_model, pcos_scaler, pcos_cols, endo_model, endo_scaler, endo_cols, load_err = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">FemTriage</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-assisted triage for PCOS & endometriosis · Harvard HSIL Hackathon 2026</div>', unsafe_allow_html=True)

if load_err:
    st.error(f"⚠️ Could not load models: `{load_err}`\n\nMake sure your `models/` folder is committed to the repo with all six `.pkl` files.")
    st.stop()

# ── Thresholds ────────────────────────────────────────────────────────────────
LOW_THRESHOLD  = 0.35
HIGH_THRESHOLD = 0.65

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("patient_form"):

    # — Demographics
    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", min_value=10, max_value=60, value=28)
    bmi = c2.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)

    # — Hormonal labs
    st.markdown('<div class="section-label">Hormonal Labs</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    fsh  = c1.number_input("FSH (mIU/mL)",  min_value=0.0, max_value=50.0, value=5.5, step=0.1)
    lh   = c2.number_input("LH (mIU/mL)",   min_value=0.0, max_value=80.0, value=5.0, step=0.1)
    amh  = c3.number_input("AMH (ng/mL)",   min_value=0.0, max_value=20.0, value=2.5, step=0.1)
    c1, c2, c3 = st.columns(3)
    tsh  = c1.number_input("TSH (mIU/L)",   min_value=0.0, max_value=20.0, value=2.5, step=0.1)
    prl  = c2.number_input("PRL (ng/mL)",   min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    vitd = c3.number_input("Vit D3 (ng/mL)",min_value=0.0, max_value=150.0, value=30.0, step=0.1)

    # — Morphology
    st.markdown('<div class="section-label">Morphology & Cycle</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    whr      = c1.number_input("Waist:Hip Ratio", min_value=0.5, max_value=1.5, value=0.85, step=0.01)
    foll_l   = c2.number_input("Follicle No. (L)", min_value=0, max_value=30, value=6)
    foll_r   = c3.number_input("Follicle No. (R)", min_value=0, max_value=30, value=6)
    c1, c2 = st.columns(2)
    cycle_ri = c1.selectbox("Cycle Type", options=[2, 4], format_func=lambda x: "Regular (2)" if x == 2 else "Irregular (4)")
    cycle_len= c2.number_input("Cycle Length (days)", min_value=1, max_value=90, value=28)

    # — Symptoms
    st.markdown('<div class="section-label">Symptoms</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    wt_gain  = c1.radio("Weight Gain",      [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=False)
    hair_grw = c2.radio("Hair Growth",      [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=False)
    skin_drk = c3.radio("Skin Darkening",   [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=False)
    hair_los = c4.radio("Hair Loss",        [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=False)
    pimples  = c5.radio("Pimples",          [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=False)

    c1, c2 = st.columns(2)
    fast_food = c1.radio("Fast Food (regular)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    exercise  = c2.radio("Regular Exercise",    [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

    # — Endometriosis-specific
    st.markdown('<div class="section-label">Endometriosis Indicators</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    men_irr  = c1.radio("Menstrual Irregularity",      [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    infert   = c2.radio("Infertility",                 [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
    pain_lvl = c3.number_input("Chronic Pain Level (0–10)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    hormone_abn = st.radio("Hormone Level Abnormality", [0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)

    submitted = st.form_submit_button("Run Triage Assessment")

# ── Inference ─────────────────────────────────────────────────────────────────
if submitted:
    # PCOS
    lh_fsh_ratio  = lh / (fsh + 1e-6)
    total_follicles = foll_l + foll_r

    pcos_input_map = {
        "Age (yrs)":           age,
        "BMI":                 bmi,
        "FSH(mIU/mL)":        fsh,
        "LH(mIU/mL)":         lh,
        "AMH(ng/mL)":         amh,
        "TSH (mIU/L)":        tsh,
        "PRL(ng/mL)":         prl,
        "Vit D3 (ng/mL)":     vitd,
        "Waist:Hip Ratio":    whr,
        "Follicle No. (L)":   foll_l,
        "Follicle No. (R)":   foll_r,
        "Cycle(R/I)":         cycle_ri,
        "Cycle length(days)": cycle_len,
        "Weight gain(Y/N)":   wt_gain,
        "hair growth(Y/N)":   hair_grw,
        "Skin darkening (Y/N)": skin_drk,
        "Hair loss(Y/N)":     hair_los,
        "Pimples(Y/N)":       pimples,
        "Fast food (Y/N)":    fast_food,
        "Reg.Exercise(Y/N)":  exercise,
        "LH_FSH_ratio":       lh_fsh_ratio,
        "Total_Follicles":    total_follicles,
    }
    pcos_x = np.array([[pcos_input_map[f] for f in pcos_cols]])
    pcos_prob = pcos_model.predict_proba(pcos_scaler.transform(pcos_x))[0][1]

    # Endo
    pain_hormone = pain_lvl * hormone_abn
    endo_input_map = {
        "Age":                      age,
        "BMI":                      bmi,
        "Menstrual_Irregularity":   men_irr,
        "Chronic_Pain_Level":       pain_lvl,
        "Hormone_Level_Abnormality":hormone_abn,
        "Infertility":              infert,
        "Pain_Hormone":             pain_hormone,
    }
    endo_x = np.array([[endo_input_map[f] for f in endo_cols]])
    endo_prob = endo_model.predict_proba(endo_scaler.transform(endo_x))[0][1]

    max_prob = max(pcos_prob, endo_prob)
    if max_prob < LOW_THRESHOLD:
        level, css, advice = "Low Risk", "low", "Routine follow-up. Reassess if symptoms change."
    elif max_prob < HIGH_THRESHOLD:
        level, css, advice = "Moderate Risk", "mod", "Further lab workup and specialist consultation recommended."
    else:
        level, css, advice = "High Risk — Refer Now", "high", "Urgent gynaecological referral indicated. Do not delay."

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div class="result-card {css}">
        <div class="result-label {css}">{level}</div>
        <div class="result-sub">{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    def prob_bar(name, prob, color):
        pct = int(prob * 100)
        return f"""
        <div class="prob-row">
            <div class="prob-name">{name}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <div class="prob-pct">{pct}%</div>
        </div>"""

    st.markdown(
        prob_bar("PCOS", pcos_prob, "#c4604a") +
        prob_bar("Endometriosis", endo_prob, "#7b5ea7"),
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="disclaimer">
    ⚕️ <strong>Clinical Disclaimer:</strong> FemTriage is a decision-support tool only.
    Results are not a diagnosis. All triage outputs must be reviewed and confirmed by a qualified
    clinician before any action is taken. This tool does not replace clinical judgment.
    </div>
    """, unsafe_allow_html=True)
