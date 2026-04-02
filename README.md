# FemTriage — AI Clinical Triage for PCOS & Endometriosis

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-orange?style=flat-square&logo=scikit-learn&logoColor=white)
![PCOS AUC](https://img.shields.io/badge/PCOS_AUC-0.957-brightgreen?style=flat-square)
![Endo AUC](https://img.shields.io/badge/Endo_AUC-0.640-yellow?style=flat-square)
![Harvard HSIL](https://img.shields.io/badge/Harvard_HSIL-Hackathon_2026-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**FemTriage** is an AI-assisted clinical decision support tool built for the **Harvard Health Sciences Innovation Lab (HSIL) Hackathon, April 10–11 2026**. It uses real machine learning models trained on clinical datasets to generate explainable risk scores for **Polycystic Ovary Syndrome (PCOS)** and **Endometriosis** — two of the most chronically underdiagnosed conditions in women's health.

> Average time to endometriosis diagnosis: **7–10 years.** PCOS affects **1 in 10** women of reproductive age. FemTriage is built to help close that gap.

---

## ✨ Features

- Dual-condition risk scoring — PCOS and Endometriosis in a single assessment
- Real **GradientBoosting classifiers** trained on clinical data (not rule-based heuristics)
- **Explainability** via feature importance — shows clinicians which factors drove the score
- Clinically safe **35/65 triage thresholds** (Low / Moderate / High risk)
- Engineered features: `LH_FSH_ratio`, `Total_Follicles`, `Pain_Hormone` interaction
- Standalone **offline-capable HTML demo** — full GBM inference in-browser, zero server needed
- Exportable **.pkl model files** ready for Streamlit or FastAPI

---

## 📊 Model Performance

| Condition | Algorithm | Dataset | AUC-ROC | F1 Score | CV |
|---|---|---|---|---|---|
| PCOS | GradientBoostingClassifier | 541 patients | **0.957** | **0.842** | Stratified 5-fold |
| Endometriosis | GradientBoostingClassifier | 10,000 patients | **0.640** | **0.432** | Stratified 5-fold |

---

## 📁 Repository Structure

```
FemTriage/
├── data/
│   ├── PCOS.csv                            # Clinical PCOS dataset (541 patients)
│   └── structured_endometriosis_data.csv   # Endometriosis dataset (10,000 patients)
├── models/
│   ├── pcos_model.pkl                      # Trained GBM — PCOS
│   ├── pcos_scaler.pkl                     # StandardScaler for PCOS inputs
│   ├── pcos_feature_cols.pkl               # Ordered feature list (required for inference)
│   ├── endo_model.pkl                      # Trained GBM — Endometriosis
│   ├── endo_scaler.pkl                     # StandardScaler for endo inputs
│   └── endo_feature_cols.pkl               # Ordered feature list (required for inference)
├── femtriage.html                          # Standalone offline demo (no server needed)
├── train.py                                # Model training + evaluation script
├── app.py                                  # Streamlit web app
├── requirements.txt                        # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Option A — Run app directly (models already trained)

```bash
# 1. Clone the repo
git clone https://github.com/fatimaasif1508-code/FemTriage.git
cd FemTriage

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

> The `models/` folder with all six `.pkl` files is included in the repo.
> No retraining needed unless you want to modify the models.

### Option B — Retrain models from scratch

```bash
# After cloning and installing dependencies:

# Make sure data files are in place
ls data/
# PCOS.csv   structured_endometriosis_data.csv

# Run training (takes ~30 seconds, no GPU needed)
python train.py

# Then launch the app
streamlit run app.py
```

---

## ⚠️ Troubleshooting — "No such file or directory: models/pcos_model.pkl"

This error means the `models/` folder was not cloned correctly or `.pkl` files were blocked by `.gitignore`. Fix it with one of these steps:

**Step 1 — Regenerate models by running train.py:**
```bash
pip install -r requirements.txt
python train.py
```
This will recreate the full `models/` folder with all 6 `.pkl` files.

**Step 2 — Or, if you are the repo owner and models are missing from GitHub:**
```bash
# Check if .gitignore is blocking pkl files
cat .gitignore
# If you see *.pkl — remove or comment out that line, then:

git add models/
git commit -m "Add trained model pkl files"
git push origin main
```

**Step 3 — Verify models exist before running the app:**
```bash
ls models/
# Expected output:
# endo_feature_cols.pkl  endo_model.pkl  endo_scaler.pkl
# pcos_feature_cols.pkl  pcos_model.pkl  pcos_scaler.pkl
```

---

## 🎥 Demo

- **Offline HTML demo** — open `femtriage.html` directly in any browser. Full GBM inference runs client-side with no backend or internet connection required. Perfect for live demos.
- **Streamlit app** — run `streamlit run app.py` for the full interactive clinical interface.

### Demo Patients (preloaded in the app)

| Patient | PCOS Risk | Endo Risk | Triage |
|---|---|---|---|
| Patient A — High PCOS | **99.8%** | 29.1% | Immediate referral |
| Patient B — High Endo | 0.2% | **73.9%** | Immediate referral |
| Patient C — Borderline | 5.4% | 28.0% | Routine follow-up |

---

## 🔬 Feature Engineering

Three new features are engineered before training — these are the most clinically significant signals:

| Feature | Formula | Model Importance | Clinical Basis |
|---|---|---|---|
| `LH_FSH_ratio` | LH ÷ FSH | 3.65% (PCOS) | Rotterdam Criteria diagnostic marker |
| `Total_Follicles` | Follicles_L + Follicles_R | **38.8% (PCOS)** | #1 PCOS predictor — antral follicle count |
| `Pain_Hormone` | Chronic_Pain_Level × Hormone_Abnormality | **31.0% (Endo)** | #1 Endo predictor — interaction signal |

---

## 💻 Using the Models in Your Own Code

```python
import joblib
import numpy as np

# Load artifacts
pcos_model  = joblib.load('models/pcos_model.pkl')
pcos_scaler = joblib.load('models/pcos_scaler.pkl')
pcos_cols   = joblib.load('models/pcos_feature_cols.pkl')

endo_model  = joblib.load('models/endo_model.pkl')
endo_scaler = joblib.load('models/endo_scaler.pkl')
endo_cols   = joblib.load('models/endo_feature_cols.pkl')

# Engineer features first — always required
patient['LH_FSH_ratio']   = patient['LH(mIU/mL)'] / patient['FSH(mIU/mL)']
patient['Total_Follicles'] = patient['Follicle No. (L)'] + patient['Follicle No. (R)']
patient['Pain_Hormone']    = patient['Chronic_Pain_Level'] * patient['Hormone_Level_Abnormality']

# Predict — always use predict_proba, not predict
X_pcos = np.array([[patient[f] for f in pcos_cols]])
pcos_prob = pcos_model.predict_proba(pcos_scaler.transform(X_pcos))[0][1]

X_endo = np.array([[patient[f] for f in endo_cols]])
endo_prob = endo_model.predict_proba(endo_scaler.transform(X_endo))[0][1]

print(f"PCOS risk:  {pcos_prob * 100:.1f}%")
print(f"Endo risk:  {endo_prob * 100:.1f}%")

# Triage bands (35/65 thresholds)
def triage(prob):
    if prob < 0.35: return "Low — routine follow-up"
    if prob < 0.65: return "Moderate — further testing advised"
    return "High — immediate specialist referral"

print(triage(pcos_prob))
print(triage(endo_prob))
```

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Modeling | scikit-learn `GradientBoostingClassifier` (200 estimators, depth 4) |
| Data prep | pandas, NumPy, IQR outlier capping |
| Validation | Stratified 5-fold CV · AUC-ROC · F1 score |
| App | Streamlit · standalone HTML (offline) |
| Explainability | Feature importance · SHAP waterfall (roadmap) |
| Persistence | joblib · pickle · column order preserved via `.pkl` |

---

## 📦 requirements.txt

```
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
joblib>=1.3.0
shap>=0.44.0
streamlit>=1.32.0
matplotlib>=3.7.0
plotly>=5.18.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

FemTriage is a **research prototype** built for a hackathon. It is **not** a medical device and has not been clinically validated for deployment. Risk scores are triage indicators only — not diagnoses. Real clinical use would require multi-centre data, external validation, and regulatory approval (e.g. FDA, CE mark). The endometriosis dataset is structured/simulated; a real deployment would require clinical partnership for richer longitudinal data.

---

## 🗺 Roadmap

- [ ] SHAP local waterfall plots per patient
- [ ] FastAPI backend for REST inference
- [ ] Uterine fibroids and PMDD as additional conditions
- [ ] Clinical partnership for richer endometriosis data
- [ ] Multi-centre external validation study

---

## 👩‍💻 Authors

Built with ♥ for **Harvard HSIL Hackathon 2026**

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
