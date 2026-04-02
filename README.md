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
│   ├── PCOS.csv
│   └── structured_endometriosis_data.csv
│   ├── pcos_model.pkl
│   ├── pcos_scaler.pkl
│   ├── pcos_feature_cols.pkl
│   ├── endo_model.pkl
│   ├── endo_scaler.pkl
│   └── endo_feature_cols.pkl
├── femtriage.html
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Dataset Note

The CSV datasets are required to retrain the models. Ensure this structure exists before running `train.py`:

```
data/
├── PCOS.csv
└── structured_endometriosis_data.csv
```

If datasets are missing, training will fail with `FileNotFoundError`.

👉 The pre-trained `.pkl` files in `models/` are included in the repository — you can run the Streamlit app directly without retraining.

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/fatimaasif1508-code/FemTriage.git
cd FemTriage

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain models — requires data/ CSVs
python train.py

# Run the Streamlit app
streamlit run app.py
```

---

## 🎥 Demo

- **Offline HTML demo** — open `femtriage.html` in any browser. Full GBM inference runs client-side with no backend or internet required.
- **Streamlit app** — run `streamlit run app.py` for the full interactive interface (requires Python environment).

---

## 🔬 Feature Engineering

| Feature | Formula | Importance | Clinical basis |
|---|---|---|---|
| `LH_FSH_ratio` | LH ÷ FSH | 3.65% (PCOS) | Rotterdam Criteria marker |
| `Total_Follicles` | Follicles_L + Follicles_R | **38.8% (PCOS)** | #1 PCOS predictor — antral follicle count |
| `Pain_Hormone` | Pain_Level × Hormone_Abnormality | **31.0% (Endo)** | #1 Endo predictor — interaction signal |

---

## 💻 Using the Models

```python
import joblib
import numpy as np

pcos_model  = joblib.load('models/pcos_model.pkl')
pcos_scaler = joblib.load('models/pcos_scaler.pkl')
pcos_cols   = joblib.load('models/pcos_feature_cols.pkl')

patient['LH_FSH_ratio']   = patient['LH(mIU/mL)'] / patient['FSH(mIU/mL)']
patient['Total_Follicles'] = patient['Follicle No. (L)'] + patient['Follicle No. (R)']

X = np.array([[patient[f] for f in pcos_cols]])
pcos_prob = pcos_model.predict_proba(pcos_scaler.transform(X))[0][1]
```

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Modeling | scikit-learn GradientBoostingClassifier |
| Data | pandas, NumPy, IQR outlier capping |
| Validation | Stratified 5-fold CV, AUC-ROC, F1 |
| App | Streamlit, standalone HTML demo |
| Explainability | Feature importance, SHAP (roadmap) |
| Saving | joblib/pickle, column order preserved |

---

## ⚠️ Disclaimer

FemTriage is a **research prototype** built for a hackathon. It is **not** a medical device and has not been clinically validated for deployment. Risk scores are triage indicators only — not diagnoses. Real clinical use would require multi-centre validation, external testing, and regulatory approval.

---

## 🗺 Roadmap

- [ ] SHAP local waterfall plots per patient
- [ ] FastAPI backend for REST inference
- [ ] Uterine fibroids and PMDD as additional conditions
- [ ] Clinical partnership for richer endometriosis data
- [ ] Multi-centre external validation study

---

Built with ♥ for **Harvard HSIL Hackathon 2026** · MIT License
