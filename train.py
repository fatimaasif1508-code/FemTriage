"""
FemTriage — Model Training Script
Harvard HSIL Hackathon 2026

Trains two GradientBoosting classifiers:
  1. PCOS model         (541 clinical patients)
  2. Endometriosis model (10,000 structured patients)

Outputs (saved to models/):
  pcos_model.pkl          Trained GBM classifier
  pcos_scaler.pkl         StandardScaler fitted on training data
  pcos_feature_cols.pkl   Ordered feature list — MUST match at inference time

  endo_model.pkl
  endo_scaler.pkl
  endo_feature_cols.pkl

Usage:
  python train.py
  python train.py --pcos-data data/PCOS.csv --endo-data data/structured_endometriosis_data.csv
"""

import argparse
import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_PCOS_PATH = "data/PCOS.csv"
DEFAULT_ENDO_PATH = "data/structured_endometriosis_data.csv"

RANDOM_STATE = 42
CV_FOLDS = 5

# Triage thresholds (35/65 — clinically conservative)
LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

PCOS_BASE_FEATURES = [
    "Age (yrs)",
    "BMI",
    "FSH(mIU/mL)",
    "LH(mIU/mL)",
    "AMH(ng/mL)",
    "TSH (mIU/L)",
    "PRL(ng/mL)",
    "Vit D3 (ng/mL)",
    "Waist:Hip Ratio",
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
]

ENDO_BASE_FEATURES = [
    "Age",
    "BMI",
    "Menstrual_Irregularity",
    "Chronic_Pain_Level",
    "Hormone_Level_Abnormality",
    "Infertility",
]

OUTLIER_CAP_COLS = ["FSH(mIU/mL)", "LH(mIU/mL)", "AMH(ng/mL)", "PRL(ng/mL)", "TSH (mIU/L)"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def iqr_cap(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Cap outliers using IQR method — preserves all rows."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df


def engineer_pcos_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add LH_FSH_ratio and Total_Follicles engineered features."""
    df = df.copy()
    df["LH_FSH_ratio"] = df["LH(mIU/mL)"] / (df["FSH(mIU/mL)"] + 1e-6)
    df["Total_Follicles"] = df["Follicle No. (L)"] + df["Follicle No. (R)"]
    return df


def engineer_endo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Pain_Hormone interaction term."""
    df = df.copy()
    df["Pain_Hormone"] = df["Chronic_Pain_Level"] * df["Hormone_Level_Abnormality"]
    return df


def evaluate(model, X, y, scaler, label: str) -> dict:
    """Run stratified CV and print full metrics report."""
    X_s = scaler.transform(X)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    auc_scores = cross_val_score(model, X_s, y, cv=cv, scoring="roc_auc")
    f1_scores  = cross_val_score(model, X_s, y, cv=cv, scoring="f1")
    acc_scores = cross_val_score(model, X_s, y, cv=cv, scoring="accuracy")

    print(f"\n  {label} — Stratified {CV_FOLDS}-fold Cross-Validation")
    print(f"    AUC-ROC : {auc_scores.mean():.3f}  (±{auc_scores.std():.3f})")
    print(f"    F1      : {f1_scores.mean():.3f}  (±{f1_scores.std():.3f})")
    print(f"    Accuracy: {acc_scores.mean():.3f}  (±{acc_scores.std():.3f})")

    y_pred = model.predict(X_s)
    report = classification_report(y, y_pred, target_names=["Negative", "Positive"])
    print(f"\n  Classification report ({label}):")
    for line in report.splitlines():
        print("    " + line)

    return {
        "auc": round(auc_scores.mean(), 3),
        "f1":  round(f1_scores.mean(), 3),
        "acc": round(acc_scores.mean(), 3),
    }


def feature_importance_report(model, feature_names: list, label: str, top_n: int = 8) -> None:
    """Print ranked feature importances."""
    pairs = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: -x[1],
    )
    print(f"\n  Top {top_n} features — {label}:")
    for name, imp in pairs[:top_n]:
        bar = "█" * int(imp * 200)
        print(f"    {name:<35} {imp:.4f}  {bar}")


def save_artifacts(model, scaler, feature_cols: list, prefix: str) -> None:
    """Save model, scaler, and feature column list to models/."""
    model_path = os.path.join(MODELS_DIR, f"{prefix}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{prefix}_scaler.pkl")
    cols_path   = os.path.join(MODELS_DIR, f"{prefix}_feature_cols.pkl")

    joblib.dump(model,        model_path)
    joblib.dump(scaler,       scaler_path)
    joblib.dump(feature_cols, cols_path)

    print(f"\n  Saved:")
    for path in [model_path, scaler_path, cols_path]:
        size_kb = os.path.getsize(path) // 1024
        print(f"    {path}  ({size_kb} KB)")


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_pcos(data_path: str, tune_hyperparams: bool = False) -> dict:
    print_section("PCOS Model Training")

    # Load & clean
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Target distribution: {df['PCOS (Y/N)'].value_counts().to_dict()}")

    # Prepare features
    df = df[PCOS_BASE_FEATURES + ["PCOS (Y/N)"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    print(f"  After dropna: {df.shape[0]} rows")

    df = iqr_cap(df, OUTLIER_CAP_COLS)
    df = engineer_pcos_features(df)

    final_features = PCOS_BASE_FEATURES + ["LH_FSH_ratio", "Total_Follicles"]
    X = df[final_features].values
    y = df["PCOS (Y/N)"].values

    # Scale
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Train
    if tune_hyperparams:
        print("\n  Running GridSearchCV (this may take a few minutes)...")
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1],
        }
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            GradientBoostingClassifier(subsample=0.8, random_state=RANDOM_STATE),
            param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1,
        )
        gs.fit(X_s, y)
        model = gs.best_estimator_
        print(f"  Best params: {gs.best_params_}")
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        )
        model.fit(X_s, y)

    metrics = evaluate(model, X, y, scaler, "PCOS")
    feature_importance_report(model, final_features, "PCOS")
    save_artifacts(model, scaler, final_features, "pcos")

    return metrics


def train_endo(data_path: str, tune_hyperparams: bool = False) -> dict:
    print_section("Endometriosis Model Training")

    # Load & clean
    df = pd.read_csv(data_path)
    print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Target distribution: {df['Diagnosis'].value_counts().to_dict()}")

    df = df[ENDO_BASE_FEATURES + ["Diagnosis"]].dropna()
    df = engineer_endo_features(df)

    final_features = ENDO_BASE_FEATURES + ["Pain_Hormone"]
    X = df[final_features].values
    y = df["Diagnosis"].values

    # Scale
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Train
    if tune_hyperparams:
        print("\n  Running GridSearchCV...")
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1],
        }
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            GradientBoostingClassifier(subsample=0.8, random_state=RANDOM_STATE),
            param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1,
        )
        gs.fit(X_s, y)
        model = gs.best_estimator_
        print(f"  Best params: {gs.best_params_}")
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE,
        )
        model.fit(X_s, y)

    metrics = evaluate(model, X, y, scaler, "Endometriosis")
    feature_importance_report(model, final_features, "Endometriosis")
    save_artifacts(model, scaler, final_features, "endo")

    return metrics


# ---------------------------------------------------------------------------
# Quick inference test
# ---------------------------------------------------------------------------

def sanity_check() -> None:
    print_section("Sanity Check — Demo Patients")

    pcos_model  = joblib.load(os.path.join(MODELS_DIR, "pcos_model.pkl"))
    pcos_scaler = joblib.load(os.path.join(MODELS_DIR, "pcos_scaler.pkl"))
    pcos_cols   = joblib.load(os.path.join(MODELS_DIR, "pcos_feature_cols.pkl"))

    endo_model  = joblib.load(os.path.join(MODELS_DIR, "endo_model.pkl"))
    endo_scaler = joblib.load(os.path.join(MODELS_DIR, "endo_scaler.pkl"))
    endo_cols   = joblib.load(os.path.join(MODELS_DIR, "endo_feature_cols.pkl"))

    demo_patients = {
        "Patient A — High PCOS": {
            "Age (yrs)": 29, "BMI": 27.5, "FSH(mIU/mL)": 5.8, "LH(mIU/mL)": 9.2,
            "AMH(ng/mL)": 3.66, "TSH (mIU/L)": 4.3, "PRL(ng/mL)": 14.4,
            "Vit D3 (ng/mL)": 52.3, "Waist:Hip Ratio": 0.91,
            "Follicle No. (L)": 13, "Follicle No. (R)": 14,
            "Cycle(R/I)": 4, "Cycle length(days)": 5,
            "Weight gain(Y/N)": 1, "hair growth(Y/N)": 1, "Skin darkening (Y/N)": 1,
            "Hair loss(Y/N)": 1, "Pimples(Y/N)": 1, "Fast food (Y/N)": 1,
            "Reg.Exercise(Y/N)": 0, "LH_FSH_ratio": 9.2 / 5.8, "Total_Follicles": 27,
            # endo
            "Age": 29, "BMI_e": 27.5, "Menstrual_Irregularity": 1,
            "Chronic_Pain_Level": 2.0, "Hormone_Level_Abnormality": 0, "Infertility": 0,
        },
        "Patient B — High Endo": {
            "Age (yrs)": 33, "BMI": 22.6, "FSH(mIU/mL)": 6.1, "LH(mIU/mL)": 4.2,
            "AMH(ng/mL)": 2.1, "TSH (mIU/L)": 2.0, "PRL(ng/mL)": 22.0,
            "Vit D3 (ng/mL)": 21.5, "Waist:Hip Ratio": 0.82,
            "Follicle No. (L)": 4, "Follicle No. (R)": 3,
            "Cycle(R/I)": 2, "Cycle length(days)": 28,
            "Weight gain(Y/N)": 0, "hair growth(Y/N)": 0, "Skin darkening (Y/N)": 0,
            "Hair loss(Y/N)": 1, "Pimples(Y/N)": 0, "Fast food (Y/N)": 0,
            "Reg.Exercise(Y/N)": 1, "LH_FSH_ratio": 4.2 / 6.1, "Total_Follicles": 7,
            "Age": 33, "BMI_e": 22.6, "Menstrual_Irregularity": 1,
            "Chronic_Pain_Level": 8.5, "Hormone_Level_Abnormality": 1, "Infertility": 1,
        },
        "Patient C — Borderline": {
            "Age (yrs)": 30, "BMI": 25.1, "FSH(mIU/mL)": 5.0, "LH(mIU/mL)": 5.5,
            "AMH(ng/mL)": 2.4, "TSH (mIU/L)": 2.8, "PRL(ng/mL)": 18.0,
            "Vit D3 (ng/mL)": 32.0, "Waist:Hip Ratio": 0.85,
            "Follicle No. (L)": 7, "Follicle No. (R)": 8,
            "Cycle(R/I)": 4, "Cycle length(days)": 14,
            "Weight gain(Y/N)": 0, "hair growth(Y/N)": 0, "Skin darkening (Y/N)": 0,
            "Hair loss(Y/N)": 0, "Pimples(Y/N)": 1, "Fast food (Y/N)": 1,
            "Reg.Exercise(Y/N)": 0, "LH_FSH_ratio": 5.5 / 5.0, "Total_Follicles": 15,
            "Age": 30, "BMI_e": 25.1, "Menstrual_Irregularity": 1,
            "Chronic_Pain_Level": 4.5, "Hormone_Level_Abnormality": 0, "Infertility": 0,
        },
    }

    endo_key_map = {
        "Age": "Age", "BMI": "BMI_e",
        "Menstrual_Irregularity": "Menstrual_Irregularity",
        "Chronic_Pain_Level": "Chronic_Pain_Level",
        "Hormone_Level_Abnormality": "Hormone_Level_Abnormality",
        "Infertility": "Infertility",
    }

    print(f"\n  {'Patient':<30} {'PCOS':>10}  {'Endo':>10}  {'Triage'}")
    print(f"  {'─'*30} {'─'*10}  {'─'*10}  {'─'*20}")

    for name, pt in demo_patients.items():
        pcos_x = np.array([[pt[f] for f in pcos_cols]])
        pcos_prob = pcos_model.predict_proba(pcos_scaler.transform(pcos_x))[0][1]

        endo_vals = []
        for f in endo_cols:
            if f == "Pain_Hormone":
                endo_vals.append(pt["Chronic_Pain_Level"] * pt["Hormone_Level_Abnormality"])
            else:
                key = endo_key_map.get(f, f)
                endo_vals.append(pt[key])
        endo_x = np.array([endo_vals])
        endo_prob = endo_model.predict_proba(endo_scaler.transform(endo_x))[0][1]

        max_p = max(pcos_prob, endo_prob)
        triage = "Low — routine" if max_p < LOW_THRESHOLD else \
                 "Moderate — test" if max_p < HIGH_THRESHOLD else \
                 "HIGH — refer now"

        print(f"  {name:<30} {pcos_prob*100:>9.1f}%  {endo_prob*100:>9.1f}%  {triage}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FemTriage — train PCOS and endometriosis models"
    )
    parser.add_argument("--pcos-data", default=DEFAULT_PCOS_PATH,
                        help=f"Path to PCOS CSV (default: {DEFAULT_PCOS_PATH})")
    parser.add_argument("--endo-data", default=DEFAULT_ENDO_PATH,
                        help=f"Path to endo CSV (default: {DEFAULT_ENDO_PATH})")
    parser.add_argument("--tune", action="store_true",
                        help="Run GridSearchCV hyperparameter tuning (~2 min extra)")
    parser.add_argument("--skip-pcos", action="store_true", help="Skip PCOS training")
    parser.add_argument("--skip-endo", action="store_true", help="Skip endo training")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  FemTriage — Model Training")
    print("  Harvard HSIL Hackathon 2026")
    print("=" * 60)

    results = {}

    if not args.skip_pcos:
        results["pcos"] = train_pcos(args.pcos_data, tune_hyperparams=args.tune)
    else:
        print("\n  [Skipped PCOS training]")

    if not args.skip_endo:
        results["endo"] = train_endo(args.endo_data, tune_hyperparams=args.tune)
    else:
        print("\n  [Skipped Endo training]")

    sanity_check()

    print_section("Training Complete")
    if "pcos" in results:
        print(f"  PCOS  →  AUC: {results['pcos']['auc']}  |  F1: {results['pcos']['f1']}")
    if "endo" in results:
        print(f"  Endo  →  AUC: {results['endo']['auc']}  |  F1: {results['endo']['f1']}")
    print(f"\n  All artifacts saved to: {os.path.abspath(MODELS_DIR)}/")
    print("\n  To run the app:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
