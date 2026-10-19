"""AgriMarket-Intelligence data pipeline for the dataset included in this ZIP.

The included CSV contains soil, weather, crop, irrigation and yield columns.
It does NOT contain Agmarknet market prices, Area, or Production. This script
therefore analyzes the supplied data and predicts Crop_Yield_ton_per_hectare.

Run from the repository root:
    python data_pipeline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "raw_agmarknet.csv"
DEFAULT_OUTPUT = ROOT / "data" / "cleaned_agriculture_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TARGET = "Crop_Yield_ton_per_hectare"


def load_data(path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load the source CSV and normalize column labels."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPlace raw_agmarknet.csv in the project's data/ folder."
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("The input CSV has no rows.")
    if TARGET not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET}' is missing. Found columns: {list(df.columns)}"
        )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, trim text, convert numeric fields and handle missing values."""
    data = df.copy().drop_duplicates().reset_index(drop=True)
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].astype("string").str.strip()
        data[col] = data[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Convert columns that are already numeric or mostly numeric, excluding text categories.
    categorical = {"Soil_Type", "Region", "Season", "Crop_Type", "Irrigation_Type"}
    for col in data.columns:
        if col not in categorical:
            converted = pd.to_numeric(data[col], errors="coerce")
            if converted.notna().sum() > 0:
                data[col] = converted

    # A row without a target cannot train/evaluate a supervised model.
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET]).reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No valid rows remain after checking target column '{TARGET}'.")

    # Replace infinite numeric values; missing feature values are imputed inside the ML pipeline.
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return data


def run_analysis(df: pd.DataFrame) -> dict:
    """Print summary analysis and run a chi-square test when categories permit."""
    print("\n=== DATA OVERVIEW ===")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("\nMissing values by column:\n", df.isna().sum().to_string())
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        print("\n=== NUMERIC SUMMARY ===\n", numeric.describe().round(3).to_string())
        corr = numeric.corr(numeric_only=True)
        corr_path = REPORT_DIR / "correlation_matrix.csv"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        corr.to_csv(corr_path)
        print(f"\nCorrelation matrix saved to: {corr_path.relative_to(ROOT)}")

    result = {"rows_after_cleaning": int(len(df)), "columns": list(df.columns)}
    if "Season" in df.columns and "Crop_Type" in df.columns:
        table = pd.crosstab(df["Season"].fillna("Unknown"), df["Crop_Type"].fillna("Unknown"))
        if table.shape[0] > 1 and table.shape[1] > 1:
            chi2, p_value, dof, _ = chi2_contingency(table)
            result["season_crop_chi_square"] = {
                "chi2": float(chi2), "p_value": float(p_value), "degrees_of_freedom": int(dof)
            }
            print("\n=== CHI-SQUARE: SEASON VS CROP TYPE ===")
            print(f"chi-square={chi2:.3f}, p-value={p_value:.6g}, dof={dof}")
    return result


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Train a Random Forest regression pipeline and return metrics/importances."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer(transformers=[
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])
    model = Pipeline(steps=[
        ("preprocessor", prep),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "target": TARGET,
        "rows_used": int(len(df)),
        "test_rows": int(len(y_test)),
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "random_state": 42,
    }

    # Get feature importances using transformed feature names when supported.
    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
        values = model.named_steps["regressor"].feature_importances_
        importance = pd.DataFrame({"feature": names, "importance": values}).sort_values(
            "importance", ascending=False
        )
    except (AttributeError, ValueError):
        importance = pd.DataFrame(columns=["feature", "importance"])

    print("\n=== RANDOM FOREST YIELD PREDICTION ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    if not importance.empty:
        print("\nTop 10 feature importances:\n", importance.head(10).to_string(index=False))
    return model, metrics, importance


def run_pipeline(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Run cleaning, analysis, ML, and export files for Tableau/reporting."""
    raw = load_data(input_path)
    cleaned = clean_data(raw)
    analysis = run_analysis(cleaned)
    model, metrics, importance = train_model(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "crop_yield_random_forest.joblib")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (REPORT_DIR / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    if not importance.empty:
        importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    print(f"\nCleaned data saved to: {output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path}")
    print(f"Model saved to: {MODEL_DIR / 'crop_yield_random_forest.joblib'}")
    print(f"Metrics saved to: {REPORT_DIR / 'model_metrics.json'}")
    return cleaned


if __name__ == "__main__":
    run_pipeline()
