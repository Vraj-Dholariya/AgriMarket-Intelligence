"""
AgriMarket Intelligence — Data Pipeline
========================================
Dataset: Crop Yield Prediction (Soil + Climate + Farm Inputs)
Columns: N, P, K, Soil_pH, Soil_Moisture, Soil_Type, Organic_Carbon,
         Temperature, Humidity, Rainfall, Sunlight_Hours, Wind_Speed,
         Region, Altitude, Season, Crop_Type, Irrigation_Type,
         Fertilizer_Used, Pesticide_Used, Crop_Yield_ton_per_hectare
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "N", "P", "K", "Soil_pH", "Soil_Moisture", "Organic_Carbon",
    "Temperature", "Humidity", "Rainfall", "Sunlight_Hours",
    "Wind_Speed", "Altitude", "Fertilizer_Used", "Pesticide_Used",
    "Crop_Yield_ton_per_hectare",
]
CATEGORICAL_COLS = ["Soil_Type", "Region", "Season", "Crop_Type", "Irrigation_Type"]
TARGET_COL = "Crop_Yield_ton_per_hectare"


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_raw_data(csv_path: str = "data/raw_agmarknet.csv") -> pd.DataFrame:
    """Load the raw crop-yield dataset and normalise column names."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. "
            "Please place raw_agmarknet.csv in the data/ folder."
        )
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    print(f"[load] Raw shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset: duplicates, types, missing values, outliers."""
    cleaned = df.copy()

    # --- Remove duplicates ---
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    print(f"[clean] Removed {before - len(cleaned)} duplicate rows.")

    # --- Strip whitespace from categoricals ---
    for col in CATEGORICAL_COLS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip().str.title()

    # --- Coerce numeric columns ---
    for col in NUMERIC_COLS:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # --- Replace infinities ---
    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- Missing value report ---
    missing = cleaned.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("[clean] Missing values detected:")
        print(missing)
    else:
        print("[clean] No missing values found.")

    # --- Impute numeric with median grouped by Crop_Type ---
    for col in NUMERIC_COLS:
        if col not in cleaned.columns or col == TARGET_COL:
            continue
        if cleaned[col].isnull().any():
            if "Crop_Type" in cleaned.columns:
                group_median = cleaned.groupby("Crop_Type")[col].transform("median")
                cleaned[col] = cleaned[col].fillna(group_median)
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # --- Impute categoricals with mode ---
    for col in CATEGORICAL_COLS:
        if col in cleaned.columns and cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    print(f"[clean] Cleaned shape: {cleaned.shape}")
    return cleaned


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive useful composite features."""
    df = df.copy()

    # Nutrient balance index
    df["NPK_Total"] = df["N"] + df["P"] + df["K"]
    df["NP_Ratio"] = df["N"] / (df["P"] + 1)  # +1 to avoid div-by-zero

    # Climate composite
    df["Heat_Moisture_Index"] = df["Temperature"] * df["Humidity"] / 100
    df["Rainfall_per_SunHour"] = df["Rainfall"] / (df["Sunlight_Hours"] + 0.1)

    # Encode categoricals as integer labels (for correlation / modelling)
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[f"{col}_Code"] = le.fit_transform(df[col].astype(str))

    print(f"[feature] Feature-engineered shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. Statistical Analysis
# ---------------------------------------------------------------------------
def pearson_correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for key numeric features."""
    corr_cols = [c for c in NUMERIC_COLS if c in df.columns]
    corr_df = df[corr_cols].corr(method="pearson")
    print("\n[stats] Pearson Correlation Matrix (key columns):")
    print(corr_df[[TARGET_COL]].sort_values(TARGET_COL, ascending=False))
    return corr_df


def chi_square_test(df: pd.DataFrame, col_a: str = "Season", col_b: str = "Crop_Type") -> dict:
    """Chi-square test of independence between two categorical columns."""
    contingency = pd.crosstab(df[col_a].astype(str), df[col_b].astype(str))
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    result = {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": dof,
        "interpretation": (
            "REJECT H₀ — Season and Crop_Type are NOT independent (significant association)"
            if p_value < 0.05
            else "FAIL TO REJECT H₀ — Season and Crop_Type appear independent"
        ),
    }
    print(f"\n[stats] Chi-Square Test ({col_a} vs {col_b}):")
    for k, v in result.items():
        if k not in ("contingency_table", "expected_table"):
            print(f"  {k}: {v}")
    return result


# ---------------------------------------------------------------------------
# 5. Machine Learning — Random Forest
# ---------------------------------------------------------------------------
def prepare_model_data(df: pd.DataFrame) -> tuple:
    """Return (X, y) ready for sklearn, using one-hot encoded categoricals."""
    model_df = df.copy()

    # Use the label-coded columns already added by engineer_features
    feature_candidates = (
        [c for c in NUMERIC_COLS if c in model_df.columns and c != TARGET_COL]
        + [f"{c}_Code" for c in CATEGORICAL_COLS if f"{c}_Code" in model_df.columns]
        + ["NPK_Total", "NP_Ratio", "Heat_Moisture_Index", "Rainfall_per_SunHour"]
    )
    feature_candidates = [c for c in feature_candidates if c in model_df.columns]

    X = model_df[feature_candidates].fillna(model_df[feature_candidates].median())
    y = model_df[TARGET_COL].fillna(model_df[TARGET_COL].median())
    return X, y


def train_random_forest_model(df: pd.DataFrame) -> tuple:
    """Train a Random Forest Regressor and return metrics + feature importances."""
    X, y = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    feature_importance = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )

    metrics = {
        "target": TARGET_COL,
        "rows_used": len(X),
        "test_rows": len(X_test),
        "MAE": round(float(mae), 6),
        "RMSE": round(float(rmse), 6),
        "R2": round(float(r2), 6),
        "random_state": 42,
    }

    print("\n[model] Random Forest — Yield Prediction")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²  : {r2:.4f}")
    print("\n  Top 10 Feature Importances:")
    print(feature_importance.head(10).to_string())

    return metrics, feature_importance, rf


# ---------------------------------------------------------------------------
# 6. Export helpers
# ---------------------------------------------------------------------------
def save_outputs(
    cleaned_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    chi_result: dict,
    metrics: dict,
    feature_importance: pd.Series,
    output_csv: str = "data/cleaned_agriculture_data.csv",
    output_xlsx: str = "data/cleaned_agriculture_data.xlsx",
):
    """Save all analysis outputs to disk."""
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_csv, index=False)
    print(f"[export] Cleaned CSV → {out_csv}")

    # Excel (for Tableau)
    out_xlsx = Path(output_xlsx)
    cleaned_df.to_excel(out_xlsx, index=False)
    print(f"[export] Excel for Tableau → {out_xlsx}")

    # Correlation matrix
    corr_path = Path("reports/correlation_matrix.csv")
    corr_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(corr_path)
    print(f"[export] Correlation matrix → {corr_path}")

    # Feature importance
    fi_path = Path("reports/feature_importance.csv")
    feature_importance.reset_index().rename(
        columns={"index": "Feature", 0: "Importance"}
    ).to_csv(fi_path, index=False)
    print(f"[export] Feature importance → {fi_path}")

    # Model metrics
    mm_path = Path("reports/model_metrics.json")
    with open(mm_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[export] Model metrics → {mm_path}")

    # Analysis summary (for reporting)
    summary = {
        "rows_after_cleaning": len(cleaned_df),
        "columns": list(cleaned_df.columns),
        "numeric_summary": cleaned_df[NUMERIC_COLS].describe().round(3).to_dict(),
        "season_crop_chi_square": {
            k: v for k, v in chi_result.items()
            if k not in ("contingency_table", "expected_table")
        },
        "model_r2": metrics["R2"],
        "model_rmse": metrics["RMSE"],
    }
    sm_path = Path("reports/analysis_summary.json")
    with open(sm_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[export] Analysis summary → {sm_path}")


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    csv_path: str = "data/raw_agmarknet.csv",
    output_csv: str = "data/cleaned_agriculture_data.csv",
    output_xlsx: str = "data/cleaned_agriculture_data.xlsx",
) -> pd.DataFrame:
    """Execute the complete data pipeline end-to-end."""
    print("=" * 60)
    print(" AgriMarket Intelligence — Full Data Pipeline")
    print("=" * 60)

    raw_df = load_raw_data(csv_path)
    cleaned_df = clean_dataset(raw_df)
    cleaned_df = engineer_features(cleaned_df)

    corr_df = pearson_correlation_analysis(cleaned_df)
    chi_result = chi_square_test(cleaned_df, "Season", "Crop_Type")
    metrics, feature_importance, _ = train_random_forest_model(cleaned_df)

    save_outputs(cleaned_df, corr_df, chi_result, metrics, feature_importance,
                 output_csv, output_xlsx)

    print("\n" + "=" * 60)
    print(" Pipeline complete!")
    print(f" Rows: {len(cleaned_df)}  |  Columns: {len(cleaned_df.columns)}")
    print(" Next: open notebooks/01_dav_analysis.ipynb")
    print("=" * 60)
    return cleaned_df


if __name__ == "__main__":
    run_pipeline()
