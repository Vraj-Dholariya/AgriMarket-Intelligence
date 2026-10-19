import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_raw_data(csv_path: str = "data/raw_agmarknet.csv") -> pd.DataFrame:
    """Load the raw Agmarknet dataset and normalize column names."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'. Please download the Agmarknet CSV and save it there."
        )

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw agricultural dataset and standardize values."""
    cleaned = df.copy()

    # Remove duplicate rows
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Strip whitespace from object columns
    for col in cleaned.select_dtypes(include=['object']).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # Convert numeric columns to numeric when possible
    for col in cleaned.columns:
        if col.lower() in {'area', 'production', 'modal_price', 'yield'}:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

    # Standardize common column names if necessary
    rename_map = {
        'Area (Hectares)': 'Area',
        'Production (Tonnes)': 'Production',
        'Modal Price (Rs./Quintal)': 'Modal_Price',
        'Crop': 'Crop',
        'Season': 'Season',
    }
    cleaned = cleaned.rename(columns={k: v for k, v in rename_map.items() if k in cleaned.columns})

    # Ensure required columns exist
    required = {'Area', 'Production', 'Modal_Price', 'Crop', 'Season'}
    missing = required - set(cleaned.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Replace infinite values with NaN and convert to float
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        cleaned[col] = cleaned[col].replace([np.inf, -np.inf], np.nan)

    # Fill missing numeric values using median grouped by Crop
    for col in numeric_cols:
        if col == 'Crop':
            continue
        crop_group_median = cleaned.groupby('Crop')[col].transform('median')
        cleaned[col] = cleaned[col].fillna(crop_group_median)
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer agricultural features used in analysis and modeling."""
    df = df.copy()

    # Prevent divide-by-zero issues
    df['Yield'] = np.where(df['Area'] > 0, df['Production'] / df['Area'], np.nan)
    df['Estimated_Revenue_INR'] = df['Production'] * df['Modal_Price']

    # Fill remaining yield gaps using crop-wise median
    if 'Yield' in df.columns:
        grouped_yield = df.groupby('Crop')['Yield'].transform('median')
        df['Yield'] = df['Yield'].fillna(grouped_yield)
        df['Yield'] = df['Yield'].fillna(df['Yield'].median())

    if 'Estimated_Revenue_INR' in df.columns:
        grouped_revenue = df.groupby('Crop')['Estimated_Revenue_INR'].transform('median')
        df['Estimated_Revenue_INR'] = df['Estimated_Revenue_INR'].fillna(grouped_revenue)
        df['Estimated_Revenue_INR'] = df['Estimated_Revenue_INR'].fillna(df['Estimated_Revenue_INR'].median())

    return df


def pearson_correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlations for key agricultural metrics."""
    corr_cols = ['Area', 'Production', 'Yield', 'Modal_Price']
    corr_df = df[corr_cols].corr(method='pearson')
    print("\nPearson Correlation Analysis")
    print(corr_df)
    return corr_df


def chi_square_test(df: pd.DataFrame):
    """Run a Chi-square test of independence between Season and Crop."""
    contingency = pd.crosstab(df['Season'].astype(str), df['Crop'].astype(str))
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print("\nChi-Square Test of Independence: Season vs Crop")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Degrees of freedom: {dof}")
    return {
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'contingency_table': contingency,
        'expected_table': pd.DataFrame(expected, index=contingency.index, columns=contingency.columns),
    }


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare a machine-learning-ready dataset for yield prediction."""
    model_df = df.copy()

    # Keep the target and exclude non-predictive ID columns if present
    exclude_columns = {'Yield', 'Estimated_Revenue_INR'}

    # Convert categorical variables into dummy indicators
    for col in ['Season', 'Crop']:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype(str).str.strip()
    model_df = pd.get_dummies(model_df, columns=[col for col in ['Season', 'Crop'] if col in model_df.columns], dtype=float)

    feature_columns = [
        c for c in model_df.columns
        if c not in exclude_columns and pd.api.types.is_numeric_dtype(model_df[c])
    ]

    X = model_df[feature_columns].fillna(model_df[feature_columns].median())
    y = model_df['Yield'].fillna(model_df['Yield'].median())
    return X, y


def train_random_forest_model(df: pd.DataFrame):
    """Train a Random Forest Regressor and report the yield prediction metrics."""
    X, y = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regressor = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    feature_importance = pd.Series(regressor.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("\nRandom Forest Regressor for Yield Prediction")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print("\nTop Feature Importances:")
    print(feature_importance.head(10))

    metrics = {
        'RMSE': rmse,
        'R2': r2,
    }
    return metrics, feature_importance


def run_pipeline(csv_path: str = "data/raw_agmarknet.csv", output_path: str = "data/cleaned_agriculture_data.csv") -> pd.DataFrame:
    """Execute the complete data pipeline and save the cleaned dataset."""
    raw_df = load_raw_data(csv_path)
    cleaned_df = clean_dataset(raw_df)
    cleaned_df = engineer_features(cleaned_df)

    print("\nDataset loaded and cleaned successfully.")
    print(f"Rows after cleaning: {len(cleaned_df)}")

    pearson_correlation_analysis(cleaned_df)
    chi_square_test(cleaned_df)
    train_random_forest_model(cleaned_df)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output, index=False)
    print(f"\nCleaned dataset saved to: {output}")
    return cleaned_df


if __name__ == "__main__":
    run_pipeline()

    print("\nPipeline complete. Use the cleaned Agmarknet dataset in Tableau and the notebook for visualization.")
    print("Next step: open notebooks/01_dav_analysis.ipynb")






