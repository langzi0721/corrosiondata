# File name: SHAP_Combined_Analysis.py
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore unnecessary warnings for cleaner output
warnings.filterwarnings('ignore')

# -------------------------- Global Configuration --------------------------
# Feature range constraints for data filtering
RANGES = {
    'Temp': (20, 80),
    'Cl': (5, 35),
    'PREN': (10, 36),
    'pH': (5, 9),
    'O': (0, 8)
}

# Feature name definitions
FEATURE_NAMES = ['PREN', 'pH', 'Temp', 'Cl', 'O']
SUMMARY_FEATURE_LABELS = ["PREN", "pH", "Temperature", "Cl⁻", "Oxygen"]


# -------------------------- Data Loading & Preprocessing --------------------------
def load_and_preprocess_data(file_path):
    """
    Load dataset from Excel file and perform basic preprocessing
    Args:
        file_path: Path to the Excel data file
    Returns:
        X_scaled_full: Standardized feature matrix
        X_raw_full: Original feature matrix (before scaling)
        y_full: Target variable array
    """
    # Load data and keep only numerical columns (skip first 2 columns)
    df_full = pd.read_excel(file_path).iloc[:, 2:].select_dtypes(include=['float64', 'int64'])

    # Separate features and target, fill missing values with median
    X_raw_full = df_full.iloc[:, 1:-1].fillna(df_full.median()).astype(float)
    y_full = df_full.iloc[:, -1].astype(int)
    X_raw_full.columns = FEATURE_NAMES

    # Standardize features
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X_raw_full)

    return X_scaled_full, X_raw_full, y_full


# -------------------------- Model Training --------------------------
def train_random_forest(X, y):
    """
    Train Random Forest Classifier for SHAP analysis
    Args:
        X: Feature matrix
        y: Target variable
    Returns:
        Trained Random Forest model
    """
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=123,
        max_depth=None,
        min_samples_split=2
    )
    rf_model.fit(X, y)
    return rf_model


# -------------------------- SHAP Value Calculation --------------------------
def calculate_shap_values(model, X):
    """
    Calculate SHAP values for the trained model
    Args:
        model: Trained Random Forest model
        X: Feature matrix
    Returns:
        SHAP values for class 1 (corrosion occurrence)
    """
    explainer = shap.TreeExplainer(model)
    shap_values_full = explainer.shap_values(X)
    # For classification models, extract SHAP values for class 1 (crevice corrosion occurrence)
    shap_corrosion_full = shap_values_full[1] if isinstance(shap_values_full, list) else shap_values_full[:, :, 1]
    return shap_corrosion_full


# -------------------------- Data Filtering --------------------------
def filter_data_by_ranges(X_raw, shap_vals):
    """
    Filter data based on predefined feature ranges
    Args:
        X_raw: Original feature matrix
        shap_vals: SHAP values array
    Returns:
        Filtered feature matrix and corresponding SHAP values
    """
    mask = np.ones(len(X_raw), dtype=bool)
    for col, (min_val, max_val) in RANGES.items():
        if col in X_raw.columns:
            mask &= (X_raw[col] >= min_val) & (X_raw[col] <= max_val)

    X_filtered = X_raw[mask].reset_index(drop=True)
    shap_filtered = shap_vals[mask]
    return X_filtered, shap_filtered


# -------------------------- Main Execution --------------------------
if __name__ == "__main__":
    # Replace with your actual file path when running
    FILE_PATH = r"data"

    # Step 1: Load and preprocess data
    X_scaled_full, X_raw_full, y_full = load_and_preprocess_data(FILE_PATH)

    # Step 2: Train model
    rf_model = train_random_forest(X_scaled_full, y_full)

    # Step 3: Calculate SHAP values
    shap_corrosion_full = calculate_shap_values(rf_model, X_scaled_full)

    # Step 4: Filter data
    X_plot, shap_plot = filter_data_by_ranges(X_raw_full, shap_corrosion_full)

    # -------------------------- Key Information Output --------------------------
    # Print basic data information
    print("=== Basic Data Information ===")
    print(f"Original data rows: {len(X_raw_full)}")
    print(f"Filtered data rows: {len(X_plot)}")
    print(f"Feature list: {FEATURE_NAMES}")

    # Print model parameters
    print("\n=== Model Parameters ===")
    print(f"Random Forest n_estimators: {rf_model.n_estimators}")
    print(f"Class weight: {rf_model.class_weight}")
    print(f"Random state: {rf_model.random_state}")

    # Print SHAP value statistics (Class 1)
    print("\n=== SHAP Value Statistics (Class 1) ===")
    print(f"Overall SHAP mean: {np.mean(shap_corrosion_full):.4f}")
    print(f"Overall SHAP std: {np.std(shap_corrosion_full):.4f}")
    print(f"Filtered SHAP mean: {np.mean(shap_plot):.4f}")
    print(f"Filtered SHAP std: {np.std(shap_plot):.4f}")

    # Print feature-wise SHAP statistics (filtered data)
    print("\n=== Feature-wise SHAP Statistics (Filtered) ===")
    for idx, feat in enumerate(FEATURE_NAMES):
        feat_shap = shap_plot[:, idx]
        print(f"{feat}: Mean={np.mean(feat_shap):.4f}, Min={np.min(feat_shap):.4f}, Max={np.max(feat_shap):.4f}")

    # -------------------------- Plotting Placeholder --------------------------
    # Plotting logic will be implemented here:
    # 1. SHAP dependence plots
    # 2. SHAP summary plots

    print("\n✅ Data processing and SHAP value calculation completed. Key information printed to console.")