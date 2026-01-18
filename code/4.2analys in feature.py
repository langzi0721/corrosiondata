import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Global Configuration
FILE_PATH = r'data-path'

# Data Reading and Preprocessing
try:
    df = pd.read_excel(FILE_PATH, usecols=lambda col: col != 'DOI')
    df.columns = df.columns.str.strip()

    print("=== Basic Information of Raw Data ===")
    print(f"Data shape: {df.shape}")
    print(f"Data columns: {list(df.columns)}")
    print("\n=== First 5 Rows of Raw Data ===")
    print(df.head())


    def get_col_name(keywords, columns):
        for col in columns:
            if any(k.lower() in col.lower() for k in keywords):
                return col
        return None


    cols_map = {
        'PREN': get_col_name(['PREN'], df.columns),
        'pH': get_col_name(['pH'], df.columns),
        'Temp': get_col_name(['Temp', 'Temperature'], df.columns),
        'Cl': get_col_name(['Cl', 'Chloride'], df.columns),
        'Oxy': get_col_name(['Oxy', 'Oxygen', 'DO_'], df.columns),
        'Target': get_col_name(['Corrosion', 'Target', 'Label'], df.columns)
    }

    print("\n=== Matched Feature Columns ===")
    for key, val in cols_map.items():
        print(f"{key}: {val}")

    plot_numeric_cols = [cols_map['PREN'], cols_map['pH'], cols_map['Temp'], cols_map['Cl'], cols_map['Oxy']]
    for col in plot_numeric_cols:
        df[col] = df[col].fillna(df[col].median()).astype(float)

    X_cols = plot_numeric_cols
    X = df[X_cols].fillna(df.median(numeric_only=True))
    y = df[cols_map['Target']].astype(int)

    print("\n=== Basic Information of Feature Matrix X ===")
    print(f"X shape: {X.shape}")
    print(f"Missing values in X:\n{X.isnull().sum()}")
    print(f"\nDescriptive Statistics of X:\n{X.describe()}")

    print("\n=== Basic Information of Target Variable y ===")
    print(f"y shape: {y.shape}")
    print(f"Value distribution of y:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("\n=== Data Split Results ===")
    print(f"Training set X: {X_train.shape}, Test set X: {X_test.shape}")
    print(f"Training set y: {y_train.shape}, Test set y: {y_test.shape}")
    print(f"Mean of scaled training set (per feature): {np.round(X_train_scaled.mean(axis=0), 4)}")
    print(f"Standard deviation of scaled training set (per feature): {np.round(X_train_scaled.std(axis=0), 4)}")

except Exception as e:
    print(f"Error in data processing: {e}")
    exit()

# Correlation Analysis
print("\n=== Correlation Matrix Calculation Results ===")
SIMPLIFIED_LABELS = {
    cols_map['PREN']: 'PREN',
    cols_map['pH']: 'pH',
    cols_map['Temp']: 'Temperature',
    cols_map['Cl']: 'Cl⁻',
    cols_map['Oxy']: 'Oxygen',
    cols_map['Target']: 'Corrosion'
}
df_corr_plot = df[plot_numeric_cols + [cols_map['Target']]].copy()
df_corr_plot.rename(columns=SIMPLIFIED_LABELS, inplace=True)

methods = {
    'Pearson': 'pearson',
    'Spearman': 'spearman',
    'Kendall': 'kendall'
}
for name, method_val in methods.items():
    corr_matrix = df_corr_plot.corr(method=method_val)
    print(f"\n--- {name} Correlation Matrix ---")
    print(corr_matrix.round(4))

# Model Training and Feature Importance
print("\n=== Model Training and Feature Importance ===")
rf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=123)
rf_model.fit(X_train_scaled, y_train)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=123)
gb_model.fit(X_train_scaled, y_train)

display_labels = ['PREN', 'pH', 'Temperature', 'Cl⁻', 'Oxygen']
models = {'Random Forest (RF)': rf_model, 'Gradient Boosting (GB)': gb_model}

for name, m in models.items():
    importances = m.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': display_labels,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    print(f"\n--- {name} Feature Importance (Descending Order) ---")
    print(feat_imp.round(4))

print("\n✅ All data processing, correlation analysis, and model training completed! All results printed to console.")