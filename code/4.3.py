import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------- Configuration --------------------------
FILE_PATH = r"data"

# Smart column name recognition
def get_col(df, keywords):
    for col in df.columns:
        if any(k.lower() in col.lower() for k in keywords):
            return col
    return None

# -------------------------- 1. Data Preprocessing --------------------------
# Load raw data
df_raw = pd.read_excel(FILE_PATH)
df_raw.columns = df_raw.columns.str.strip()

# Identify key columns
c_pren = get_col(df_raw, ['PREN'])
c_ph = get_col(df_raw, ['pH'])
c_temp = get_col(df_raw, ['Temp', 'Temperature'])
c_cl = get_col(df_raw, ['Cl', 'Chloride'])
c_oxy = get_col(df_raw, ['Oxy', 'Oxygen', 'DO_'])
c_target = get_col(df_raw, ['Target', 'Label', 'Corrosion'])

# Build feature matrix (exclude Cr)
features = [c_pren, c_ph, c_temp, c_cl, c_oxy]
X = df_raw[features].fillna(df_raw.median(numeric_only=True))
y = df_raw[c_target].astype(int)

# -------------------------- 2. Model Training & SHAP Weight Calculation --------------------------
# Train Random Forest model
rf_model_esi = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf_model_esi.fit(X, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(rf_model_esi)
shap_v = explainer.shap_values(X)
sv_corrosion = shap_v[1] if isinstance(shap_v, list) else shap_v[:, :, 1]

# Extract raw SHAP weights
w_map = dict(zip(X.columns, np.abs(sv_corrosion).mean(axis=0)))
ρ1 = w_map[c_ph]  # SHAP weight for pH
ρ2 = w_map[c_temp] # SHAP weight for Temperature
ρ3 = w_map[c_cl]   # SHAP weight for Chloride

# Normalize weights
sum_ρ = ρ1 + ρ2 + ρ3
m1 = ρ1 / sum_ρ  # Normalized coefficient for pH
m2 = ρ2 / sum_ρ  # Normalized coefficient for Temperature
m3 = ρ3 / sum_ρ  # Normalized coefficient for Chloride

# -------------------------- 3. Environmental Parameter Normalization --------------------------
def robust_norm(series, reverse=False):
    q10 = series.quantile(0.1)
    q90 = series.quantile(0.9)
    norm = (series - q10) / (q90 - q10)
    if reverse:
        norm = 1 - norm
    norm_clipped = norm.clip(0, 1)
    return norm_clipped

# Normalize environmental parameters
x1 = robust_norm(X[c_ph], reverse=True)  # pH normalization (reverse)
x2 = robust_norm(X[c_temp])              # Temperature normalization
x3 = robust_norm(X[c_cl])                # Chloride normalization

# Calculate ESI (Environmental Sensitivity Index)
esi = m1 * x1 + m2 * x2 + m3 * x3

# Group ESI values
bins = [0, 0.3, 0.6, 1.0]
labels = ['Mild', 'Moderate', 'Severe']
esi_groups = pd.cut(esi, bins=bins, labels=labels, include_lowest=True)

# Calculate PREN contribution ratio
pren_ratios = []
pren_idx = list(X.columns).index(c_pren)
for label in labels:
    mask = (esi_groups == label)
    if mask.any():
        mean_impacts = np.abs(sv_corrosion[mask]).mean(axis=0)
        ratio = (mean_impacts[pren_idx] / np.sum(mean_impacts)) * 100
        pren_ratios.append(ratio)
    else:
        pren_ratios.append(0)

# -------------------------- 4. 3D Probability Plot Model (Extended) --------------------------
# Parameter setup
ENGLISH_LABELS = {
    'Cr': 'Cr (%)',
    'PREN': 'PREN',
    'pH': 'PH',
    'Temperature_C': 'Temperature',
    'Cl_Concentration_gL': 'Cl⁻',
    'DO_Concentration_mgL': 'Oxygen',
}
numeric_cols = ['Cr', 'PREN', 'pH', 'Temperature_C', 'Cl_Concentration_gL', 'DO_Concentration_mgL']

BASE_VALUES = {
    'pH': 7.0,
    'Temperature_C': 20.0,
    'Cl_Concentration_gL': 5.0,
    'DO_Concentration_mgL': 0.0,
    'PREN': None,
}

# Data preprocessing for 3D model
df = pd.read_excel(FILE_PATH, usecols=lambda col: col != 'DOI')
# Fix FutureWarning: replace inplace=True with direct assignment
df['Model_ID'] = df['Model_ID'].fillna(df['Model_ID'].mode()[0])

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median()).astype(float)
BASE_VALUES['PREN'] = df['PREN'].median()

# Stratified sampling
df_0 = df[df['Corrosion_Crevice_Occurred'] == 0].copy().reset_index(drop=True)
df_1 = df[df['Corrosion_Crevice_Occurred'] == 1].copy().reset_index(drop=True)
X_0 = df_0.drop('Corrosion_Crevice_Occurred', axis=1)
y_0 = df_0['Corrosion_Crevice_Occurred']
X_0_train, _, y_0_train, _ = train_test_split(X_0, y_0, test_size=0.3, random_state=123, stratify=y_0)
X_1 = df_1.drop('Corrosion_Crevice_Occurred', axis=1)
y_1 = df_1['Corrosion_Crevice_Occurred']
X_1_train, _, y_1_train, _ = train_test_split(X_1, y_1, test_size=0.12, random_state=123, stratify=y_1)

X_train = pd.concat([X_0_train, X_1_train], axis=0).reset_index(drop=True)
y_train = pd.concat([y_0_train, y_1_train], axis=0).reset_index(drop=True)

df_all = pd.concat([X_train, df_0, df_1], axis=0).reset_index(drop=True).drop('Corrosion_Crevice_Occurred', axis=1)
df_all_encoded = pd.get_dummies(df_all, columns=['Type', 'Model_ID'], drop_first=True)
X_train_encoded = df_all_encoded.iloc[:len(X_train), :].reset_index(drop=True)

# Standardization
scaler = StandardScaler()
X_train_scaled = X_train_encoded.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_encoded[numeric_cols])

# Train Random Forest for 3D probability plots
rf_model_3d = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_split=2,
    class_weight='balanced', max_features='sqrt', random_state=123
)
rf_model_3d.fit(X_train_scaled.values, y_train)

# -------------------------- Output All Calculation Results --------------------------
# ESI Weight & Coefficient Results
print("=== ESI Calculation Results ===")
print(f"Raw SHAP Weights: ρ1(pH)={ρ1:.4f}, ρ2(Temperature)={ρ2:.4f}, ρ3(Chloride)={ρ3:.4f}")
print(f"Sum of Raw Weights: sum_ρ={sum_ρ:.4f}")
print(f"Normalized Coefficients: m1(pH)={m1:.4f}, m2(Temperature)={m2:.4f}, m3(Chloride)={m3:.4f}")
print(f"Sum of Normalized Coefficients: m1+m2+m3={m1+m2+m3:.4f} (should equal 1)")
print(f"ESI Value Range: Min={esi.min():.4f}, Max={esi.max():.4f} (should be in [0,1])")

# ESI Group Statistics
print("\n=== ESI Group Statistics ===")
group_counts = esi_groups.value_counts()
for label in labels:
    print(f"{label} (ESI ∈ {['0.00-0.30', '0.30-0.60', '0.60-1.00'][labels.index(label)]}): {group_counts.get(label, 0)} samples")

# PREN Contribution Ratio Results
print("\n=== PREN Contribution Ratio by ESI Regime ===")
for label, ratio in zip(labels, pren_ratios):
    print(f"{label} Environment (ESI ∈ {['0.00-0.30', '0.30-0.60', '0.60-1.00'][labels.index(label)]}): PREN Contribution Ratio = {ratio:.1f}%")

# -------------------------- Plotting Instructions  --------------------------
print("\n=== Plotting Instructions ===")
print("1. PREN Contribution Ratio Bar Plot:")
print("   - Plot type: Bar chart with 3 bars (Mild/Moderate/Severe ESI regimes)")
print("   - X-axis: ESI Regimes (Mild, Moderate, Severe)")
print("   - Y-axis: PREN Contribution Ratio (%)")
print("   - Recommended style: Colors (#43A047, #FB8C00, #E53935), black edge, width=0.5")
print("   - Add value labels on top of each bar (1 decimal place)")
print("   - Title: 'Impact of PREN Across ESI Regimes (Cr Removed)'")
print("   - Font: Times New Roman, Title fontsize=22, Axis labels fontsize=20, Ticks fontsize=16")

print("\n2. 3D Probability Plots (RF Model):")
print("   - Plot pairs: (pH vs PREN), (Temperature vs PREN), (Cl⁻ vs PREN), (Oxygen vs PREN)")
print("   - Plot type: 3D surface plot (cmap='RdYlBu_r', alpha=0.8)")
print("   - Axes: X=Environmental Parameter, Y=PREN, Z=Corrosion Probability")
print("   - Recommended size: figsize=(7,5), dpi=400")
print("   - Font: Times New Roman, Label fontsize=12, Tick fontsize=10")
print("   - View angle: elev=20, azim=45")
print("   - Add colorbar (shrink=0.5, aspect=12, pad=0.1)")