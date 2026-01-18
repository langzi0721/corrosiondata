import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
import warnings

warnings.filterwarnings('ignore')

# 数据处理
file_path = r'data-path'
df = pd.read_excel(file_path, usecols=lambda col: col != 'DOI')

df['Model_ID'].fillna(df['Model_ID'].mode()[0], inplace=True)
numeric_cols = ['Cr', 'PREN', 'pH', 'Temperature_C', 'Cl_Concentration_gL', 'DO_Concentration_mgL']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median()).astype(float)

df_0 = df[df['Corrosion_Crevice_Occurred'] == 0].copy().reset_index(drop=True)
df_1 = df[df['Corrosion_Crevice_Occurred'] == 1].copy().reset_index(drop=True)

X_0 = df_0.drop('Corrosion_Crevice_Occurred', axis=1)
y_0 = df_0['Corrosion_Crevice_Occurred']
X_0_train, X_0_test, y_0_train, y_0_test = train_test_split(
    X_0, y_0, test_size=0.3, random_state=123, stratify=y_0
)

X_1 = df_1.drop('Corrosion_Crevice_Occurred', axis=1)
y_1 = df_1['Corrosion_Crevice_Occurred']
X_1_train, X_1_test, y_1_train, y_1_test = train_test_split(
    X_1, y_1, test_size=0.12, random_state=123, stratify=y_1
)

X_train = pd.concat([X_0_train, X_1_train], axis=0).reset_index(drop=True)
y_train = pd.concat([y_0_train, y_1_train], axis=0).reset_index(drop=True)
X_test = pd.concat([X_0_test, X_1_test], axis=0).reset_index(drop=True)
y_test = pd.concat([y_0_test, y_1_test], axis=0).reset_index(drop=True)

df_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
df_all_encoded = pd.get_dummies(df_all, columns=['Type', 'Model_ID'], drop_first=True)
X_train_encoded = df_all_encoded.iloc[:len(X_train), :].reset_index(drop=True)
X_test_encoded = df_all_encoded.iloc[len(X_train):, :].reset_index(drop=True)

numeric_features = ['Cr', 'PREN', 'pH', 'Temperature_C', 'Cl_Concentration_gL', 'DO_Concentration_mgL']
scaler = StandardScaler()
X_train_scaled = X_train_encoded.copy()
X_test_scaled = X_test_encoded.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train_encoded[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test_encoded[numeric_features])

X_full = pd.concat([X_train_scaled, X_test_scaled], axis=0)
y_full = pd.concat([y_train, y_test], axis=0)
X_train_arr = X_train_scaled.values
X_test_arr = X_test_scaled.values

# 初始化模型
RF_MODEL = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_split=2,
    class_weight='balanced', max_features='sqrt', random_state=123
)
SVM_MODEL = SVC(
    kernel='rbf', probability=True, C=0.8, gamma='scale',
    class_weight='balanced', random_state=123
)
MLP_MODEL = MLPClassifier(
    hidden_layer_sizes=(50, 20), activation='relu', solver='adam',
    max_iter=500, alpha=0.001, random_state=123
)
GB_MODEL = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    subsample=0.8, random_state=123
)
XGB_MODEL = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    use_label_encoder=False, eval_metric='logloss', random_state=123
)

MODELS = {
    'RF': RF_MODEL,
    'SVM': SVM_MODEL,
    'MLP': MLP_MODEL,
    'GB': GB_MODEL,
    'XGBoost': XGB_MODEL
}

# 打印模型参数
print("=" * 80)
print("Model Parameters Summary:")
print("=" * 80)
for name, model in MODELS.items():
    params = model.get_params()
    if name == 'RF':
        relevant_params = {k: params[k] for k in ['n_estimators', 'max_depth', 'class_weight', 'random_state']}
    elif name == 'SVM':
        relevant_params = {k: params[k] for k in ['C', 'kernel', 'class_weight', 'random_state']}
    elif name == 'MLP':
        relevant_params = {k: params[k] for k in
                           ['hidden_layer_sizes', 'activation', 'solver', 'max_iter', 'random_state']}
    elif name == 'GB':
        relevant_params = {k: params[k] for k in
                           ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'random_state']}
    elif name == 'XGBoost':
        relevant_params = {k: params[k] for k in
                           ['n_estimators', 'learning_rate', 'max_depth', 'eval_metric', 'random_state']}
    print(f"\n{name} Model Parameters:")
    for key, value in relevant_params.items():
        print(f"  {key}: {value}")
print("=" * 80)

# 5折交叉验证
print("\nPerforming 5-Fold Cross-Validation on 5 Models...")
print("=" * 80)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
all_cv_results = {}
rf_metrics_cv = {
    'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []
}

for name, model in MODELS.items():
    metrics_cv = {
        'Fold': [], 'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []
    }
    print(f"\nProcessing {name} model...")
    for fold, (train_index, val_index) in enumerate(skf.split(X_full, y_full)):
        X_cv_train, X_cv_val = X_full.iloc[train_index].values, X_full.iloc[val_index].values
        y_cv_train, y_cv_val = y_full.iloc[train_index], y_full.iloc[val_index]
        model.fit(X_cv_train, y_cv_train)
        y_cv_pred = model.predict(X_cv_val)
        acc = accuracy_score(y_cv_val, y_cv_pred)
        prec = precision_score(y_cv_val, y_cv_pred, zero_division=0)
        rec = recall_score(y_cv_val, y_cv_pred, zero_division=0)
        f1 = f1_score(y_cv_val, y_cv_pred, zero_division=0)
        metrics_cv['Fold'].append(fold + 1)
        metrics_cv['Model'].append(name)
        metrics_cv['Accuracy'].append(acc)
        metrics_cv['Precision'].append(prec)
        metrics_cv['Recall'].append(rec)
        metrics_cv['F1-Score'].append(f1)
        if name == 'RF':
            rf_metrics_cv['Accuracy'].append(acc)
            rf_metrics_cv['Precision'].append(prec)
            rf_metrics_cv['Recall'].append(rec)
            rf_metrics_cv['F1-Score'].append(f1)
        print(f"  Fold {fold + 1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    all_cv_results[name] = pd.DataFrame(metrics_cv)

# 交叉验证结果排序与打印
cv_mean_accuracy = {name: df['Accuracy'].mean() for name, df in all_cv_results.items()}
sorted_models = sorted(cv_mean_accuracy.items(), key=lambda item: item[1], reverse=True)
top_2_models_names = ['GB', 'RF']

print("\nCross-Validation Mean Accuracy Ranking:")
print("=" * 80)
for rank, (name, mean_acc) in enumerate(sorted(cv_mean_accuracy.items(), key=lambda item: item[1], reverse=True)):
    print(f"Rank {rank + 1}: {name} (Mean Accuracy: {mean_acc:.4f})")
print(f"\nDetailed analysis focuses on Top 2 Models: {top_2_models_names} (GB and RF)")
print("=" * 80)

# 随机森林交叉验证摘要
print("\nTable 1: 5-Fold CV Performance Summary (Random Forest Only)")
print("=" * 80)
rf_cv_df = pd.DataFrame(rf_metrics_cv)
cv_summary = rf_cv_df.describe().loc[['mean', 'std', 'min', 'max']]
cv_summary.index = ['Mean', 'Std', 'Min', 'Max']
print(cv_summary.to_string(float_format='%.4f'))
print("=" * 80)

# 多模型ROC曲线相关指标打印
print("\nMulti-Model ROC Curve Metrics on Testing Set...")
print("=" * 80)
roc_metrics = {}
for name, model in MODELS.items():
    model.fit(X_train_arr, y_train)
    y_test_proba = model.predict_proba(X_test_arr)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    roc_metrics[name] = roc_auc
    print(f"{name}: AUC = {roc_auc:.4f}")
print("=" * 80)

# 混淆矩阵与分类报告（Top2模型）
print(f"\nEvaluating Top 2 Models: {top_2_models_names}")
print("=" * 80)
for model_name in top_2_models_names:
    model = MODELS[model_name]
    model.fit(X_train_arr, y_train)
    y_train_pred = model.predict(X_train_arr)
    y_test_pred = model.predict(X_test_arr)

    # 训练集指标
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

    # 测试集指标
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print(f"\n{model_name} Training Set Metrics:")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  Precision: {train_prec:.4f}")
    print(f"  Recall: {train_rec:.4f}")
    print(f"  F1-Score: {train_f1:.4f}")
    print(f"\n{model_name} Testing Set Metrics:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall: {test_rec:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"\n{model_name} Testing Set Classification Report:")
    print(classification_report(y_test, y_test_pred))
    print("-" * 60)
print("=" * 80)

# 集成学习
print("\nEvaluating Ensemble Model: GB + RF Voting Classifier...")
print("=" * 80)
VOTE_MODEL = VotingClassifier(
    estimators=[('gb', GB_MODEL), ('rf', RF_MODEL)],
    voting='soft',
    weights=[1.0, 1.0],
    n_jobs=-1
)
VOTE_MODEL.fit(X_train_arr, y_train)
y_test_pred_vote = VOTE_MODEL.predict(X_test_arr)
vote_test_accuracy = accuracy_score(y_test, y_test_pred_vote)
vote_test_precision = precision_score(y_test, y_test_pred_vote, zero_division=0)
vote_test_recall = recall_score(y_test, y_test_pred_vote, zero_division=0)
vote_test_f1 = f1_score(y_test, y_test_pred_vote, zero_division=0)

print(f"Ensemble (GB+RF) Testing Accuracy: {vote_test_accuracy:.4f}")
print(f"Ensemble (GB+RF) Testing Precision: {vote_test_precision:.4f}")
print(f"Ensemble (GB+RF) Testing Recall: {vote_test_recall:.4f}")
print(f"Ensemble (GB+RF) Testing F1-Score: {vote_test_f1:.4f}")
print(f"Ensemble Testing Misclassifications: {sum(y_test != y_test_pred_vote)}")
print("\nEnsemble (GB+RF) Testing Set Classification Report:")
print(classification_report(y_test, y_test_pred_vote))
print("=" * 80)
print("\nAll analysis completed! All results are printed above.")