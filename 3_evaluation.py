import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support
)
import warnings
import json
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)

LABEL_NAMA  = ['Pendek (≤2 hr)', 'Sedang (3-5 hr)', 'Panjang (>5 hr)']
LABEL_SHORT = ['Pendek', 'Sedang', 'Panjang']

# ── LOAD DATA TEST & MODEL ─────────────────────────────────────────────────────
X_test     = pd.read_csv('data/X_test.csv')
y_test     = pd.read_csv('data/y_test.csv').squeeze()
model_xgb  = joblib.load('dashboard/model_xgb.pkl')
model_rf   = joblib.load('dashboard/model_rf.pkl')
nama_terbaik = joblib.load('dashboard/nama_model.pkl')
FITUR        = joblib.load('dashboard/fitur.pkl')

# ── FUNGSI EVALUASI ────────────────────────────────────────────────────────────
def evaluasi_model(model, nama, X_test, y_test):
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    p_cls, r_cls, f_cls, support_cls = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
    )
    per_kelas = {
        LABEL_SHORT[i]: {
            'precision': round(float(p_cls[i]), 4),
            'recall'   : round(float(r_cls[i]), 4),
            'f1'       : round(float(f_cls[i]), 4),
            'support'  : int(support_cls[i]),
        } for i in range(3)
    }

    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {nama}")
    print(f"{'='*50}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Precision (w)  : {prec:.4f}")
    print(f"  Recall (w)     : {rec:.4f}")
    print(f"  F1-Score (w)   : {f1:.4f}")
    print(f"  F1-Score (macro): {f1_macro:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=LABEL_SHORT, zero_division=0)}")
    print(f"  >> Recall kelas 'Panjang': {per_kelas['Panjang']['recall']:.4f} "
          f"(dari {per_kelas['Panjang']['support']} kasus aktual)")
    if per_kelas['Panjang']['recall'] < 0.6:
        print(f"  [PERINGATAN] Recall kelas 'Panjang' di bawah 60% — model sering "
              f"'melewatkan' pasien LOS panjang. Pertimbangkan tuning ulang atau "
              f"fitur tambahan sebelum digunakan untuk perencanaan kapasitas.")

    return y_pred, {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'f1_macro': round(float(f1_macro), 4),
        'per_kelas': per_kelas,
    }

pred_xgb, metrik_xgb = evaluasi_model(model_xgb, 'XGBoost', X_test, y_test)
pred_rf,  metrik_rf  = evaluasi_model(model_rf,  'Random Forest', X_test, y_test)

# ── CONFUSION MATRIX — absolut & ternormalisasi ───────────────────────────────
cm_all = {}
fig, axes = plt.subplots(2, 2, figsize=(18, 13))

for col_idx, (y_pred, nama) in enumerate(zip([pred_xgb, pred_rf], ['XGBoost', 'Random Forest'])):
    cm_abs = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    sns.heatmap(cm_abs, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_SHORT, yticklabels=LABEL_SHORT, ax=axes[0, col_idx])
    axes[0, col_idx].set_title(f'Confusion Matrix (Jumlah) — {nama}')
    axes[0, col_idx].set_xlabel('Prediksi')
    axes[0, col_idx].set_ylabel('Aktual')

    cm_norm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2], normalize='true')
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=LABEL_SHORT, yticklabels=LABEL_SHORT, ax=axes[1, col_idx])
    axes[1, col_idx].set_title(f'Confusion Matrix (% per kelas aktual) — {nama}')
    axes[1, col_idx].set_xlabel('Prediksi')
    axes[1, col_idx].set_ylabel('Aktual')
    
    cm_all[nama] = cm_abs.tolist()

plt.tight_layout()
plt.savefig('data/confusion_matrix.png', dpi=200)
print("\nConfusion matrix disimpan: data/confusion_matrix.png")

with open('data/cm.json', 'w') as f:
    json.dump(cm_all, f, indent=2)
print("Confusion matrix arrays disimpan: data/cm.json")

# ── TABEL PERBANDINGAN ─────────────────────────────────────────────────────────
df_compare = pd.DataFrame({
    'Metrik'       : ['Accuracy', 'Precision (weighted)', 'Recall (weighted)',
                       'F1-Score (weighted)', 'F1-Score (macro)'],
    'XGBoost'      : [metrik_xgb['accuracy'], metrik_xgb['precision'],
                      metrik_xgb['recall'],   metrik_xgb['f1'], metrik_xgb['f1_macro']],
    'Random Forest': [metrik_rf['accuracy'],  metrik_rf['precision'],
                      metrik_rf['recall'],    metrik_rf['f1'], metrik_rf['f1_macro']],
})
df_compare = df_compare.round(4)
print(f"\n{df_compare.to_string(index=False)}")
print(f"\nModel terbaik (dari tahap training): {nama_terbaik}")

metrik_all = {'XGBoost': metrik_xgb, 'Random Forest': metrik_rf}
with open('data/metrik.json', 'w') as f:
    json.dump(metrik_all, f, indent=2)
print("Metrik (termasuk breakdown per kelas) disimpan: data/metrik.json")

# ── SHAP: FEATURE IMPORTANCE YANG LEBIH ANDAL ─────────────────────────────────
print("\n[SHAP] Menghitung SHAP values untuk XGBoost & Random Forest...")

X_shap = X_test.sample(n=min(500, len(X_test)), random_state=42) if len(X_test) > 500 else X_test

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

shap_summary = {}

for ax, model, nama in zip(axes, [model_xgb, model_rf], ['XGBoost', 'Random Forest']):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_shap)

    # Penyesuaian kompatibilitas versi pustaka SHAP untuk model multikelas.
    # Penyatuan array (stacking) dilakukan agar struktur keluaran shap_values()
    # selalu konsisten berbentuk (n_sampel, n_fitur, n_kelas) terlepas dari versi
    # pustaka yang digunakan (lama vs baru), guna memfasilitasi proses rata-rata 
    # (averaging) lintas kelas.
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)  # -> (n_sampel, n_fitur, n_kelas)

    mean_abs_shap = np.abs(sv).mean(axis=(0, 2))
    urutan = np.argsort(mean_abs_shap)

    ax.barh(np.array(FITUR)[urutan], mean_abs_shap[urutan], color='#1565c0')
    ax.set_title(f'SHAP Feature Importance — {nama}')
    ax.set_xlabel('Mean(|SHAP value|) — rata-rata lintas 3 kelas')

    shap_summary[nama] = {
        FITUR[i]: round(float(mean_abs_shap[i]), 5) for i in range(len(FITUR))
    }

plt.tight_layout()
plt.savefig('data/shap_summary.png', dpi=200)
print("SHAP summary plot disimpan: data/shap_summary.png")

with open('data/shap_summary.json', 'w') as f:
    json.dump(shap_summary, f, indent=2)
print("Nilai SHAP (mean |value| per fitur) disimpan: data/shap_summary.json")

for nama in ['XGBoost', 'Random Forest']:
    urutan_shap = sorted(shap_summary[nama], key=shap_summary[nama].get, reverse=True)
    print(f"  [{nama}] Urutan fitur menurut SHAP: {urutan_shap}")
