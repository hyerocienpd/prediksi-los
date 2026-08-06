import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
import warnings
warnings.filterwarnings('ignore')

os.makedirs('dashboard', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ── LOAD DATA BERSIH ───────────────────────────────────────────────────────────
df = pd.read_csv('data/data_clean.csv')
print(f"Data dimuat: {len(df)} baris")

FITUR = ['usia', 'jenis_kelamin', 'bab_diagnosis',
         'ruang_perawatan', 'jenis_perawatan', 'cara_datang', 'hari_masuk']
TARGET = 'los_kategori'
kolom_kategorikal = ['jenis_kelamin', 'bab_diagnosis', 'ruang_perawatan',
                      'jenis_perawatan', 'cara_datang', 'hari_masuk']
idx_kategorikal = [FITUR.index(c) for c in kolom_kategorikal]

X = df[FITUR].copy()
y = df[TARGET]

# ── SPLIT DATA DULU: 80% TRAIN, 20% TEST (STRATIFIED, GROUPED BY RM) ───────────
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf.split(X, y, groups=df['rm']))

X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = df['rm'].iloc[train_idx]
print(f"Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")

print("\nDistribusi kelas (train):")
print(y_train.value_counts().sort_index())

# ── LABEL ENCODING ─────────────────────────────────────────────────────────────
# Rasionalisasi:
# Label Encoding diterapkan pada seluruh fitur kategorikal (termasuk hari_masuk)
# untuk memenuhi arsitektur SMOTENC yang mensyaratkan fitur kategorikal 
# direpresentasikan dalam indeks kolom tunggal. One-Hot Encoding dihindari 
# karena berisiko menghasilkan sampel sintetik dengan kombinasi dummy yang 
# tidak valid secara matematis. Asumsi ordinal palsu pada model berbasis pohon 
# dinilai tidak signifikan mengingat mekanisme pemisahan simpul (node splitting) 
# tidak bertumpu pada jarak metrik linear.
encoders = {}
X_train = X_train_raw.copy()
X_test = X_test_raw.copy()

for col in kolom_kategorikal:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))

    test_vals = X_test[col].astype(str)
    known_classes = set(le.classes_)
    mask_unknown = ~test_vals.isin(known_classes)
    
    encoded = np.full(len(test_vals), -1, dtype=int)
    encoded[~mask_unknown] = le.transform(test_vals[~mask_unknown])
    X_test[col] = encoded

    encoders[col] = le

joblib.dump(encoders, 'dashboard/encoders.pkl')
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)


# ── SETUP SMOTENC ────────────────────────────────────────────────────────────
# Menentukan target oversampling untuk kelas minoritas ("Panjang" / kelas 2)
n_panjang = (y_train == 2).sum()
target_smote = int(n_panjang * 2.5) # Menggandakan proporsi kelas minoritas

smotenc = SMOTENC(categorical_features=idx_kategorikal, random_state=42, sampling_strategy={2: target_smote})
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# ── MODEL 1: XGBOOST PIPELINE ─────────────────────────────────────────────────
print("\n[XGBoost] Mulai Grid Search dengan SMOTENC Pipeline...")
pipeline_xgb = ImbPipeline([
    ('smotenc', smotenc),
    ('model', XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', random_state=42))
])

param_grid_xgb = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
}

grid_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)

# Fit pipeline
grid_xgb.fit(X_train, y_train, groups=groups_train)

print(f"XGBoost best params  : {grid_xgb.best_params_}")
print(f"XGBoost best CV F1(m): {grid_xgb.best_score_:.4f}")
model_xgb = grid_xgb.best_estimator_.named_steps['model']

# ── MODEL 2: RANDOM FOREST PIPELINE ───────────────────────────────────────────
print("\n[Random Forest] Mulai Grid Search dengan SMOTENC Pipeline...")
pipeline_rf = ImbPipeline([
    ('smotenc', smotenc),
    ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
])

param_grid_rf = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5],
}

grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train, groups=groups_train)

print(f"Random Forest best params  : {grid_rf.best_params_}")
print(f"Random Forest best CV F1(m): {grid_rf.best_score_:.4f}")
model_rf = grid_rf.best_estimator_.named_steps['model']

# ── CEK OVERFITTING & EVALUASI ───────────────────────────────────────────────
def cek_overfitting(model, nama, X_train, y_train, X_test, y_test):
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test  = accuracy_score(y_test, model.predict(X_test))
    f1_train  = f1_score(y_train, model.predict(X_train), average='macro', zero_division=0)
    f1_test   = f1_score(y_test, model.predict(X_test), average='macro', zero_division=0)

    gap_acc = acc_train - acc_test
    gap_f1  = f1_train - f1_test

    print(f"\n[{nama}] Train acc: {acc_train:.4f} | Test acc: {acc_test:.4f} | Gap acc: {gap_acc:.4f}")
    print(f"[{nama}] Train F1(macro): {f1_train:.4f} | Test F1(macro): {f1_test:.4f} | Gap F1(macro): {gap_f1:.4f}")
    return acc_test, f1_test

acc_xgb, f1m_xgb = cek_overfitting(grid_xgb.best_estimator_, 'XGBoost', X_train, y_train, X_test, y_test)
acc_rf,  f1m_rf  = cek_overfitting(grid_rf.best_estimator_,  'Random Forest', X_train, y_train, X_test, y_test)

# ── PILIH MODEL TERBAIK ──────────────────────────────────────────────────────
if f1m_xgb >= f1m_rf:
    model_terbaik = model_xgb
    nama_terbaik  = 'XGBoost'
else:
    model_terbaik = model_rf
    nama_terbaik  = 'Random Forest'

print(f"\nModel terbaik (berdasarkan F1-macro di test set): {nama_terbaik}")

joblib.dump(model_xgb,    'dashboard/model_xgb.pkl')
joblib.dump(model_rf,     'dashboard/model_rf.pkl')
joblib.dump(model_terbaik,'dashboard/model.pkl')
joblib.dump(nama_terbaik, 'dashboard/nama_model.pkl')
joblib.dump(FITUR,        'dashboard/fitur.pkl')
print("Model disimpan di folder dashboard/")
