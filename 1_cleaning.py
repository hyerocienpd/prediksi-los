import os
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
df = pd.read_excel('data/rekam_medis.xlsx')
print(f"Data awal: {len(df)} baris")

# ── STANDARISASI NAMA KOLOM ────────────────────────────────────────────────────
df.columns = df.columns.str.strip()

# Standarisasi nama kolom sesuai kebutuhan pemodelan
df = df.rename(columns={
    'tglmasuk': 'tgl_masuk',
    'tglkeluar': 'tgl_keluar',
    'usia': 'usia_raw',
    'kode_ICD': 'kode_diagnosis',
    'ruang': 'ruang_perawatan',
})

# Menghapus spasi tambahan (padding) pada fitur bertipe teks
kolom_object = df.select_dtypes(include='object').columns
for col in kolom_object:
    df[col] = df[col].astype(str).str.strip()

# ── FILTER: HANYA PASIEN PULANG ATAS PERSETUJUAN MEDIS ────────────────────────
# Nilai asli: '1. Atas Persetujuan Medis'
df = df[df['status_keluar'].str.contains('Atas Persetujuan Medis', case=False, na=False)]
print(f"Setelah filter status keluar: {len(df)} baris")

# ── HITUNG LENGTH OF STAY ──────────────────────────────────────────────────────
df['tgl_masuk']  = pd.to_datetime(df['tgl_masuk'],  errors='coerce')
df['tgl_keluar'] = pd.to_datetime(df['tgl_keluar'], errors='coerce')
df['los'] = (df['tgl_keluar'] - df['tgl_masuk']).dt.days

# Hapus baris dengan tanggal tidak valid atau LoS negatif
df = df[df['los'].notna() & (df['los'] >= 0)]

# Hapus anomali LoS (batas maksimum wajar: 19 hari sesuai data RSU Aulia)
df = df[df['los'] <= 19]
print(f"Setelah filter anomali tanggal & LoS: {len(df)} baris")

# ── DEDUPLIKASI KUNJUNGAN (ASUMSI DIAGNOSIS UTAMA) ───────────────────────────
# Pada kunjungan dengan >1 kode diagnosis (sekitar 2,8% dari total kunjungan),
# diambil baris pertama sesuai urutan input SIMRS. Pendekatan ini mengasumsikan 
# bahwa diagnosis pertama yang tercatat merepresentasikan diagnosis utama 
# penyebab rawat inap, sebuah praktik pencatatan yang umum di RS Indonesia.
df = df.sort_values('no_registrasi')
df = df.drop_duplicates(subset='no_registrasi', keep='first')
print(f"Setelah deduplikasi: {len(df)} baris")

# ── HAPUS MISSING VALUES ───────────────────────────────────────────────────────
df = df.dropna(subset=['ruang_perawatan', 'jenis_perawatan', 'cara_datang'])
print(f"Setelah hapus missing values: {len(df)} baris")

# ── FEATURE ENGINEERING: USIA ──────────────────────────────────────────────────
# Konversi usia dari string (format: '31 Th 4 Bl 13 Hr') ke angka tahun
# Fungsi untuk mengekstraksi dan mengonversi format teks 'Th Bl Hr' menjadi desimal tahun

def parse_usia(usia_str):
    if pd.isna(usia_str):
        return np.nan
    usia_str = str(usia_str).strip()
    tahun = re.search(r'(\d+)\s*Th', usia_str, re.IGNORECASE)
    bulan = re.search(r'(\d+)\s*Bl', usia_str, re.IGNORECASE)
    hari  = re.search(r'(\d+)\s*Hr', usia_str, re.IGNORECASE)
    t = int(tahun.group(1)) if tahun else 0
    b = int(bulan.group(1)) / 12 if bulan else 0
    h = int(hari.group(1)) / 365 if hari else 0
    if not tahun and not bulan and not hari:
        try:
            return float(usia_str)
        except:
            return np.nan
    return round(t + b + h, 4)

df['usia'] = df['usia_raw'].apply(parse_usia)
df = df[df['usia'].notna()]
print(f"Setelah hapus kegagalan parsing usia: {len(df)} baris")

# ── FEATURE ENGINEERING: KODE DIAGNOSIS ICD-10 ────────────────────────────────
# Kelompokkan berdasarkan huruf pertama kode (bab ICD-10)
df['kode_diagnosis'] = df['kode_diagnosis'].astype(str).str.strip()
df['bab_diagnosis']  = df['kode_diagnosis'].str[0].str.upper()

# ── FEATURE ENGINEERING: HARI MASUK ────────────────────────────────────────────
# Mengetahui hari masuk pasien (memungkinkan deteksi perbedaan LoS di akhir pekan)
indo_days = {'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu', 'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'}
df['hari_masuk'] = df['tgl_masuk'].dt.day_name().map(indo_days)

# ── LABEL TARGET: KATEGORI LoS ────────────────────────────────────────────────
def kategori_los(los):
    if los <= 2:
        return 0   # Pendek
    elif los <= 5:
        return 1   # Sedang
    else:
        return 2   # Panjang

df['los_kategori'] = df['los'].apply(kategori_los)

# ── STATISTIK DESKRIPTIF UNTUK BAB 1 / BAB 3 ──────────────────────────────────
print("\n=== STATISTIK DESKRIPTIF ===")
print(f"Jumlah rekaman bersih akhir : {len(df)} baris")
print(f"Rata-rata LoS               : {round(df['los'].mean(), 2)} hari")
print(f"Median LoS                  : {df['los'].median()} hari")
print(f"Rentang LoS                 : {df['los'].min()} - {df['los'].max()} hari")
print(f"Jumlah kasus LoS maksimum   : {(df['los'] == df['los'].max()).sum()} kasus")

print("\nDistribusi kategori LoS (jumlah):")
print(df['los_kategori'].value_counts().rename({0: 'Pendek', 1: 'Sedang', 2: 'Panjang'}))
print("\nDistribusi kategori LoS (%):")
print((df['los_kategori'].value_counts(normalize=True) * 100).round(2).rename({0: 'Pendek', 1: 'Sedang', 2: 'Panjang'}))

print("\nJumlah kasus per bab diagnosis (ICD-10, huruf pertama kode):")
print(df['bab_diagnosis'].value_counts())
print(f"\n  -> Bab 'P' (perinatal/neonatal): {(df['bab_diagnosis'] == 'P').sum()} kasus")

# ── SIMPAN HASIL CLEANING ──────────────────────────────────────────────────────
KOLOM_FINAL = [
    'rm', 'no_registrasi', 'tgl_masuk', 'usia', 'jenis_kelamin',
    'kode_diagnosis', 'Diagnosa', 'bab_diagnosis',
    'ruang_perawatan', 'jenis_perawatan', 'cara_datang',
    'hari_masuk', 'los', 'los_kategori'
]
df_clean = df[KOLOM_FINAL].copy()

os.makedirs('data', exist_ok=True)
df_clean.to_csv('data/data_clean.csv', index=False)
print(f"\nData bersih disimpan: data/data_clean.csv ({len(df_clean)} baris)")

# ── VERIFIKASI KODE DIAGNOSIS ──────────────
top_diagnosis = df['kode_diagnosis'].value_counts().head(10)
print("\nTop 10 kode_diagnosis (dataset bersih, n=5.390):")
print(top_diagnosis)